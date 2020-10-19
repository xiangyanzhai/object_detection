# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import numpy as np
import torch.nn as nn
import torch

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


# torch.backends.cudnn.benchmark =True
from tool.config import Config

from tool.get_anchors import get_anchors
from tool.torch_PC_test import ProposalCreator

from torchvision.ops import RoIAlign
from torchvision.models import vgg16
from tool.RPN_net import RPN_net
from tool.Fast_net import Fast_net
import torch.nn.functional as F
from tool.faster_predict import predict

ce_loss = nn.CrossEntropyLoss()


def SmoothL1Loss(net_loc_train, loc, sigma, num):
    t = torch.abs(net_loc_train - loc)
    a = t[t < 1]
    b = t[t >= 1]
    loss1 = (a * sigma) ** 2 / 2
    loss2 = b - 0.5 / sigma ** 2
    loss = (loss1.sum() + loss2.sum()) / num
    return loss


class Faster_Rcnn(nn.Module):
    def __init__(self, config):
        super(Faster_Rcnn, self).__init__()
        self.config = config
        self.Mean = torch.tensor(config.Mean, dtype=torch.float32)
        self.num_anchor = len(config.anchor_scales) * len(config.anchor_ratios)
        self.anchors = get_anchors(np.ceil(self.config.img_max / 16 + 1), self.config.anchor_scales,
                                   self.config.anchor_ratios)

        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms,
                                  n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)

        self.features = vgg16().features[:-1]
        self.rpn = RPN_net(512, self.num_anchor)
        self.roialign = RoIAlign((7, 7), 1 / 16., 2)
        self.fast = Fast_net(config.num_cls, 512 * 7 * 7, 4096)

    def process_im(self, x):
        x = x[None]
        x = x[..., [2, 1, 0]]
        x = x.permute(0, 3, 1, 2)
        H, W = x.shape[2:]
        ma = max(H, W)
        mi = min(H, W)
        # xx=self.config.img_min
        # xx = np.random.choice([640, 672, 704, 736, 768, 800])
        scale = min(self.config.img_max / ma, self.config.img_min / mi)
        nh = int(round(H * scale))
        nw = int(round(W * scale))
        scale = torch.tensor([nw / W, nh / H, nw / W, nh / H], dtype=torch.float32)
        scale = cuda(scale)
        x = F.interpolate(x, size=(nh, nw))
        x = x.permute(0, 2, 3, 1)
        return x, scale

    def roi_layer(self, loc, score, anchor, img_size):
        roi = self.PC(loc, score, anchor, img_size, train=self.config.is_train)
        roi_inds = cuda(torch.zeros((roi.size()[0], 1)))
        roi = torch.cat([roi_inds, roi], dim=1)
        return roi

    # def forward(self, x):
    #     x = x.float().cuda()
    #     #
    #     x, scale = self.process_im(x)
    #     x = x - self.Mean.cuda()
    #     x = x.permute(0, 3, 1, 2)
    #     img_size = x.shape[2:]
    #     x = self.features(x)
    #     map_H, map_W = x.shape[2:]
    #     rpn_logits, rpn_loc = self.rpn(x)
    #
    #     tanchors = self.anchors[:map_H, :map_W]
    #     tanchors = tanchors.contiguous().view(-1, 4).cuda()
    #     rpn_score = F.softmax(rpn_logits, dim=-1)[:, 1]
    #     roi = self.roi_layer(rpn_loc.data, rpn_score.data, tanchors, img_size)
    #     x = roialign(x, roi)
    #     print(roi.shape)
    #     fast_logits, fast_loc = self.fast(x)
    #
    #     fast_loc = fast_loc * torch.tensor([0.1, 0.1, 0.2, 0.2]).cuda()
    #     pre = predict(fast_loc, F.softmax(fast_logits, dim=-1), roi[:, 1:], img_size[0], img_size[1], )
    #     pre[:,:4]=pre[:,:4]/scale
    #     return pre

    def forward(self, x):
        x = x.float()
        x = cuda(x)

        x, scale = self.process_im(x)
        x = x - cuda(self.Mean)
        x = x.permute(0, 3, 1, 2)

        img_size = x.shape[2:]

        x = self.features(x)
        rpn_logits, rpn_loc = self.rpn(x)
        map_H, map_W = x.shape[2:]
        tanchors = self.anchors[:map_H, :map_W]
        tanchors = tanchors.contiguous().view(-1, 4)
        tanchors = cuda(tanchors)
        roi = self.roi_layer(rpn_loc, F.softmax(rpn_logits, dim=-1)[:, 1], tanchors, img_size)

        x = self.roialign(x, roi)
        fast_logits, fast_loc = self.fast(x)
        fast_loc = fast_loc * cuda(torch.tensor([0.1, 0.1, 0.2, 0.2]))
        pre = predict(fast_loc, F.softmax(fast_logits, dim=-1), roi[:, 1:], img_size[0], img_size[1], )
        pre[:, :4] = pre[:, :4] / scale
        return pre


from datetime import datetime
from mAP.voc_mAP import mAP
import cv2
import joblib


def test(model, config, model_file):
    model = model(config)
    model.eval()
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    cuda(model)

    test_dir = r'D:\dataset\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007/JPEGImages/'
    names = os.listdir(test_dir)
    names = [name.split('.')[0] for name in names]

    names = sorted(names)

    i = 0
    m = 1000000

    Res = {}
    start_time = datetime.now()

    for name in names[:m]:
        i += 1
        print(datetime.now(), i)
        im_file = test_dir + name + '.jpg'
        img = cv2.imread(im_file)
        img = torch.tensor(img)
        res = model(img)
        res = res.cpu()
        res = res.detach().numpy()

        Res[name] = res

    print('==========', datetime.now() - start_time)
    joblib.dump(Res, 'Faster_vgg16_1.pkl')
    GT = joblib.load('../mAP/voc_GT.pkl')
    AP = mAP(Res, GT, 20, iou_thresh=0.5, use_07_metric=True, e=0.05)
    print(AP)
    AP = AP.mean()
    print(AP)


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]
    Mean = [123.675, 116.280, 103.530, ]
    model = Faster_Rcnn
    config = Config(False, Mean, None, img_max=1000, img_min=600, roi_min_size=16)
    model_file = '../train_one_GPU/models/vgg16_90000_2.pth'
    test(model, config, model_file)

# [0.73909475 0.78086676 0.7028246  0.60206878 0.61172184 0.82720654
#  0.83681608 0.85766458 0.55384811 0.77337219 0.63840307 0.83138268
#  0.84698006 0.76642416 0.7870849  0.44357723 0.75472496 0.67486161
#  0.81124731 0.74521005]
# 0.7292690138425238