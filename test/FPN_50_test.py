# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn as nn

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


# torch.backends.cudnn.benchmark =True
from tool.config import Config

from tool.get_anchors import get_anchors

from tool.torch_PC_FPN import ProposalCreator

from torchvision.ops import MultiScaleRoIAlign
from tool.resnet import resnet50
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork as FPN_net, LastLevelMaxPool
from tool.FPN_RPN import RPN_net
from tool.FPN_Fast import Fast_net
from tool.faster_predict import predict
import torch.nn.functional as F
from collections import OrderedDict


class Faster_Rcnn(nn.Module):
    def __init__(self, config):
        super(Faster_Rcnn, self).__init__()
        self.config = config
        self.Mean = torch.tensor(config.Mean, dtype=torch.float32)
        self.num_anchor = len(config.anchor_scales) * len(config.anchor_ratios)
        self.anchors = []
        self.num_anchor = []
        for i in range(5):
            self.num_anchor.append(len(config.anchor_scales[i]) * len(config.anchor_ratios[i]))
            stride = 4 * 2 ** i
            print(stride, self.config.anchor_scales[i], self.config.anchor_ratios[i])
            anchors = get_anchors(np.ceil(self.config.img_max / stride + 1), self.config.anchor_scales[i],
                                  self.config.anchor_ratios[i], stride=stride)
            print(anchors.shape)
            self.anchors.append(anchors)

        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms,
                                  n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)

        self.features = resnet50()
        self.fpn = FPN_net([256, 512, 1024, 2048], 256, extra_blocks=LastLevelMaxPool())
        self.rpn = RPN_net(256, self.num_anchor[0])
        self.roialign = MultiScaleRoIAlign(['feat0', 'feat1', 'feat2', 'feat3'], 7, 2)
        self.fast = Fast_net(config.num_cls, 256 * 7 * 7, 1024)
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.fast_num = 0
        self.fast_num_P = 0

    def roi_layer(self, loc, score, anchor, img_size, map_HW):
        roi = self.PC(loc, score, anchor, img_size, map_HW, train=self.config.is_train)
        return roi

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

        # NHWC RGB
        return x, scale

    def pooling(self, P, roi, roi_inds):
        x = []
        inds = []
        index = cuda(torch.arange(roi.shape[0]))
        for i in range(4):
            t = roi_inds == i
            x.append(roialign_list[i](P[i], roi[t]))
            inds.append(index[t])
        x = torch.cat(x, dim=0)
        inds = torch.cat(inds, dim=0)
        inds = inds.argsort()
        x = x[inds]
        return x

    def forward(self, x):
        x = x.float()
        x = cuda(x)
        x, scale = self.process_im(x)
        x = x - cuda(self.Mean)
        x = x[..., [2, 1, 0]]
        x = x.permute(0, 3, 1, 2)
        img_size = x.shape[2:]
        C = self.features(x)
        feature_dict = OrderedDict()
        for i in range(len(C)):
            feature_dict['feat%d' % i] = C[i]

        feature_dict = self.fpn(feature_dict)

        rpn_logits, rpn_loc = self.rpn(list(feature_dict.values()))
        tanchors = []
        map_HW = []
        for i in range(5):
            if i == 4:
                H, W = feature_dict['pool'].shape[2:4]
            else:
                H, W = feature_dict['feat%d' % i].shape[2:4]
            map_HW.append((H, W))
            tanchors.append(self.anchors[i][:H, :W].contiguous().view(-1, 4))
        tanchors = cuda(torch.cat(tanchors, dim=0))

        roi = self.roi_layer(rpn_loc.data, F.softmax(rpn_logits.data, dim=-1)[:, 1],
                             tanchors,
                             img_size, map_HW)

        x = self.roialign(feature_dict, [roi], [tuple(img_size)])
        fast_logits, fast_loc = self.fast(x)
        fast_loc = fast_loc * cuda(torch.tensor([0.1, 0.1, 0.2, 0.2]))
        pre = predict(fast_loc, F.softmax(fast_logits, dim=-1), roi, img_size[0], img_size[1], )
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
    m = 100000000

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
    joblib.dump(Res, 'FPN_50.pkl')
    GT = joblib.load('../mAP/voc_GT.pkl')
    AP = mAP(Res, GT, 20, iou_thresh=0.5, use_07_metric=True, e=0.05)
    print(AP)
    AP = AP.mean()
    print(AP)


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]
    Mean = [123.675, 116.280, 103.530, ]
    config = Config(False, Mean, None, lr=0.00125, weight_decay=0.0001, img_max=1333, img_min=800,
                    anchor_scales=[[32], [64], [128], [256], [512]],
                    anchor_ratios=[[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]], fast_n_sample=512,
                    roi_min_size=[4, 8, 16, 32, 64],
                    roi_train_pre_nms=12000,
                    roi_train_post_nms=2000,
                    roi_test_pre_nms=6000,
                    roi_test_post_nms=1000, )
    model = Faster_Rcnn
    model_file = r'../train_one_GPU/models/FPN_50_90000_1.pth'
    test(model, config, model_file)

# [0.832859   0.84634415 0.80093688 0.69832556 0.71472847 0.86016459
#  0.87141688 0.8868125  0.61121053 0.823123   0.73037859 0.87026016
#  0.84513683 0.8240956  0.79131482 0.54478267 0.78263337 0.73879865
#  0.83208511 0.79926681]
# 0.7852337085259007

#scheduler.step()
# [0.83373099 0.79050381 0.79071721 0.67060483 0.71884434 0.87354488
#  0.87599354 0.88727139 0.60218415 0.83905339 0.72490257 0.88197833
#  0.86463234 0.82261894 0.84333185 0.5240511  0.77074536 0.74407417
#  0.82244895 0.76135338]
# 0.782129276046848