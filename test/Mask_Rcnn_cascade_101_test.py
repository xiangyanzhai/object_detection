# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

import torch

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


import torch.nn as nn
import torch.nn.functional as F
# torch.backends.cudnn.benchmark =True
from tool.config import Config

from tool.get_anchors import get_anchors
from tool.torch_ATC_FPN import AnchorTargetCreator
from tool.torch_PC_FPN import ProposalCreator
from tool.torch_PTC_mask import ProposalTargetCreator
from torchvision.ops import MultiScaleRoIAlign
from tool.resnet import resnet101
from tool.FPN_net import FPN_net
from tool.FPN_RPN import RPN_net
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork as FPN_net, LastLevelMaxPool
from torchvision.ops import RoIAlign, roi_align
from tool.FPN_Fast import Fast_net
from tool.Mask_net import Mask_net
import torch.nn.functional as F
from tool.read_Data_mask import Read_Data
from tool.cascade_predict import predict
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from collections import OrderedDict


def SmoothL1Loss(net_loc_train, loc, sigma, num):
    t = torch.abs(net_loc_train - loc)
    a = t[t < 1]
    b = t[t >= 1]
    loss1 = (a * sigma) ** 2 / 2
    loss2 = b - 0.5 / sigma ** 2
    loss = (loss1.sum() + loss2.sum()) / num
    return loss


class Mask_Rcnn(nn.Module):
    def __init__(self, config):
        super(Mask_Rcnn, self).__init__()
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

        self.features = resnet101()
        self.fpn = FPN_net([256, 512, 1024, 2048], 256, extra_blocks=LastLevelMaxPool())
        self.rpn = RPN_net(256, self.num_anchor[0])
        self.roialign_7 = MultiScaleRoIAlign(['feat0', 'feat1', 'feat2', 'feat3'], 7, 2)
        self.roialign_14 = MultiScaleRoIAlign(['feat0', 'feat1', 'feat2', 'feat3'], 14, 2)
        # self.roialign_28 = RoIAlign((28, 28), 1.0, 2)
        self.fast = Fast_net(config.num_cls, 256 * 7 * 7, 1024)
        self.fast_2 = Fast_net(config.num_cls, 256 * 7 * 7, 1024)
        self.fast_3 = Fast_net(config.num_cls, 256 * 7 * 7, 1024)
        self.mask_net = Mask_net(256, config.num_cls)

        self.loc_std1 = [1. / 10, 1. / 10, 1. / 5, 1. / 5]
        self.loc_std2 = [1. / 20, 1. / 20, 1. / 10, 1. / 10]
        self.loc_std3 = [1. / 30, 1. / 30, 1. / 15, 1. / 15]
        self.weights = [1.0, 1.0, 1.0]

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

    def pooling(self, P, roi, roi_inds, size):
        x = []
        inds = []
        index = cuda(torch.arange(roi.shape[0]))

        for i in range(4):
            t = roi_inds == i
            if size == 7:
                x.append(roialign_list_7[i](P[i], roi[t]))
            else:

                x.append(roialign_list_14[i](P[i], roi[t]))
            inds.append(index[t])

        x = torch.cat(x, dim=0)
        inds = torch.cat(inds, dim=0)
        inds = inds.argsort()
        x = x[inds]
        return x

    def roi_layer(self, roi):

        area = roi[:, 2:] - roi[:, :2] + 1
        area = area.prod(dim=-1)
        roi_inds = torch.floor(4.0 + torch.log(area ** 0.5 / 224.0) / np.log(2.0))
        roi_inds = roi_inds.clamp(2, 5) - 2

        roi = torch.cat([cuda(torch.zeros(roi.shape[0], 1)), roi], dim=-1)
        return roi, roi_inds

    def forward(self, x):

        x = cuda(x.float())

        x, scale = self.process_im(x)
        x = x - cuda(self.Mean)
        x = x[..., [2, 1, 0]]
        x = x.permute(0, 3, 1, 2)

        img_size = x.shape[2:]

        C = self.features(x)
        feature_dict = OrderedDict()
        for i in range(len(C)):
            feature_dict['feat%d' % i] = C[i]

        # print('=========================')
        feature_dict = self.fpn(feature_dict)
        # for i in list(feature_dict.values()):
        #     print(i.shape)
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
        roi = self.PC(rpn_loc.data, F.softmax(rpn_logits.data, dim=-1)[:, 1], tanchors, img_size, map_HW,
                      train=self.config.is_train)

        x = self.roialign_7(feature_dict, [roi], [tuple(img_size)])
        fast_logits, fast_loc = self.fast(x)

        fast_loc = fast_loc[:, 1:] * cuda(torch.tensor(self.loc_std1))

        roi = self.loc2bbox(fast_loc, roi[:, None])
        score = F.softmax(fast_logits, dim=-1)[:, 1:]
        pre_bboxes = roi * self.weights[0]
        pre_score = score * self.weights[0]
        _, inds = score.max(dim=-1)
        t = torch.arange(score.shape[0])
        roi = roi[t, inds]
        roi, inds = self.filter_bboxes(roi, img_size, self.config.roi_min_size[0])
        pre_bboxes = pre_bboxes[inds]
        pre_score = pre_score[inds]

        x = self.roialign_7(feature_dict, [roi], [tuple(img_size)])
        fast_logits, fast_loc = self.fast_2(x)

        fast_loc = fast_loc[:, 1:] * cuda(torch.tensor(self.loc_std2))
        roi = self.loc2bbox(fast_loc, roi[:, None])
        score = F.softmax(fast_logits, dim=-1)[:, 1:]
        pre_bboxes = pre_bboxes + roi * self.weights[1]
        pre_score = pre_score + score * self.weights[1]
        _, inds = score.max(dim=-1)
        t = torch.arange(score.shape[0])
        roi = roi[t, inds]
        roi, inds = self.filter_bboxes(roi, img_size, self.config.roi_min_size[0])

        pre_bboxes = pre_bboxes[inds]
        pre_score = pre_score[inds]
        if roi.shape[0] == 0:
            return cuda(torch.zeros((0, 6))), cuda(torch.zeros((0, 28, 28)))

        x = self.roialign_7(feature_dict, [roi], [tuple(img_size)])

        fast_logits, fast_loc = self.fast_3(x)

        fast_loc = fast_loc[:, 1:] * cuda(torch.tensor(self.loc_std3))
        roi = self.loc2bbox(fast_loc, roi[:, None])
        score = F.softmax(fast_logits, dim=-1)[:, 1:]
        pre_bboxes = pre_bboxes + roi * self.weights[2]
        pre_score = pre_score + score * self.weights[2]
        pre_bboxes = pre_bboxes / sum(self.weights)
        pre_score = pre_score / sum(self.weights)

        pre = predict(pre_bboxes, pre_score, img_size[0], img_size[1], iou_thresh_=0.5, c_thresh=0.05)[:100]

        if pre.shape[0] == 0:
            return pre, cuda(torch.zeros((0, 28, 28)))

        roi = pre[:100]
        inds_b = roi[:, -1].long() + 1

        # roi, roi_inds = self.roi_layer(roi[..., :4])

        net_mask = self.roialign_14(feature_dict, [roi[..., :4]], [tuple(img_size)])
        net_mask = self.mask_net(net_mask)
        net_mask = torch.sigmoid(net_mask)
        inds_a = torch.arange(roi.shape[0])

        mask = net_mask[inds_a, inds_b]
        pre[:, :4] = pre[:, :4] / scale
        return pre, mask

    def filter_bboxes(self, roi, img_size, roi_min_size):
        h, w = img_size
        roi[:, slice(0, 4, 2)] = torch.clamp(roi[:, slice(0, 4, 2)], 0, w)
        roi[:, slice(1, 4, 2)] = torch.clamp(roi[:, slice(1, 4, 2)], 0, h)
        hw = roi[:, 2:4] - roi[:, :2]
        inds = hw >= roi_min_size
        inds = inds.all(dim=-1)
        roi = roi[inds]
        return roi, inds

    def loc2bbox(self, pre_loc, anchor):
        c_hw = anchor[..., 2:4] - anchor[..., 0:2]
        c_yx = anchor[..., :2] + c_hw / 2
        yx = pre_loc[..., :2] * c_hw + c_yx
        hw = torch.exp(pre_loc[..., 2:4]) * c_hw
        yx1 = yx - hw / 2
        yx2 = yx + hw / 2
        bboxes = torch.cat((yx1, yx2), dim=-1)
        return bboxes


from datetime import datetime
import joblib


def loadNumpyAnnotations(data):
    """
    Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    :param  data (numpy.ndarray)
    :return: annotations (python nested list)
    """
    print('Converting ndarray to lists...')
    assert (type(data) == np.ndarray)
    print(data.shape)
    assert (data.shape[1] == 7)
    N = data.shape[0]
    ann = []
    for i in range(N):
        if i % 1000000 == 0:
            print('{}/{}'.format(i, N))

        ann += [{
            'image_id': int(data[i, 0]),
            'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
            'score': data[i, 5],
            'category_id': int(data[i, 6]),
        }]

    return ann


import cv2
from pycocotools import mask as maskUtils


def loadNumpyAnnotations_mask(data, mask):
    global oh, ow

    t = {
        'image_id': int(data[0]),
        'bbox': [data[1], data[2], data[3], data[4]],
        'score': data[5],
        'category_id': int(data[6]),
    }

    res_mask = np.zeros((oh, ow), dtype=np.uint8, order='F')
    bbox = t['bbox']
    bbox = np.round(bbox)
    bbox = bbox.astype(np.int32)
    x1 = bbox[0]
    y1 = bbox[1]
    w = bbox[2]
    h = bbox[3]
    x2 = x1 + w
    y2 = y1 + h

    x1 = np.clip(x1, 0, ow)
    x2 = np.clip(x2, 0, ow)

    y1 = np.clip(y1, 0, oh)
    y2 = np.clip(y2, 0, oh)
    w = x2 - x1
    h = y2 - y1

    img = cv2.resize(mask, (w, h))
    img = np.round(img)
    img = img.astype(np.uint8)

    res_mask[y1:y2, x1:x2] = img
    tt = maskUtils.encode(res_mask)
    tt['counts'] = tt['counts'].decode('utf-8')
    t["segmentation"] = tt

    return t


import codecs
import json
from tool import eval_coco_box
from tool import eval_coco_segm


def test(model, config, model_file):
    global oh, ow
    catId2cls, cls2catId, catId2name = joblib.load(
        r'../data_preprocess/(catId2cls,cls2catId,catId2name).pkl')
    model = model(config)
    model.eval()
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    cuda(model)
    test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/coco/val2017/'
    names = os.listdir(test_dir)
    names = [name.split('.')[0] for name in names]
    names = sorted(names)

    i = 0
    mm = 1000000
    Res = []
    Res_mask = []
    start_time = datetime.now()
    for name in names[:mm]:
        i += 1

        print(datetime.now(), i)

        im_file = test_dir + name + '.jpg'
        img = cv2.imread(im_file)
        oh, ow = img.shape[:2]

        with torch.no_grad():
            img = torch.tensor(img)
            res, res_mask = model(img)

        res = res.cpu().numpy()
        # res = res.detach().numpy()
        res_mask = res_mask.cpu()
        res_mask = res_mask.detach().numpy()

        wh = res[:, 2:4] - res[:, :2]

        imgId = int(name)
        m = res.shape[0]

        imgIds = np.zeros((m, 1)) + imgId

        cls = res[:, 5]
        cid = map(lambda x: cls2catId[x], cls)
        cid = list(cid)
        cid = np.array(cid)
        cid = cid.reshape(-1, 1)

        res = np.concatenate((imgIds, res[:, :2] + 1, wh, res[:, 4:5], cid), axis=1)
        # Res=np.concatenate([Res,res])
        res = np.round(res, 4)
        Res.append(res)
        Res_mask += map(loadNumpyAnnotations_mask, res[:100], res_mask[:100])

    Res = np.concatenate(Res, axis=0)

    Ann = loadNumpyAnnotations(Res)
    print('==================================', mm, datetime.now() - start_time)
    # with codecs.open('Mask_Rcnn_bbox_ohem_256_gpu2.json', 'w', 'ascii') as f:
    #     json.dump(Ann, f)
    # with codecs.open('Mask_Rcnn_segm_ohem_256_gpu2.json', 'w', 'ascii') as f:
    #     json.dump(Res_mask, f)
    # eval_coco_box.eval('Mask_Rcnn_bbox_ohem_256_gpu2.json', mm)
    # eval_coco_segm.eval('Mask_Rcnn_segm_ohem_256_gpu2.json', mm)

    with codecs.open('py_Mask_Rcnn_bbox.json', 'w', 'ascii') as f:
        json.dump(Ann, f)
    with codecs.open('py_Mask_Rcnn_segm.json', 'w', 'ascii') as f:
        json.dump(Res_mask, f)
    print(mm)
    eval_coco_box.eval('py_Mask_Rcnn_bbox.json', mm,
                       annFile=r'/home/zhai/PycharmProjects/Demo35/dataset/coco/annotations/instances_val2017.json')
    eval_coco_segm.eval('py_Mask_Rcnn_segm.json', mm)

    print('==========', datetime.now() - start_time)


if __name__ == "__main__":
    Mean = [123.68, 116.78, 103.94]
    Mean = [123.675, 116.280, 103.530, ]
    config = Config(False, Mean, None, lr=0.00125, weight_decay=0.0001, num_cls=80, img_max=1333, img_min=800,
                    anchor_scales=[[32], [64], [128], [256], [512]],
                    anchor_ratios=[[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]], fast_n_sample=512,
                    roi_min_size=[1, 1, 1, 1, 1],
                    roi_train_pre_nms=12000,
                    roi_train_post_nms=2000,
                    roi_test_pre_nms=6000,
                    roi_test_post_nms=1000, )
    model = Mask_Rcnn
    model_file = '../train_one_GPU/models/Mask_Rcnn_cascade_101_90000_1.pth'
    model_file = '../train_M_GPU_single_node/models/Mask_Rcnn_cascade_101_4x_360000_1_0.pth'
    test(model, config, model_file)

#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.254
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.417
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.268
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.274
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.343
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.243
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.388
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.207
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.441
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.541
# Loading and preparing results...
# DONE (t=1.46s)
# creating index...
# index created!
# 5000
# Running per image evaluation...
# Evaluate annotation type *segm*
# DONE (t=32.21s).
# Accumulating evaluation results...
# DONE (t=5.19s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.222
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.387
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.227
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.095
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.239
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.321
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.218
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.338
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.353
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.384
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
# ========== 0:28:33.487019
