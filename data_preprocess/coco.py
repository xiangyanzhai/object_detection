# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import joblib
from pycocotools.coco import COCO
import cv2

train_file = r'/home/zhai/PycharmProjects/Demo35/dataset/coco/instances_valminusminival2014.json'
train_file = r'D:\dataset\annotations_trainval2017\annotations/instances_train2017.json'
cocoGt = COCO(train_file)
imgIds = cocoGt.getImgIds()
imgIds = sorted(imgIds)

catIds = cocoGt.getCatIds()
print(type(catIds))

catId2cls = {}
cls2catId = {}
catId2name = {}
name2cls = {}
name2catId = {}
cls2name = {}
for i in range(len(catIds)):
    catId2cls[catIds[i]] = i
    cls2catId[i] = catIds[i]
cats = cocoGt.loadCats(cocoGt.getCatIds())
for cat in cats:
    a = cat['id']
    name = cat['name']
    catId2name[a] = name
    name2catId[name] = a
    name2cls[name] = catId2cls[a]
    cls2name[catId2cls[a]] = name
joblib.dump((catId2cls, cls2catId, catId2name), '(catId2cls,cls2catId,catId2name).pkl')

nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
print(len(nms))
nms = set([cat['supercategory'] for cat in cats])
print(len(nms))
imgDir = r'D:\dataset\train2017/'
tname = '000000000000'
c = 0
print(cls2catId.keys())
print(catId2cls.keys())
print(catId2name.keys())


def draw_gt(im, gt):
    im = im.astype(np.uint8)
    boxes = gt.astype(np.int32)
    for box in boxes:
        # print(box)
        y1, x1, y2, x2 = box[:4]
        print(box)
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
        print(box[-1])
    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


# writer = tf.python_io.TFRecordWriter('coco_train2017.tf')
print(len(imgIds))

np.random.seed(50)
np.random.shuffle(imgIds)
coco_bboxes_2017 = []
coco_imgpaths_2017 = []
coco_mask_2017 = []

for i in imgIds[:]:
    Id = str(i)
    l = len(Id)
    name = tname[:-l] + Id
    file = imgDir + name + '.jpg'
    # im = tf.gfile.FastGFile(file, 'rb').read()
    img = cv2.imread(file)

    h, w = img.shape[:2]

    AnnIds = cocoGt.getAnnIds([i], )
    Anns = cocoGt.loadAnns(AnnIds)
    bboxes = np.zeros((0, 6))
    Counts = []
    for Ann in Anns:
        tt = cocoGt.annToRLE(Ann)

        hh, ww = tt['size']
        counts = tt['counts']
        if hh != h or ww != w:
            continue
        bbox = Ann['bbox']
        bbox = np.array(bbox)
        x1, y1 = bbox[:2]
        x2, y2 = bbox[:2] + bbox[2:]
        catId = Ann['category_id']
        cls = catId2cls[catId]
        iscrowd = Ann['iscrowd']
        if iscrowd == 1:
            continue
        t = np.array([[y1, x1, y2, x2, cls, iscrowd]])
        bboxes = np.concatenate((bboxes, t), axis=0)
        Counts.append(counts)
    if bboxes.shape[0] == 0:
        continue
    bboxes = bboxes.astype(np.float32)
    bboxes[:, :4] = bboxes[:, :4] - 1
    bboxes[:, slice(0, 4, 2)] = np.clip(bboxes[:, slice(0, 4, 2)], 0, h - 1)
    bboxes[:, slice(1, 4, 2)] = np.clip(bboxes[:, slice(1, 4, 2)], 0, w - 1)

    # xyxy
    bboxes = bboxes[:, [1, 0, 3, 2, 4, 5]]
    bboxes = bboxes.astype(np.float32)

    coco_bboxes_2017.append(bboxes)
    coco_imgpaths_2017.append(file)
    coco_mask_2017.append(Counts)
    c += 1
    print(c)
joblib.dump(coco_bboxes_2017, 'coco_bboxes_2017.pkl')
joblib.dump(coco_mask_2017, 'coco_mask_2017.pkl')
joblib.dump(coco_imgpaths_2017, 'coco_imgpaths_2017.pkl')

if __name__ == "__main__":
    pass
