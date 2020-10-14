# !/usr/bin/python
# -*- coding:utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
if __name__ == "__main__":
    os.system(
        '/home/zhai/anaconda3/envs/py36/bin/python  -m torch.distributed.launch --nproc_per_node=2   /home/zhai/PycharmProjects/object_detection/train_M_GPU_single_node/Mask_Rcnn_cascade_101.py')
    pass

#norm=35
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.621
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.467
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.239
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.468
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.571
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.341
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.531
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.554
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.346
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.598
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.710
# Loading and preparing results...
# DONE (t=0.79s)
# creating index...
# index created!
# 5000
# Running per image evaluation...
# Evaluate annotation type *segm*
# DONE (t=46.07s).
# Accumulating evaluation results...
# DONE (t=5.55s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.361
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.582
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.180
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.509
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.455
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.473
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.267
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.642
# ========== 0:18:57.835548
