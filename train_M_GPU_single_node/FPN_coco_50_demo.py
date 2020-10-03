# !/usr/bin/python
# -*- coding:utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
if __name__ == "__main__":
    os.system(
        '/home/zhai/anaconda3/envs/py36/bin/python  -m torch.distributed.launch --nproc_per_node=2   /home/zhai/PycharmProjects/object_detection/train_M_GPU_single_node/FPN_coco_50.py')
    pass

#norm=5
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
 # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.589
 # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.397
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.490
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.306
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.482
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.506
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.314
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.542
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.648

