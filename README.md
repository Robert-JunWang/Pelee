# Pelee: A Real-Time Object Detection System on Mobile Devices
This repository contains the code for Pelee. The code is based on the [SSD](https://github.com/weiliu89/caffe/tree/ssd) framework. 

## Results & Models

The tables below show the results on PASCAL VOC 2007.

| Method | VOC 2007 test *mAP* | FPS (Intel i7) |FPS (iPhone 6s) |FPS (iPhone 8) | # parameters | Models 
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| YOLOv2-288 | 69.0 | 1.0 | - | - | 58.0M |- |
| DSOD300_smallest| 73.6 | 1.3 | - | - | 5.9M |- |
| Tiny-YOLOv2 | 57.1 | 2.4 | 9.3 | 23.8 | 15.9M |- |
| SSD+MobileNet | 68.0 | 6.1 | 16.1 | 22.8 | 5.8M |- |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Pelee | 70.5 | 6.7 | 17.1 | 23.6 | 5.4M |- |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
