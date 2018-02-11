# Pelee: A Real-Time Object Detection System on Mobile Devices
This repository contains the code for the following paper. 

[Pelee: A Real-Time Object Detection System on Mobile Devices](https://openreview.net/pdf?id=r1PhalTLf)

The code is based on the [SSD](https://github.com/weiliu89/caffe/tree/ssd) framework. 

## Results 

The table below shows the results on PASCAL VOC 2007 test.

| Method | mAP (%) | FPS (Intel i7) |FPS (iPhone 6s) |FPS (iPhone 8) | # parameters 
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|
| YOLOv2-288 | 69.0 | 1.0 | - | - | 58.0M |
| DSOD300_smallest| 73.6 | 1.3 | - | - |5.9M |
| Tiny-YOLOv2 | 57.1 | 2.4 | 9.3 | 23.8 | 15.9M |
| SSD+MobileNet | 68.0 | 6.1 | 16.1 | 22.8 |5.8M |
| [Pelee](https://drive.google.com/file/d/1mq54LMPoiyqzrxnu6T5naObCDtZYFNbQ/view?usp=sharing) | 70.5 | 6.7 | 17.1 | 23.6 | 5.4M |

## Preparation 

0. Install SSD (https://github.com/weiliu89/caffe/tree/ssd) following the instructions there, including: (1) Install SSD caffe; (2) Download PASCAL VOC 2007 and 2012 datasets; and (3) Create LMDB file. Make sure you can run it without any errors.

1. Download the pretrained [PeleeNet](https://drive.google.com/file/d/1OBzEnD5VEB_q_B8YkLx-i3PMHVO-wagk/view?usp=sharing) model. By default, we assume the model is stored in $CAFFE_ROOT/models/
2. Clone this repository and create a soft link to $CAFFE_ROOT/examples 
  ```shell
  git clone https://github.com/Robert-JunWang/Pelee.git
  ln -sf `pwd`/Pelee $CAFFE_ROOT/examples/pelee
  ```
## Training & Testing

- Train a Pelee model on VOC 07+12:

  ```shell
  cd $CAFFE_ROOT
  python examples/pelee/train_voc.py
  ```
- Evaluate the model:

  ```shell
  cd $CAFFE_ROOT
  python examples/pelee/eval_voc.py


## Models
- Pascal VOC 07+12: [Download (20.3M)](https://drive.google.com/file/d/1OBzEnD5VEB_q_B8YkLx-i3PMHVO-wagk/view?usp=sharing) 


