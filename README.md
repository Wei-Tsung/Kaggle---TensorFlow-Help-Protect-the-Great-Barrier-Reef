# TensorFlow - Help Protect the Great Barrier Reef

### Detect crown-of-thorns starfish in underwater image data


<img src="https://storage.googleapis.com/kaggle-media/competitions/Google-Tensorflow/video_thumb_kaggle.png">

## Solution Ranking : Silver Medal

## Introduction
[Kaggle : Official Competition in Detail](https://www.kaggle.com/c/tensorflow-great-barrier-reef)

- Goal of the Competition
The goal of this competition is to accurately identify starfish in real-time by building an object detection model trained on underwater videos of coral reefs.

Your work will help researchers identify species that are threatening Australia's Great Barrier Reef and take well-informed action to protect the reef for future generations.

- Competition Task：**Computer Vision : Object Detection**, **Recommended Architecture：Yolov5**

- Dataset Core Challenges：The core challenge is that the images(many frames) are all from videos, and there are no target(0 shown up) in most of the images, which will cause difficulty for model training.

- Evaluation Metric：**IOU + F2** 

- Recommended Reading : Kaggle Essay About EDA（Exploratory Data Analysis）：[Learning to Sea: Underwater img Enhancement + EDA | Kaggle](https://www.kaggle.com/soumya9977/learning-to-sea-underwater-img-enhancement-eda)

- YOLOv5 baseline：[Great-Barrier-Reef: YOLOv5 [train]](https://www.kaggle.com/awsaf49/great-barrier-reef-yolov5-train ) 

  

## Dataset

Official Dataset Link [TensorFlow - Help Protect the Great Barrier Reef](https://www.kaggle.com/c/petfinder-pawpularity-score/data)

Dataset Volume： Training Set： 23,000+, Testset：Roughly 13,000 images, The format of the prediction is a bounding box, and a confidence level score for each identified starfish. Each image may contain any number (including 0) of starfish.。 

This competition uses a hidden test set that will be served by an API to ensure you evaluate the images in the same order they were recorded within each video. When your submitted notebook is scored, the actual test data (including a sample submission) will be availabe to your notebook.

### Dataset includes train.csv and test.csv :

- video_id - ID number of the video the image was part of. The video ids are not meaningfully ordered.
- video_frame - The frame number of the image within the video. Expect to see occasional gaps in the frame number from when the diver surfaced.
- sequence - ID of a gap-free subset of a given video. The sequence ids are not meaningfully ordered.
- sequence_frame - The frame number within a given sequence.
- image_id - ID code for the image, in the format '{video_id}-{video_frame}'
- annotations - The bounding boxes of any starfish detections in a string format that can be evaluated directly with Python. Does not use the same format as the predictions you will submit. Not available in test.csv. A bounding box is described by the pixel coordinate (x_min, y_min) of its upper left corner within the image together with its width and height in pixels.


## 解决方案思路
#### Cross-Validation
As for cross-validation,
We used two types of strategies：一种是简单的视频分割，另一种是基于子序列的5折。后期我们主要看的是子序列交叉验证的分数。子序列交叉验证我们只保留了有bboxes的图像。

#### Yolov5
我们在yolov5l6上训练，调整了模型结构（增大了模型中featuremap尺寸以更好的检测小目标）和超参数，在3100分辨率的图片上进行训练，模型使用adam和20个epoch进行训练。推理是在没有tta的3100*1.5分辨率上完成的。

#### Tracker

Tracker是一种视频目标追踪技术，本次竞赛我们使用了norfair库（Norfair is a light-weight custom Python library，for real-time Object Detection）。
在推理阶段起到了很好的连续跟踪之前图像中检测到的目标，参数上，我们选择则较为严格的跟踪策略。



## 比赛上分历程 Scores in progress

1. yolov5 baseline，Public score 0.630。
2. 增加yolov5中yaml文件中的数据增强部分（后续实验表明不宜过大）, Public score 0.634。
3. 降低推理阶段的 NMS confidence threshold至0.15，Public score 0.657。
4. 提高 IoU threshold 至 0.3 ，Public score 0.670。
5. 在Inference的时候进行TTA(flip)，没有涨分。
6. 在Inference的时候进行image enhance(例如clahe)，没有涨分。
7. 加入视频目标追踪 tracker，Public score 0.690 。
8. 修改模型结构 [-1, 1, Conv, [128, 3, 1]],  # 1-P2/4，Public score 0.710。
9. 将tracker目标，与原始目标都进行wbf融合，Public score 0.712。
10. 增加图像尺寸至3100，同时增加epoch，Public score 0.715。



## yolov5 参数表

datasets.yaml

```yaml
names:
- cots
nc: 1
path: /kaggle/working
train: /kaggle/working/train.txt
val: /kaggle/working/val.txt
```



hyp.yaml

```yaml
lr0: 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.3  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.10  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.5  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 0.25  # image mosaic (probability)
mixup: 0.25 # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
```



model.yaml（yolov5l6）

```yaml
# Parameters
nc: 1  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple
anchors:
  - [19,27,  44,40,  38,94]  # P3/8
  - [96,68,  86,152,  180,137]  # P4/16
  - [140,301,  303,264,  238,542]  # P5/32
  - [436,615,  739,380,  925,792]  # P6/64

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 1]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [768, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [768]],
   [-1, 1, Conv, [1024, 3, 2]],  # 9-P6/64
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 11
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [768, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 8], 1, Concat, [1]],  # cat backbone P5
   [-1, 3, C3, [768, False]],  # 15

   [-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 19

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 23 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 20], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 26 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 16], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [768, False]],  # 29 (P5/32-large)

   [-1, 1, Conv, [768, 3, 2]],
   [[-1, 12], 1, Concat, [1]],  # cat head P6
   [-1, 3, C3, [1024, False]],  # 32 (P6/64-xlarge)

   [[23, 26, 29, 32], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5, P6)
  ]
```





## Code、Dataset Structure

+ Code Base (In Jupyter Notebook)
  + gbr_train.ipynb
  + gbr_inference.ipynb
+ Training Data
  - Official Dataset：[TensorFlow - Help Protect the Great Barrier Reef](https://www.kaggle.com/c/petfinder-pawpularity-score/data)
  - train-5folds.csv
+ Additional Dataset For Inference
  + [YOLOv5 font | Kaggle](https://www.kaggle.com/awsaf49/yolov5-font)
  + [bbox lib ds | Kaggle](https://www.kaggle.com/awsaf49/bbox-lib-ds)
  + [loguru lib ds | Kaggle](https://www.kaggle.com/awsaf49/loguru-lib-ds)
  + [YOLOv5 lib ds | Kaggle](https://www.kaggle.com/awsaf49/yolov5-lib-ds)




## Final Result Reflection Summary

我们的方案基于YOLOv5完成，我们修改了模型结构，增大了模型中feature map尺寸以更好的检测海星(小目标)，因为本次训练集只基于3段视频，于是我们之后又增加了一些数据增强的强度，来防止过拟合，取得了不错的效果。在最后的推理阶段，我们加入了视频目标追踪Tracker，来连续跟踪之前检测到的目标。很遗憾我们在推理阶段尝试了各种形变、颜色等TTA都没有提升精度。最终我们获得了Private LB: 0.699 (Top2%) 的成绩。

 

<img src="https://user-images.githubusercontent.com/26456083/86477109-5a7ca780-bd7a-11ea-9cb7-48d9fd6848e7.jpg">
