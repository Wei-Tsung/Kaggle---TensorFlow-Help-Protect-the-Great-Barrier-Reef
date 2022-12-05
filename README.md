# TensorFlow - Help Protect the Great Barrier Reef

### Detect crown-of-thorns starfish in underwater image data


<img src="https://storage.googleapis.com/kaggle-media/competitions/Google-Tensorflow/video_thumb_kaggle.png">

## Solution Ranking : Silver Medal

## Introduction
[Kaggle : Official Competition in Detail](https://www.kaggle.com/c/tensorflow-great-barrier-reef)

本次竞赛中，参赛者目标是通过建立一个在珊瑚礁水下视频中训练的**物体检测模型**，实时准确地识别**棘冠海星（COTS）**

- Competition Task：**Computer Vision : Object Detection**, Recommended Architecture：**Yolov5**

- Dataset Core Challenges：赛题数据图像的数量适中，挑战点在于图像均是来自于视频，以及大部分图像中并没有目标的出现，会不太好训练。 Training Set：23,000+Images，Hideen Testset：Roughly 13000 images。

- Evaluation Metric：**IOU + F2** 

- Recommended Reading : Kaggle Essay About EDA（Exploratory Data Analysis）：[Learning to Sea: Underwater img Enhancement + EDA | Kaggle](https://www.kaggle.com/soumya9977/learning-to-sea-underwater-img-enhancement-eda)

- YOLOv5 baseline：[Great-Barrier-Reef: YOLOv5 [train]](https://www.kaggle.com/awsaf49/great-barrier-reef-yolov5-train ) 

  

## Dataset

Official Dataset Link [TensorFlow - Help Protect the Great Barrier Reef](https://www.kaggle.com/c/petfinder-pawpularity-score/data)

本次比赛数据规模如下： 训练集： 23,000 多张图片 测试集：约13,000 张图片 预测的格式是一个边界框，以及每个被识别的海星的置信度。**每张图片可能包含任意数量（包括0）的海星**。 

本次比赛使用一个隐藏的测试集，该测试集将由一个API提 供，以确保你按照每个视频中记录的相同顺序来评估图像。 当你提交的笔记本被打分时，实际的测试数据（包括提交 的样本）将提供给你的笔记本。

数据中包含了train.csv和test.csv 

- video_id -图片所在的视频的ID号。视频ID是没有意义的排序。
- video_frame - 视频中图像的帧号。希望看到从潜水员浮出水面时起，帧号中偶尔会有空隙。
- sequence - 视频的（无间隙的）ID序列。序列的ID是没有意义的排序。
- sequence_frame - 一个给定的序列中的帧号。
- image_id -图像的ID代码，格式为"{video_id}-{video_frame} 
- annotations - 海星的标注边界框，其格式为可以直接用Python计算的字符串。边界框由图像中左上角 的像素坐标（x_min, y_min）以及其宽度和高度（像素）来描述 (x,y,h,w)。


## 解决方案思路
#### 交叉验证

对于交叉验证，我们一开始使用了两种策略：一种是简单的视频分割，另一种是基于子序列的5折。后期我们主要看的是子序列交叉验证的分数。子序列交叉验证我们只保留了有bboxes的图像。

#### Yolov5
我们在yolov5l6上训练，调整了模型结构（增大了模型中featuremap尺寸以更好的检测小目标）和超参数，在3100分辨率的图片上进行训练，模型使用adam和20个epoch进行训练。推理是在没有tta的3100*1.5分辨率上完成的。

#### Tracker

Tracker是一种视频目标追踪技术，本次竞赛我们使用了norfair库（Norfair 是一个可定制的轻量级 Python 库，用于实时对象跟踪）。在推理阶段起到了很好的连续跟踪之前图像中检测到的目标，参数上，我们选择则较为严格的跟踪策略。



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
+ 推理额外数据集 Additional Dataset For Inference
  + [YOLOv5 font | Kaggle](https://www.kaggle.com/awsaf49/yolov5-font)
  + [bbox lib ds | Kaggle](https://www.kaggle.com/awsaf49/bbox-lib-ds)
  + [loguru lib ds | Kaggle](https://www.kaggle.com/awsaf49/loguru-lib-ds)
  + [YOLOv5 lib ds | Kaggle](https://www.kaggle.com/awsaf49/yolov5-lib-ds)




## Final Result Reflection

竞赛是由Tensorflow官方组织举办，参赛者的目标是通过建立一个视频中的物体检测模型，实时准确地识别食珊瑚的棘冠海星，以保护大堡礁。我们的方案基于YOLOv5完成，我们修改了模型结构，增大了模型中feature map尺寸以更好的检测海星(小目标)，因为本次训练集只基于3段视频，于是我们之后又增加了一些数据增强的强度，来防止过拟合，取得了不错的效果。在最后的推理阶段，我们加入了视频目标追踪Tracker，来连续跟踪之前检测到的目标。很遗憾我们在推理阶段尝试了各种形变、颜色等TTA都没有提升精度。最终我们获得了Private LB: 0.699 (Top2%) 的成绩。


<img src="data:image/png;base64,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">
