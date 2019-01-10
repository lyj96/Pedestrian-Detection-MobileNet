## Environment

- Tensorflow 1.9.0
- OpenCV  3.4.2
- Numpy 1.15.4
- Cuda 9.0
- Anaconda 5.1.0 / Python 3.6.4
- Protobuf 3.6.1
- [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

## 项目说明

1. 是对一篇行人检测博客的实现[How to Automate Surveillance Easily with Deep Learning](https://medium.com/nanonets/how-to-automate-surveillance-easily-with-deep-learning-4eb4fa0cd68d) 
2. 基于谷歌目标检测API ：[Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
3. 使用[ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)作为预训练模型
4. 数据集：[Town Centre Dataset](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets)
5. 训练使用显卡RTX 2080 Ti，GPU加速训练 ，训练步数：50000

## 文件说明

- object_detection、slim、utils.py：谷歌目标检测API及其依赖文件
- extract_GT.py：将数据的标注由csv格式转为xml格式
- extract_towncentre.py：从视频中提取训练和测试数据，其中前3600帧为训练集，后900帧为测试集
- creat_tf_record.py：将提取的图片和标注转换为tf_record格式
- pedestrian_detection：训练完成后，导出的模型
- test_data：900张测试图片及其标签，以及原始视频的后2500帧，用于实时检测应用
- test_image.jpg：用于计算单张图片的测试时间的图片
- image_object_detection.py：检测单张图片，并输出检测时间
- real_time_detection.py：用于实时检测，视频流可以选择摄像头或者本地视频文件

## 如何使用

```python
python real_time_detection.py  \ 
    --stream 0 \   #0为摄像头检测，1为本地视频
    --video_dir test_data/test.avi \ #本地视频的路径
    --output_dir output #输出检测结果视频的路径，不指定则不输出
```

## 参考

[Pedestrian-Detection](https://github.com/thatbrguy/Pedestrian-Detection)（基于[annotations](https://github.com/thatbrguy/Pedestrian-Detection/tree/master/annotations)实现）

[MobileNet-SSDLite-RealSense-TF](https://github.com/PINTO0309/MobileNet-SSDLite-RealSense-TF)

[Object-Detection-MobileNet](https://github.com/Sid2697/Object-Detection-MobileNet#functionalities)

[Quick Start: Distributed Training on the Oxford-IIIT Pets Dataset on Google Cloud](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md)



