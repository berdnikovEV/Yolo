# Yolo v3 Object Detection in Tensorflow
Yolo v3 is an algorithm that uses deep convolutional neural networks to detect objects. <br> <br>
[Kaggle notebook](https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow) 

### Setup


```
pip install -r requirements.txt

wget -P weights https://pjreddie.com/media/files/yolov3.weights

python load_weights.py
```

### Usage

```
python detect.py data/video/hrodna.mp4
```

