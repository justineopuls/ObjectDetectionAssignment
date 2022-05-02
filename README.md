# Object Detection Assignment
### Object detection assignment for Deep Learning Course
<br>

Justine Marcus A. Opulencia <br>
2018-07957

<br>

For this assignment, the [Faster R-CNN model with a ResNet-50-FPN backbone](https://pytorch.org/vision/stable/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn/) was used. I mostly followed the [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html/) given in Google Classroom to create [train.py](train.py) and [test.py](test.py).
<br>

### Install requirements
```
pip install -r requirements.txt
```

<br>

### Training

Training can be done by running
```
python train.py
```
The script should automatically download the drinks dataset and the fasterrcnn_resnet50_fpn model.

<br>

### Testing

Testing can be done by running
```
python test.py
```
The script should automatically download the drinks dataset and the pretrained model.
