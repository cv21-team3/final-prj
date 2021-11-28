## Main Figure Segmentor
### Introduction
segmentor.py segments the main characters of each photo with the assistance of the masked R-CNN. The base code is from https://learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-rcnn-in-opencv-python-c/ and was modified to yield intended results.

### How to Run
Use the following commands for execution,
```python3 segmentor.py --image=FILE_NAME --mode=MODE```
where ```--mode=segment``` option yields an image with a full segmentation, whereas ```--mode=black``` option yields an image with a main figure segment and an area painted black outside the segment.

For example,
```python3 segmentor.py --image=people.jpg --mode=segment```
command yields an image that involves a segmentation result of people.jpg.

While you are running with the ```--mode=black``` option, you have to identify the index to designate the main figure.

The result will be saved in the same directory as test_generator.py
