# PCN in Pytorch
_**WARNING: This implement is in an almost finished state. See the [Results](#Results) section for more details.**_ 

Progressive Calibration Networks (PCN) is an accurate rotation-invariant face detector running at real-time speed on CPU. This is an implementation for PCN.

This is a pytorch implemented version of the [original repo](https://github.com/Jack-CV/FaceKit/tree/master/PCN)

## Getting Started

A separate Python environment is recommended.
+ Python3.6 (other python3 version may be ok, but I haven't tested them)
+ Pytorch == 1.0
+ opencv4 (opencv3 is ok)
+ numpy

install dependences using pip or conda
```
pip3 install numpy opencv-python
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision (optional)
```
or install using conda
```
conda install opencv numpy
conda install pytorch-cpu torchvision-cpu -c pytorch
```

## Usage
```
python pcn.py path/to/image 
```

## Results
The original implement uses C++/caffe and I'm not sure whether this is the main reason that results in certain failed examples as follow:

<img height=200 width=200 src="result/ret_5.jpg">
<img height=200 width=200 src="result/ret_10.jpg">

These are successful examples:

<img height=200 width=200 src="result/ret_0.jpg">
<img height=200 width=200 src="result/ret_2.jpg">
<img height=200 width=200 src="result/ret_11.jpg">
<img height=200 width=200 src="result/ret_25.jpg">

Contributions are welcome!


### Citing
@inproceedings{shiCVPR18pcn,
    Author = {Xuepeng Shi and Shiguang Shan and Meina Kan and Shuzhe Wu and Xilin Chen},
    Title = {Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks},
    Booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    Year = {2018}
}

