# PCN in Pytorch

[中文版](README-zh.md).


Progressive Calibration Networks (PCN) is an accurate rotation-invariant face detector running at real-time speed on CPU. This is an implementation for PCN.

This is a pytorch implementation version of the [original repo](https://github.com/Jack-CV/FaceKit/tree/master/PCN)

## Getting Started

A separate Python environment is recommended.
+ Python3.5+ (Python3.5, Python3.6 are tested)
+ Pytorch == 1.0
+ opencv4 (opencv3.4.5 is tested also)
+ numpy

install dependences using `pip`
```bash
pip3 install numpy opencv-python
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision (optional)
```
or install using `conda`
```bash
conda install opencv numpy
conda install pytorch-cpu torchvision-cpu -c pytorch
```

## Usage
```bash
cd pytorch-PCN
python demo.py path/to/image 
```
or use webcam demo
```bash
python webcam.py
```

## Install
```
cd pytorch-PCN && pip install .
```

## Results
<img src="result/ret_5.jpg">
<img src="result/ret_10.jpg">
<img src="result/ret_11.jpg">
<img src="result/ret_25.jpg">

More results can be found in `result` directory, or you can run the script to generate them.

There is still one image failed. Pull requests to fix it is welcome.
<img src="result/ret_20.jpg">

### License
This code is distributed under the [BSD 2-Clause license](LICENSE).

### Citing & Thanks
    @inproceedings{shiCVPR18pcn,
        Author = {Xuepeng Shi and Shiguang Shan and Meina Kan and Shuzhe Wu and Xilin Chen},
        Title = {Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks},
        Booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        Year = {2018}
    }

### Wishes
For anyone who hear, see, think about or use this repo, I hope them gain temporary happiness and everlasting happiness
