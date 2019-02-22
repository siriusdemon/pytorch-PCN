# PCN in Pytorch
_**WARNING: 这个Pytorch实现接近完成，详见[Results](#Results)一节.**_ 

[English version](README.md).

渐进校准网络(PCN)是一个又快又准，又具有旋转不变性的人脸检测模型，能在CPU上达到实时，本仓库是[原仓库](https://github.com/Jack-CV/FaceKit/tree/master/PCN)的一个Pytorch实现。

## Getting Started

推荐使用一个独立的Python环境。
+ Python3.5+ (Python3.5, Python3.6 are tested)
+ Pytorch == 1.0
+ opencv4 (opencv3.4.5 is tested also)
+ numpy

使用`pip`或者`conda`来安装依赖。
```
pip3 install numpy opencv-python
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision (optional)
```
使用`conda`的命令
```
conda install opencv numpy
conda install pytorch-cpu torchvision-cpu -c pytorch
```

## Usage
```
python pcn.py path/to/image 
```

## Results
这是一些成功检测的例子。

<img src="result/ret_2.jpg">
<img src="result/ret_11.jpg">
<img src="result/ret_25.jpg">

原仓库使用`c++`和`caffe`，我不确定这是不是造成部分测试用例失败的原因。

<img src="result/ret_5.jpg">
<img src="result/ret_10.jpg">

你可以在`result`文件夹下找到更多的例子，或者你也可以运行脚本来生成。

## Contributions
正如我在*Results*一节提到的，我需要帮忙。我主要是在[我自己的仓库](https://github.com/siriusdemon/hackaway/tree/master/projects/pcn)里调试。如果你有兴趣修复这个bug，你可以在我的仓库里找到一些可能有点儿用的测试脚本。为了避免给大家造成困惑，我就只放在我自己的仓库里。

### License
这份代码的许可证是[BSD 2-Clause license](LICENSE).

### Citing & Thanks
    @inproceedings{shiCVPR18pcn,
        Author = {Xuepeng Shi and Shiguang Shan and Meina Kan and Shuzhe Wu and Xilin Chen},
        Title = {Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks},
        Booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        Year = {2018}
    }