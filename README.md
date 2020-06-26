# LightNet
[LightNet: A Dual Spatiotemporal Encoder Network Model for Lightning Prediction](https://dl.acm.org/doi/10.1145/3292500.3330717) In *KDD* 2019

This is a Keras implementation of LightNet.

core.py       ---- For training and inference    
generator.py  ---- Data loader    
models.py ---- Define models    
scores.py     ---- For calculating evaluation metrics   

Requirements:   
Python 3.5.2, Keras 2.2.4, Numpy, Tensorflow 1.6.0


### Warning
Our code seems not to be sensitive to the version of python (3.5, 3.6) or Tensorflow (1.6.0~1.13.1), but is very dependent on the version of Keras (2.2.4). This is because we need to change the source code of Keras to fix a bug in its ConvLSTM2D (cf. https://github.com/keras-team/keras/issues/9761). See below. 

You may need to remove the following code 
```python
    inputs, initial_state, constants = _standardize_args(
         inputs, initial_state, constants, self._num_constants)
```
from "keras/layers/convolutional_recurrent.py", due to a bug in ConvLSTM2D of keras. cf. https://github.com/keras-team/keras/issues/9761

The deep learning (DL) framework we used is somewhat out-of-date. As a personal suggestion, you can try to implement your model by PyTorch or any other new DL framework, if you plan to conduct long-term research in this area.


# Reference  
```
@inproceedings{geng2019lightnet,
  title={LightNet: A Dual Spatiotemporal Encoder Network Model for Lightning Prediction},
  author={Geng, Yangli-ao and Li, Qingyong and Lin, Tianyang and Jiang, Lei and Xu, Liangtao and Zheng, Dong and Yao, Wen and Lyu, Weitao and Zhang, Yijun},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2439--2447},
  year={2019},
  organization={ACM}
}
```
