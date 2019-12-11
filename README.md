# LightNet
LightNet: A Dual Spatiotemporal Encoder Network Model for Lightning Prediction In KDD 2019

This is a Keras implementation of Light, a dual-source spatiotemporal neural network for grid-level lightning forecast.

core.py       ---- For training and inference    
generator.py  ---- Data loader    
models.py ---- Define models    
scores.py     ---- For calculating evaluation metrics   

Requirements:   
Python 3.5.2, Keras 2.2.4, Numpy

Reference:    
@inproceedings{geng2019lightnet,
  title={LightNet: A Dual Spatiotemporal Encoder Network Model for Lightning Prediction},
  author={Geng, Yangli-ao and Li, Qingyong and Lin, Tianyang and Jiang, Lei and Xu, Liangtao and Zheng, Dong and Yao, Wen and Lyu, Weitao and Zhang, Yijun},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2439--2447},
  year={2019},
  organization={ACM}
}
