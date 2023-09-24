# CV
computer vision algorithm
# structurePreviwe.py 针对BN,LN,GN,SN,WS这几个归一化模块进行了实现

# modules
- 在本模块中针对yolov5代码中mosaic放法改进，提升了对小物体检测的性能
- 引入senet模块，并加入backbone的最后一层，作为特征提取
- 复写了upsampling 和downsampling模块
- 自定了normalizer模块
- 在common.py中加入了DroupBlock2D modules ，改善传统dropout丢失特征太多或太少的问题
- 同时优化了yolo部分代码的结构，如，多尺度训练，动态计算AMP等代码结构
