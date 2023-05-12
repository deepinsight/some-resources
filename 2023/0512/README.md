
## CFAT2023初赛参考方法

## baseline方法

请参考 [trainer.py](trainer.py)



### 1. 依赖库

1. PyTorch
2. pytorch_lightning
3. timm
4. albumentations

### 2. 执行步骤

1. 将SuHiFiMask数据集解压至 ``data/`` 文件夹.
2. 执行 ``python trainer.py`` 启动训练, 可参考代码修改训练参数。默认采用timm中的resnet50-d网络骨干。
3. 可修改``datasets/dataset.py``中的代码来调整数据预处理和数据增强逻辑。
4. 训练完毕模型参数将默认生成至 ``work_dirs/1`` 文件夹.
5. 执行``python submit_A.py work_dirs/1/epoch=xx-val_loss=xxxx.ckpt ./submit_A.txt`` 生成初赛提交文件。

