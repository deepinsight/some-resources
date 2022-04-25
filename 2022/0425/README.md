
## 初赛测评图片集及白盒模型下载地址

[https://pan.baidu.com/s/1_Vu_Ba2H9WFcWWAt8Hbxlg](https://pan.baidu.com/s/1_Vu_Ba2H9WFcWWAt8Hbxlg)  密码:plx6

## 白盒模型推理示例

请参考 [model_inference.py](model_inference.py#L15)

## baseline方法

请参考 [attack_example.py](attack_example.py)



### 1. 依赖库

1. PyTorch>=1.6
2. onnxruntime>=1.6

### 2. 执行步骤

1. 将白盒模型的参数文件放入 ``assets/`` 文件夹.
2. 执行 ``python attack_example.py``.
3. 添加 ``--device cuda`` 参数可开启GPU运行.(需安装支持CUDA的PyTorch/onnxruntime版本)
4. 运行完毕结果将默认生成至 ``output/`` 文件夹.
