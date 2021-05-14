# AI-Studio-螺旋桨RNA结构预测竞赛第25名方案

## 项目描述
本方案使用三层双向RNN结构，单点输入数据为自身与左右相邻核酸的onehot编码信息，同时加上预测结构，
实验结果显示使用左右相邻的核酸信息会有利于训练，但再增加临近点的核酸对训练没有太大提升。
实验结果显示单独使用transformer结果要比三层双向RNN差，但双向RNN输出再结合transformer后效果更佳，
这一方案模型在other_model文件中。比赛使用模型为单纯的三层双向RNN结构。

## 项目结构
```
├── LICENSE
├── model3.py
├── other_model
│   ├── model4.py
│   ├── save
│   │   └── model4
│   │       └── RNA_net.pdparams
│   └── test_RNN.py
├── predict
├── README.md
├── requirements.txt
├── RNA_data
│   ├── B_board_112_seqs.txt
│   ├── data_explanation.txt
│   ├── dev.txt
│   ├── test_nolabel.txt
│   └── train.txt
├── save
│   └── model3
│       └── RNA_net.pdparams
└── test_RNN.py
```
## 使用方式
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/usercenter)
B：测试程序：在根目录下运行python3 test_RNN.py，运行结果将保存到predict文件夹内。
             程序运行的将是B榜测试结果，如需运行A榜测试结果需将test_RNN.py代码的
             104行的B_board_112_seqs.txt改成test_nolabel.txt
             使用的模型文件在save/model3/RNA_net.pdparams
   训练程序：在根目录下运行python3 model3.py，运行后的结果将保存在save/model3
             注意：运行训练程序后原先的保存模型将被覆盖掉。
             
   保存的模型的checkpoint是最终的运行结束时的检查点。
             
