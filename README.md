# paper-reproduction_DGM-with-domain-adversarial

复现李知航师兄的论文——《2023-李知航-基于迁移学习的超宽带测距校准算法研究》
在李玉萧师姐的DGM论文基础上添加了一个域对抗的域分类器

数据集是实验室采集的，大小是8行为一个样本，每一行有50点，label是一个2维的，表示的是测的点的坐标

loss改为了loss=loss_source+loss_target+λ loss_domain（域分类器前有一个梯度反转层，已经起到了对抗的作用，再加负号负负得正了）

