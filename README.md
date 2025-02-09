# EEG_Classify
本项目复现了Transfer Learning for Brain–Computer Interfaces:A Euclidean Space Data Alignment Approach这篇论文提出的用于EEG信号运动想象分类的欧拉对齐方法。此外还对比了多种对EEG信号运动想象分类的方法，尝试了多种特征预处理、特征提取与分类器的排列组合，最终发现该文章提出的EA+CSP+SVM方法得到的效果最佳。

# 一点对结果的分析
文章主要提出的EA（欧拉对齐）将来自不同的受试者的数据域对齐，又有基于 CSP 手工提取的特征有比较好的线性可分性，故EA+CSP+SVM方法就能得到最好的效果
而由于给出的训练集数据较少，基于统计学习的方法训练不充分，无法有效学习到有效的特征提取模式，所以效果普遍不好

# 原论文
论文如下
He He and Dongrui Wu. Transfer learning for brain–computer interfaces: A euclidean space data alignment approach. IEEE Transactions on Biomedical Engineering, 2020. 
https://ieeexplore.ieee.org/document/8701679
