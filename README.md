# SDCN
Structural Deep Clustering Network

# Paper
https://arxiv.org/abs/2002.01633

https://github.com/461054993/SDCN/blob/master/SDCN.pdf

# Dataset
Due to the limitation of file size, the complete data can be found in Baidu Netdisk:

graph: 链接:https://pan.baidu.com/s/1MEWr1KyrtBQndVNy8_y2Lw  密码:opc1

data: 链接:https://pan.baidu.com/s/1kqoWlElbWazJyrTdv1sHNg  密码:1gd4

# Code
```
python sdcn.py --name [usps|hhar|reut|acm|dblp|cite]
```

# Q&A
- Q: Why do not use distribution Q to supervise distribution P directly?<br>
  A: The reasons are two-fold: 1) Previous method has considered to use the clustering assignments as pseudo labels to re-train the encoder in a supervised manner, i.e., [deepCluster](https://arxiv.org/abs/1807.05520). However, in experiment, we find that the gradient of cross-entropy loss is too violent to prevent the embedding spaces from disturbance. 2) Although we can replace the cross-entropy loss with KL divergence, there is still a problem that we worried about, that is, there is no clustering information. The original intention of our research on deep clustering is to integrate the objective of clustering into the powerful representation ability of deep learning. Therefore, we introduce the distribution P to increase the cohesion of clustering performance, the details can be found in [DEC](http://www.jmlr.org/proceedings/papers/v48/xieb16.pdf).

# Reference
If you make advantage of the SDCN model in your research, please cite the following in your manuscript:
```
@article{sdcn2020,
title={Structural Deep Clustering Network},
author={Deyu, Bo and Xiao, Wang and Chuan, Shi and Meiqi, Zhu and Emiao, Lu and Peng, Cui},
journal={WWW},
year={2020}
}
```
