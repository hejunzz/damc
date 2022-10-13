# Domain Gap Estimation for Source Free Unsupervised Domain Adaptation with Many Classifiers
https://arxiv.org/abs/2207.05785

In theory, the success of unsupervised domain adaptation (UDA) largely relies on domain gap estimation. 
However, for source free UDA, the source domain data can not be accessed during adaptation, which poses 
great challenge of measuring the domain gap. In this paper, we propose to use many classifiers to learn 
the source domain decision boundaries, which provides a tighter upper bound of the domain gap, even if 
both of the domain data can not be simultaneously accessed. The source model is trained to push away each 
pair of classifiers whilst ensuring the correctness of the decision boundaries. In this sense, our many 
classifiers model separates the source different categories as far as possible which induces the maximum 
disagreement of many classifiers in the target domain, thus the transferable source domain knowledge is 
maximized. For adaptation, the source model is adapted to maximize the agreement among pairs of the classifiers. 
Thus the target features are pushed away from the decision boundaries. Experiments on several datasets of 
UDA show that our approach achieves state of the art performance among source free UDA approaches and can 
even compete to source available UDA methods.

## 

## Colab Jupyter notebook for SF UDA demonstration
Please refer to https://sites.google.com/site/hejunzz/da

The pre-trained source model can be downloaded from my Google drive. 

damc.visda.train.cls12.smo0.0.3.10.pth: 
https://drive.google.com/file/d/1lzMr8aab5hOvIpDTC9gLn_g8YCU3Dyy_/view?usp=sharing

damc.visda.train.cls12.smo0.0.4.10.pth: 
https://drive.google.com/file/d/1LX59Nq9gV-xqtBrV7szojDzFYubqI_nX/view?usp=sharing


