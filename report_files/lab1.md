### lab1 for Advanced Deep Learning Systems (ADLS)

## 1. What is the impact of varying batch sizes and why?
1. batchsize can affect the running time of training. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.
2. batchsize can affect the overall accuracy but we can't for sure this will improve the accuracy or not. because the smaller batchsize can be seen as a regularization technique, which can prevent the model from overfitting.
3. batchsize can affect the generalization of the model. because the smaller batchsize can be seen as a regularization technique, which can prevent the model from overfitting.
4. batchsize can affect the memory usage. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.
5. batchsize can affect the overall running time. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.
6. batchsize can affect the convergence speed. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.
7. batchsize can affect the stability of the model. because the smaller batchsize can be seen as a regularization technique, which can prevent the model from overfitting.

## 2. What is the impact of varying maximum epoch number?
1. the larger the maximum epoch number, the more iterations are needed to reach the same accuracy. so the epoch number can affect the running time of training.
2. the epoch number can affect the overall accuracy but we can't for sure this will improve the accuracy or not.
3. the epoch number can affect the generalization of the model. this is because the larger the epoch number, the more likely the model will overfit.
4. the epoch number can affect the memory usage. because the larger the epoch number, the more iterations are needed to reach the same accuracy.

## 3. What is happening with a large learning and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?
1. with a large learning rate, the model will converge faster but the accuracy will be lower. this is because the large learning rate will make the model jump over the optimal point.
2. with a small learning rate, the model will converge slower but the accuracy will be higher. this is because the small learning rate will make the model converge to the optimal point.
3. the relationship between learning rates and batch sizes is that the smaller the batch size, the smaller the learning rate should be. this is because the smaller batch size can be seen as a regularization technique, which can prevent the model from overfitting. and the smaller learning rate can also be seen as a regularization technique, which can prevent the model from overfitting.

## 4. Implement a network that has in total around 10x more parameters than the toy network.
using the similiar structure as the toy network, I implemented a network that has in total around 10x more parameters. the structure is shown below:
```python
"""
Jet Substructure Models used in the LogicNets paper
"""

import torch.nn as nn


class Test(nn.Module):
    def __init__(self, info):
        super(Test, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),
            nn.ReLU(16),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(32),  # 4
            # 2nd LogicNets Layer
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(16),
            # 3rd LogicNets Layer
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(8),
            # 3rd LogicNets Layer
            nn.Linear(8, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)

def get_test(info):
    return Test(info)

```
in this network, i name it as `Test`. the network has 4 layers. the 1st LogicNets Layer has 16 input channels and 32 output channels, the 2nd LogicNets Layer has 32 input channels and 16 output channels, the 3rd LogicNets Layer has 16 input channels and 8 output channels, the 4th LogicNets Layer has 8 input channels and 5 output channels.so the number of parameters is 16*32+32+32*16+16+16*8+8+8*5+5=1445.


## 5. Test your implementation and evaluate its performance.

using my own implementation, I trained the network using jsc dataset. the other setting of this network in the __init__ function is the same as the toy network. the training result is shown below:

