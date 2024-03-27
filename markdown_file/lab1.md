### lab1 for Advanced Deep Learning Systems (ADLS)

## 1. What is the impact of varying batch sizes and why?
1. batchsize can affect the running time of training. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.

2. batchsize can affect the overall accuracy but we can't for sure this will improve the accuracy or not. because the smaller batchsize can be seen as a regularization technique, which can prevent the model from overfitting.

3. batchsize can affect the generalization of the model. because the smaller batchsize can be seen as a regularization technique, which can prevent the model from overfitting.

4. batchsize can affect the memory usage. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.

5. batchsize can affect the overall running time. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.

6. batchsize can affect the convergence speed. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.

7. batchsize can affect the stability of the model. because the smaller batchsize can be seen as a regularization technique, which can prevent the model from overfitting.

   Try different batch sizes of the training and the results are shown as follows:

   ###### different batch sizes

   | epoch | batch_size | learning_rate | training loss | validation loss |
   | :---: | :--------: | :-----------: | :-----------: | :-------------: |
   |  50   |     64     |    0.0001     |    0.7837     |      0.878      |
   |  50   |    128     |    0.0001     |    0.9908     |     0.8581      |
   |  50   |    256     |    0.0001     |    0.9283     |      0.859      |
   |  50   |    512     |    0.0001     |     1.075     |      1.053      |
   |  50   |    1024    |    0.0001     |     1.049     |      1.086      |

   From the above form we can see that, different batch sizes have an influence on learning time. the higher the batch size is, the shorter the training time will be. this is because the training procedure is calculated by batch size and the time this batches were sent to the training network.

   ###### 2. What is the impact of varying maximum epoch number?

1. the larger the maximum epoch number, the more iterations are needed to reach the same accuracy. so the epoch number can affect the running time of training.

2. the epoch number can affect the overall accuracy but we can't for sure this will improve the accuracy or not.

3. the epoch number can affect the generalization of the model. this is because the larger the epoch number, the more likely the model will overfit.

4. the epoch number can affect the memory usage. because the larger the epoch number, the more iterations are needed to reach the same accuracy.

   Try different max epoch number, the results are as follows:

   ###### different max epoch

   | epoch | batch_size | learning_rate | training loss | validation loss |
   | :---: | :--------: | :-----------: | :-----------: | :-------------: |
   |  10   |    256     |    0.0001     |    0.9919     |     0.9922      |
   |  50   |    256     |    0.0001     |    0.9283     |      0.859      |
   |  100  |    256     |    0.0001     |     1.073     |     0.8435      |
   |  150  |    256     |    0.0001     |    0.8844     |      0.84       |
   |  200  |    256     |    0.0001     |    0.6919     |     0.8382      |

   The table show the training loss and validation loss with respect to the change of epochs. When the epoch is small than 50, the training loss decreases with increasing epochs. When the epoch is larger than 150, the training loss is much smaller than validation loss, indicating an overfitting problem.

## 3. What is happening with a large learning and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?
1. with a large learning rate, the model will converge faster but the accuracy will be lower. this is because the large learning rate will make the model jump over the optimal point.

2. with a small learning rate, the model will converge slower but the accuracy will be higher. this is because the small learning rate will make the model converge to the optimal point.

3. the relationship between learning rates and batch sizes is that the smaller the batch size, the smaller the learning rate should be. this is because the smaller batch size can be seen as a regularization technique, which can prevent the model from overfitting. and the smaller learning rate can also be seen as a regularization technique, which can prevent the model from overfitting.

   ###### different learning rates

   | epoch | batch_size | learning_rate | training loss | validation loss |
   | :---: | :--------: | :-----------: | :-----------: | :-------------: |
   |  50   |    256     |      0.1      |     1.229     |      1.197      |
   |  50   |    256     |     0.001     |    0.9184     |      0.832      |
   |  50   |    256     |    0.0001     |    0.9283     |      0.859      |
   |  50   |    256     |    0.00001    |     1.214     |      1.094      |
   |  50   |    256     |   0.000001    |     1.388     |      1.354      |

## 4. Implement a network that has in total around 10x more parameters than the toy network.
using the similiar structure as the toy network, I implemented a network that has in total around 10x more parameters. the structure is shown below:
```python
class Test(nn.Module):
    def __init__(self, info):
        super(Test, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 64),  # linear              # 2
            nn.BatchNorm1d(64),  # output_quant       # 3
            nn.ReLU(64),  # 4
            # 2nd LogicNets Layer
            nn.Linear(64, 128),  # 5
            nn.BatchNorm1d(128),  # 6
            nn.ReLU(128),  # 7
            # 3rd LogicNets Layer
            nn.Linear(128, 8),  # 8
            nn.BatchNorm1d(8),  # 9
            nn.ReLU(8),
            nn.Linear(8, 8),  # 5
            nn.BatchNorm1d(8),  # 6
            nn.ReLU(8),  # 7
            # 3rd LogicNets Layer
            nn.Linear(8, 5),  # 8
            nn.BatchNorm1d(5),  # 9
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)

def get_test(info):
    return Test(info)

```
in this network, i name it as `Test`. the network has 9 layers. 


## 5. Test your implementation and evaluate its performance.

using my own implementation, I trained the network using jsc dataset. the other setting of this network in the __init__ function is the same as the toy network. the training result is shown below:

###### Train my own network

Implement new network called **test** with **11.7k** trainable parameters to be trained (jsc-tiny has 127 trained parameters). Train the new network with hyperparameters as follows and evalute the performance in test set. We can get the following results.

| epoch | batch_size | learning_rate | validation acc | validation loss | test acc | test loss |
| :---: | :--------: | :-----------: | :------------: | :-------------: | :------: | :-------: |
|  50   |     64     |    0.00001    |     0.724      |      0.832      |  0.722   |   0.825   |

