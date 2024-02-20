### Lab 4 (Software Stream) for Advanced Deep Learning Systems (ADLS)

#### 1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.

#### 2. In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?

#### 3. You may have noticed, one problem with the channel multiplier is that it scales all layers uniformly, ideally, we would like to be able to construct networks like the following: Can you then design a search so that it can reach a network that can have this kind of structure?

```python
# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ReLU(16),
            nn.Linear(16, 32),  # output scaled by 2
            nn.ReLU(32),  # scaled by 2
            nn.Linear(32, 64),  # input scaled by 2 but output scaled by 4
            nn.ReLU(64),  # scaled by 4
            nn.Linear(64, 5),  # scaled by 4
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)
```

#### 4. Integrate the search to the chop flow, so we can run it from the command line.