"""
Jet Substructure Models used in the LogicNets paper
"""

import torch.nn as nn


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