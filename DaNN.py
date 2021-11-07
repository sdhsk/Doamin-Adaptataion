import torch.nn as nn
import torch
#固定写法

class DaNN(nn.Module):
    def __init__(self, n_input=28 * 28, n_hidden=256, n_class=31):
        super(DaNN, self).__init__()#子类DANN继承父类de属性和方法
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(n_hidden, n_class)

    def forward(self, src, tar): #feature extractor
        x_src = self.layer_input(src)
        x_tar = self.layer_input(tar)
        x_src = self.dropout(x_src)
        x_tar = self.dropout(x_tar)
        x_src_mmd = self.relu(x_src)
        x_tar_mmd = self.relu(x_tar)
        y_src = self.layer_hidden(x_src_mmd)
        return y_src, x_src_mmd, x_tar_mmd

