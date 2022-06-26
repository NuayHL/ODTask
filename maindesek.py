import torch.optim.lr_scheduler as sche
import torch.optim as optim
import torch.nn as nn

class Fu(nn.Module):
    def __init__(self):
        super(Fu, self).__init__()
        self.linear1 = nn.Linear(100,10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.linear1(x)
        x = self.softmax(x)
        return x

model = Fu()

opt = optim.Adam(model.parameters(), lr=0.001)
sch = sche.MultiStepLR(opt, [5], 0.1)

print(model.state_dict())
print(opt.state_dict())
print(sch.state_dict())