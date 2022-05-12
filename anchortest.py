import time
import torch

def test(a):
    for i in range(a.shape[0]):
        if a[i]>0.5: a[i]+=1
        else: a[i] -=1
    return time.time()

t = torch.rand(1000000).float()
tic = time.time()
print(test(t)-tic)

t=t.cuda()
tic = time.time()
print(test(t)-tic)
