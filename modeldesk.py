import torch

a = torch.tensor([-1,2,-1.5,4,5],dtype=torch.float,requires_grad=True)
print(a)
a = torch.clamp(a,-2,0)
print(a)
b = a*a
print(a)
c = b.sum()
c.backward()
print(a)