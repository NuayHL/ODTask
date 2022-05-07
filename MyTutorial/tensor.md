## Tensor

### torch.cat()
```
torch.cat(tensor_sequence, dim)
#exmaple
torch.cat((x1,x2),0)
```
将一个tensor序列（元组/列表）在指定的dim合并成为一个tensor

### torch.stack()
```
torch.stack(tensor_sequence,dim)
#exmaple
torch.cat((x1,x2),0)
```
将一个tensor序列（元组/列表）在指定的dim堆叠成为一个新的tensor

torch.cat() 和 torch.stack() 的区别：
```
>>> a = torch.rand((1,2,3))
>>> torch.stack((a,a),0).shape
torch.Size([2,1,2,3])
>>> torch.cat((a,a),0).shape
torch.Size([2,2,3])
```

### torch.squeeze() & torch.unsqueeze()
```
torch.squeeze(tensor,dim:[Optional])
tensor.squeeze(dim:[Optional])

torch.unsqueeze(tensor,dim)
tensor.unsqueeze(dim)
```
squeeze: 将tensor中只有1size的维度去除

unsqueeze: 在指定的维度增加一维