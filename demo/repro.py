import torch
import torch_xla
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm

a = torch.Tensor()
# a = torch.rand(2, 2)
b = torch.rand(2, 2)
xla_device = xm.xla_device()
import pdb
pdb.set_trace()
a = a.to(xla_device)
b = b.to(xla_device)
e = torch.cat([a, b, a], -1)
# c = b + 1
print(e)

