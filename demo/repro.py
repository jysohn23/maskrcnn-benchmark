import torch
import torch_xla
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import pdb
xla_device = xm.xla_device()
xla_device = 'cpu'
d = torch.rand(3, 1, 2).to(xla_device)
a = torch.rand(4, 2).to(xla_device)
b = torch.max(a, d)
pdb.set_trace()
print(b)

