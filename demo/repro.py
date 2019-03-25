import torch
import torch_xla
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import pdb
# xla_device = xm.xla_device()
xla_device = 'cpu'
d = torch.ones((1000, 324)).to(xla_device)
dw = d[:, 2::4]
print(dw.shape)
dw = torch.clamp(dw, max=3.1)
pdb.set_trace()
print(dw.shape)
