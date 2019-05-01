import torch
import torch_xla
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import pdb
import torch.nn.functional as F
xla_device = xm.xla_device()
xla_device = 'cpu'
# objectness = torch.load('objectness.pt').to(xla_device)
pdb.set_trace()
# labels = torch.load('labels.pt').to(xla_device).to(dtype=torch.uint8)
# sampled_inds = torch.load('sampled_inds.pt').to(xla_device)
a = torch.rand(10).to(xla_device).to(dtype=torch.uint8)
b = torch.tensor([0, 1, 2, 3]).to(xla_device)
print(a.dtype)
print(a[b].dtype)
a = torch.rand(10).to(xla_device)
b = torch.tensor([0, 1, 2, 3], dtype=torch.int64).to(xla_device)
print(a.dtype)
print(a[b].dtype)

# print(labels.dtype)
# print(labels[sampled_inds].dtype)
# loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

# print(loss)

