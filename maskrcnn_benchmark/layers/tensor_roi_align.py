import torch
import torch.nn.functional as F


def _bottom_data_slice(bottom_data, y, x):
  y = y.unsqueeze(1).unsqueeze(3)
  y = y.expand(
      bottom_data.size(0), bottom_data.size(1), y.size(2), bottom_data.size(3))
  gathered = torch.gather(bottom_data, 2, y)
  x = x.unsqueeze(1).unsqueeze(2)
  x = x.expand(gathered.size(0), gathered.size(1), gathered.size(2), x.size(3))
  return torch.gather(gathered, 3, x)


def _bilinear_interpolate(bottom_data, height, width, y, x):
  y_low = torch.where(y.long() >= height - 1, torch.full_like(y, height - 1),
                      y.long().float()).long()
  y_low = y_low.clamp(0, height - 1)
  x_low = torch.where(x.long() >= width - 1, torch.full_like(x, width - 1),
                      x.long().float()).long()
  x_low = x_low.clamp(0, width - 1)

  y_high = torch.where(y.long() >= height - 1,
                       torch.full_like(y_low, height - 1), y_low + 1)
  y_high = y_high.clamp(0, height - 1)
  x_high = torch.where(x.long() >= width - 1, torch.full_like(x_low, width - 1),
                       x_low + 1)
  x_high = x_high.clamp(0, width - 1)

  y = torch.where(y.long() >= height - 1, y_low.float(), y)
  x = torch.where(x.long() >= width - 1, x_low.float(), x)

  ly = (y - y_low.float()).unsqueeze(2)
  lx = (x - x_low.float()).unsqueeze(1)
  hy = 1. - ly
  hx = 1. - lx

  v1 = _bottom_data_slice(bottom_data, y_low, x_low)
  v2 = _bottom_data_slice(bottom_data, y_low, x_high)
  v3 = _bottom_data_slice(bottom_data, y_high, x_low)
  v4 = _bottom_data_slice(bottom_data, y_high, x_high)

  w1 = (hy * hx).unsqueeze(1)
  w2 = (hy * lx).unsqueeze(1)
  w3 = (ly * hx).unsqueeze(1)
  w4 = (ly * lx).unsqueeze(1)

  val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
  y = y.unsqueeze(1).unsqueeze(3).expand_as(val)
  x = x.unsqueeze(1).unsqueeze(2).expand_as(val)
  val = torch.where(y < -1, torch.zeros_like(val), val)
  val = torch.where(y > height, torch.zeros_like(val), val)
  val = torch.where(x < -1, torch.zeros_like(val), val)
  val = torch.where(x > height, torch.zeros_like(val), val)

  return val


def tensor_roi_align(bottom_data, bottom_rois, pooled_size, spatial_scale,
                     sampling_ratio):
  pooled_height = pooled_size[0]
  pooled_width = pooled_size[1]

  roi_batch_indices = bottom_rois[:, 0].long()
  roi_sizes = bottom_rois[:, 1:5] * spatial_scale
  roi_start_w = roi_sizes[:, 0]
  roi_start_h = roi_sizes[:, 1]
  roi_end_w = roi_sizes[:, 2]
  roi_end_h = roi_sizes[:, 3]

  roi_width = torch.max(roi_end_w - roi_start_w, torch.ones_like(roi_end_w))
  roi_height = torch.max(roi_end_h - roi_start_h, torch.ones_like(roi_end_h))
  bin_size_h = roi_height / pooled_height
  bin_size_w = roi_width / pooled_width

  pw = torch.tensor(
      [pw for pw in range(pooled_width) for _ in range(sampling_ratio)])
  ph = torch.tensor(
      [ph for ph in range(pooled_height) for _ in range(sampling_ratio)])
  x_neigh_offsets = torch.tensor([pw + .5 for pw in range(sampling_ratio)] *
                                 pooled_width)
  y_neigh_offsets = torch.tensor([ph + .5 for ph in range(sampling_ratio)] *
                                 pooled_height)

  ph = ph.unsqueeze(0)
  pw = pw.unsqueeze(0)
  bin_size_h = bin_size_h.unsqueeze(1)
  bin_size_w = bin_size_w.unsqueeze(1)
  y_neigh_offsets = y_neigh_offsets.unsqueeze(0)
  x_neigh_offsets = x_neigh_offsets.unsqueeze(0)

  roi_start_h = roi_start_h.unsqueeze(1)
  roi_start_w = roi_start_w.unsqueeze(1)
  y = roi_start_h + bin_size_h * ph.float(
  ) + bin_size_h * y_neigh_offsets / sampling_ratio
  x = roi_start_w + bin_size_w * pw.float(
  ) + bin_size_w * x_neigh_offsets / sampling_ratio

  bottom_data = bottom_data[roi_batch_indices]

  height = bottom_data.size(2)
  width = bottom_data.size(3)
  interpolated = _bilinear_interpolate(bottom_data, height, width, y, x)
  return F.avg_pool2d(interpolated, sampling_ratio)
