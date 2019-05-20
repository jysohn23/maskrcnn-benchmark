import torch
import torch.nn.functional as F


def bilinear_interpolate(bottom_data, height, width, y, x):
  y_low = torch.where(y.long() >= height - 1, torch.full_like(y, height - 1),
                      y.long().float()).long()
  y_low = y_low.clamp(0, height - 1)
  x_low = torch.where(x.long() >= width - 1, torch.full_like(x, width - 1),
                      x.long().float()).long()
  x_low = x_low.clamp(0, width - 1)

  y_high = torch.where(y.long() >= height - 1, torch.full_like(y_low, height - 1),
                       y_low + 1)
  y_high = y_high.clamp(0, height - 1)
  x_high = torch.where(x.long() >= width - 1, torch.full_like(x_low, width - 1),
                       x_low + 1)
  x_high = x_high.clamp(0, width - 1)

  y = torch.where(y.long() >= height - 1, y_low.float(), y)
  x = torch.where(x.long() >= width - 1, x_low.float(), x)

  ly = (y - y_low.float()).unsqueeze(1)
  lx = (x - x_low.float()).unsqueeze(0)
  hy = 1. - ly
  hx = 1. - lx

  v1 = bottom_data[:,y_low][:,:,x_low]
  v2 = bottom_data[:,y_low][:,:,x_high]
  v3 = bottom_data[:,y_high][:,:,x_low]
  v4 = bottom_data[:,y_high][:,:,x_high]

  w1 = hy * hx
  w2 = hy * lx
  w3 = ly * hx
  w4 = ly * lx

  val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
  val = torch.where(y < -1, torch.zeros_like(val), val)
  val = torch.where(y > height, torch.zeros_like(val), val)
  val = torch.where(x < -1, torch.zeros_like(val), val)
  val = torch.where(x > height, torch.zeros_like(val), val)

  return val


def tensor_roi_align(bottom_data, spatial_scale, channels, height, width,
                     pooled_height, pooled_width, sampling_ratio, bottom_rois):
  result = []
  for bottom_roi in bottom_rois:
    roi_batch_ind = bottom_roi[0].int()
    bottom_data_channel = bottom_data[roi_batch_ind]
    roi_sizes = bottom_roi[1:5] * spatial_scale
    roi_start_w = roi_sizes[0]
    roi_start_h = roi_sizes[1]
    roi_end_w = roi_sizes[2]
    roi_end_h = roi_sizes[3]

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

    y = roi_start_h + ph.float(
    ) * bin_size_h + y_neigh_offsets * bin_size_h / sampling_ratio
    x = roi_start_w + pw.float(
    ) * bin_size_w + x_neigh_offsets * bin_size_w / sampling_ratio
    interpolated = bilinear_interpolate(bottom_data_channel, height, width, y, x)
    interpolated = interpolated.unsqueeze(0)
    pooled = F.avg_pool2d(interpolated, sampling_ratio)
    pooled = pooled.squeeze(0)
    result.append(pooled)
  return torch.stack(result, dim=0)
