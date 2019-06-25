from maskrcnn_benchmark.layers import ROIAlign, tensor_roi_align
import random
import torch
import torch_xla
import torch_xla_py.xla_model as xm
import unittest


NUM_ROIS = 512

class TestROIAlign(unittest.TestCase):

  def _gen_rois(self, batch_size, width, height, rois_count):
    rois = []
    for i in range(0, rois_count):
      batch_id = float(int(i / NUM_ROIS))
      start_w = 0.5 * random.random() * width
      start_h = 0.5 * random.random() * height
      end_w = (0.5 + 0.5 * random.random()) * width
      end_h = (0.5 + 0.5 * random.random()) * height
      rois.append(torch.tensor([batch_id, start_w, start_h, end_w, end_h]))
    return torch.stack(rois, dim=0)

  def _gen_data(self, batch_size, channels, width, height):
    return torch.rand([batch_size, channels, width, height])

  def test_roi_align(self):
    batch_size = 1
    rois_count = NUM_ROIS * batch_size
    width = 64
    height = 64
    channels = 1024
    output_size = (14, 14)
    spatial_scale = .5
    sampling_ratio = 2
    rois = self._gen_rois(batch_size, width, height, rois_count)
    rois_3d = rois[:, 1:].reshape(batch_size, NUM_ROIS, -1)
    bottom_image_data = self._gen_data(batch_size, channels, width, height)
    roi_align = ROIAlign(
        output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
    aligned_ref = roi_align(bottom_image_data, rois)
    xla_device = xm.xla_device()
    rois_3d = rois_3d.to(device=xla_device)
    bottom_image_data = bottom_image_data.to(device=xla_device)
    aligned = tensor_roi_align(bottom_image_data, rois_3d, output_size,
                               spatial_scale, sampling_ratio)
    self.assertAlmostEqual(
        (aligned - aligned_ref).abs().max().item(), 0, places=5)


if __name__ == '__main__':
  torch.manual_seed(42)
  random.seed(42)
  unittest.main()
