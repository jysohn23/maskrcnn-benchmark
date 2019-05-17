from maskrcnn_benchmark.layers import ROIAlign, tensor_roi_align
import random
import torch
import unittest


class TestROIAlign(unittest.TestCase):

  def _gen_rois(self, batch_size, width, height, rois_count):
    rois = []
    for i in range(0, rois_count):
      batch_id = float(int(random.random() * batch_size))
      start_w = 0.5 * random.random() * width
      start_h = 0.5 * random.random() * height
      end_w = (0.5 + 0.5 * random.random()) * width
      end_h = (0.5 + 0.5 * random.random()) * height
      rois.append(torch.tensor([batch_id, start_w, start_h, end_w, end_h]))
    return torch.stack(rois, dim=0)

  def _gen_data(self, batch_size, channels, width, height):
    return torch.rand([batch_size, channels, width, height])

  def test_roi_align(self):
    rois_count = 512
    batch_size = 8
    width = 64
    height = 64
    channels = 1024
    output_size = (14, 14)
    spatial_scale = .5
    sampling_ratio = 2
    rois = self._gen_rois(batch_size, width, height, rois_count)
    bottom_image_data = self._gen_data(batch_size, channels, width, height)
    roi_align = ROIAlign(
        output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
    aligned_ref = roi_align(bottom_image_data, rois)
    aligned = tensor_roi_align(bottom_image_data, spatial_scale, channels,
                               height, width, output_size[0], output_size[1],
                               sampling_ratio, rois)
    self.assertAlmostEqual((aligned - aligned_ref).abs().max().item(), 0, places=5)


if __name__ == '__main__':
  torch.manual_seed(42)
  random.seed(42)
  unittest.main()
