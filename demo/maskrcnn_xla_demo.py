
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import torch_xla
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm


pylab.rcParams['figure.figsize'] = 20, 12


# We provide a helper class `COCODemo`, which loads a model from the config file, and performs pre-processing, model prediction and post-processing for us.
#
# We can configure several model options by overriding the config options.
# In here, we make the model run on the CPU

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])


# Let's define a few helper functions for loading images from a URL
def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


def imsave(img):
    plt.imsave('predicted.png', img[:, :, [2, 1, 0]])


# Let's now load an image from the COCO dataset. It's reference is in the comment
# from http://cocodataset.org/#explore?id=345434
image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
# imshow(image)

# ### Computing the predictions
#
# We provide a `run_on_opencv_image` function, which takes an image as it was loaded by OpenCV (in `BGR` format), and computes the predictions on them, returning an image with the predictions overlayed on the image.

# compute predictions
xla_device = xm.xla_device()
# Now we create the `COCODemo` object. It contains a few extra options for conveniency, such as the confidence threshold for detections to be shown.

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    device=xla_device,
)

import pdb
pdb.set_trace()

predictions = coco_demo.run_on_opencv_image(image)

# imshow(predictions)
imsave(predictions)
