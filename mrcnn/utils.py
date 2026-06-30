"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import cv2
import urllib.request
import shutil
import zipfile
import warnings
from distutils.version import LooseVersion
from tqdm.notebook import tqdm
from PIL import Image

#confusion matrix 

from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from string import ascii_uppercase
import mrcnn.model as modellib
import pandas as pd
import StereoFractAnalyzer as SF
import tempfile
import csv
import re

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

#URL from which to download zip of SAGE pretrained models
PRETRAIN_URL = "https://github.com/timday23/SAGE_TPD/releases/download/v1.0.0/pretrained_models.zip"


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def compute_DICE(masks1,masks2):
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :]
    
    dice = (2.*intersections)/ union
    
    return dice
    
    


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path,basename, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
            "basename": basename, #add basename to be able to retrieve original file name 
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # print(image_dtype)
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m.astype(np.float32), mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m.astype(np.float32), (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=0):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        iou_threshold=iou_threshold)
        if verbose >= 2:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose >= 2:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def download_pretrained_models(pretrained_dir_path, verbose=1):
    """Download SAGE trained weights from Releases.

    pretrained_dir_path: local path of SAGE pretrained weights
    """
    os.makedirs(pretrained_dir_path, exist_ok=True)

    zip_path = os.path.join(pretrained_dir_path, "pretrained_models.zip")
    
    if verbose > 0:
        print("Downloading pretrained models to " + pretrained_dir_path + " ...")

    with urllib.request.urlopen(PRETRAIN_URL) as resp, open(zip_path, 'wb') as out:
        shutil.copyfileobj(resp, out)

    if verbose > 0:
        print("Download complete! Extracting...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(pretrained_dir_path)

    #remove zip file after extraction
    os.remove(zip_path)
    
    if verbose > 0:
        print("... done downloading pretrained models!")

def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

    


###### Added utils for SAGE ########

class SAGEDataset(Dataset):
    """Load Dataset
    """
    def __init__(self,images_dir, particle_masks_dir, cluster_masks_dir, load_particle=True, load_cluster=True, load_cluster_as_input=False):
        super().__init__()
        self.images_dir = images_dir
        self.particle_masks_dir = particle_masks_dir
        self.cluster_masks_dir = cluster_masks_dir
        self.load_particle = load_particle  # Correctly initialize the attribute
        self.load_cluster = load_cluster  # Correctly initialize the attribute
        self.load_cluster_as_input = load_cluster_as_input
 
        self.manifest = {}   # image_id (basename or new_image_id) → full row
        self.scales = {}     # image_id → scale (nm/pix)
   
        if load_particle and load_cluster:
            self.class_names = ["particle", "cluster"]
            self.add_class("SAGE",1,"particle") #add particle class
            self.add_class("SAGE",2,"cluster") #add cluster class 
        elif load_particle:
            self.class_names=["particle"]
            self.add_class("SAGE",1,"particle") #add particle class
        elif load_cluster:
            self.class_names=["cluster"]
            self.add_class("SAGE",1,"cluster") #add cluster class 

        self.gt_available = self.load_particle or self.load_cluster
            
        #print(self.class_names)
    
    def load_dataset(self, dataset_name=None, mask_position=2):
        """Load images and masks from specified directories."""

        self._load_manifest()


        #load images
        image_filenames = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        #print(f"unsorted: {image_filenames}")
        #sort them by number
        image_filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        #print(f"Sorted: {image_filenames}")
        
        for image_id, filename in enumerate(image_filenames):
            #tqdm(image_filenames, desc="Adding images", dynamic_ncols=True,position=1)):   removing because it already loads fast
            image_path = os.path.join(self.images_dir, filename)
            basename = os.path.splitext(filename)[0] 

            #look up manifest info
            manifest_row = self.manifest.get(basename)
            scale = self.scales.get(basename)
          

            og_image_id = manifest_row["original_image_id"] if manifest_row else None
            source_dataset = manifest_row["source"] if manifest_row else None
            segmenter = manifest_row["segmenter"] if manifest_row else None

            with Image.open(image_path) as im:
                original_width, original_height = im.size

            #print(basename)
            self.add_image("SAGE", image_id=image_id,
                            path=image_path,
                            basename=basename,
                            width=original_width, height=original_height,
                            scale=scale,
                            og_image_id = og_image_id,
                            source_dataset = source_dataset,
                            segmenter=segmenter)
            
        #load masks for each image
        desc = f"Loading masks for {dataset_name}" if dataset_name else "Loading masks"
        for image_id in tqdm(range(len(self.image_info)), desc=desc, dynamic_ncols=True, position=mask_position, leave=False):
            #print(f"Loading Masks for Image {image_id}", end="\r")
            #sys.stdout.flush()
            self.load_mask(image_id)

        

    def load_image(self, image_id):
        """ Load an image from the dataset."""
        info = self.image_info[image_id]
        image = cv2.imread(info['path'])
        return image

    def image_reference(self, image_id):
        """Return the particle data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "SAGE":
            return info["path"]
        else:
            return super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """load instance masks for the particle of the given image ID."""
        
        info = self.image_info[image_id]
        masks = []
        class_ids = []
        
        if self.load_particle and self.load_cluster:
            masks_particle, class_ids_particle = self._load_class_masks(info, self.particle_masks_dir, 
                                                                        class_id=1,pattern='particle')
            #if masks_particle:
                #print(f"Loaded {len(masks_particle)} particle masks for Image ID {image_id}.")
            masks.extend(masks_particle)
            class_ids.extend(class_ids_particle)
            
            masks_cluster, class_ids_cluster = self._load_class_masks(info, self.cluster_masks_dir, 
                                                                      class_id=2,pattern='cluster')
            #if masks_cluster:
               # print(f"Loaded {len(masks_cluster)} cluster masks for Image ID {image_id}.")
            masks.extend(masks_cluster)
            class_ids.extend(class_ids_cluster)
            #print("Both particle and cluster masks")
            
        #particle masks
        elif self.load_particle:
            masks_particle, class_ids_particle = self._load_class_masks(info, self.particle_masks_dir, 
                                                                        class_id=1,pattern='particle')
            #if masks_particle:
                #print(f"Loaded {len(masks_particle)} particle masks for Image ID {image_id}.")
            masks.extend(masks_particle)
            class_ids.extend(class_ids_particle)
            #rint("only particle masks")
            
        #cluster masks
        elif self.load_cluster:
            masks_cluster, class_ids_cluster = self._load_class_masks(info, self.cluster_masks_dir, 
                                                                      class_id='1',pattern='cluster')
            #if masks_cluster:
               # print(f"Loaded {len(masks_cluster)} cluster masks for Image ID {image_id}.")
            masks.extend(masks_cluster)
            class_ids.extend(class_ids_cluster)
           #print(("only cluster masks"))
            
        #combine masks into 3d array
        if masks:
            combined_mask = np.stack(masks, axis =-1)
            return combined_mask, np.array(class_ids, dtype=np.int32)
                      
        #print(f" No masks found for image ID {image_id}.")
        return np.zeros((0,0), dtype=np.bool_),np.zeros((0,),dtype=np.int32)

    def load_cluster_input_masks(self, image_id):
        if not self.load_cluster_as_input:
            return np.zeros((0,0,0), dtype= np.bool_)

        info = self.image_info[image_id]
        masks, _, = self._load_class_masks(info, self.cluster_masks_dir, class_id=None, pattern="cluster")

        if masks:
            return np.stack(masks, axis=-1) #[H, W, N_clusters]
        else:
            return np.zeros((info.get("height", 0), info.get("width",0), 0) ,dtype=np.bool_ )
    
    def _load_class_masks(self,info, masks_dir, class_id, pattern):
        """Load msks for a specific class based on a pattern"""
        
        masks = []
        class_ids = []
        
        #construct mask filename based on image filename 
        _, image_filename = os.path.split(info['path']) 
        image_no = image_filename.split('_')[1].replace('.png','') #extract the base name without the extension to form mask filename
        #print(image_no)
        #print(f"Loading masks for image number:{image_no}")
        
        #load all masks for the current image
       
        
        if pattern in ['particle', 'cluster']:
            i = 0
            first_mask_found = False
            while True: 
                mask_filename = f"mask_{image_no}_{i:06d}.png"
                mask_path = os.path.join(masks_dir, mask_filename)
                
                if os.path.exists(mask_path):
                    #print(f"Found mask file: {mask_path}")
                    first_mask_found=True
                    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE) #load mask
                    if mask is not None:
                        masks.append(mask.astype(np.bool_))
                        class_ids.append(class_id)
                    i += 1
                elif not first_mask_found:
                    #try starting at index 1
                    if i == 0:
                        i=1
                        continue
                    else:
                        #if no indexed masks found, try single-mask naming fallback ('mask_000001.png', etc)
                        single_mask_filename = f"mask_{image_no}.png"
                        single_mask_path = os.path.join(masks_dir, single_mask_filename)
                        if os.path.exists(single_mask_path):
                            mask = cv2.imread(single_mask_path, cv2.IMREAD_GRAYSCALE) #load single mask
                            if mask is not None:
                                masks.append(mask.astype(np.bool_))
                                class_ids.append(class_id)
                        
                        break
                else:
                    break
            


        #elif pattern == 'cluster':
        #    # For clusters, load only one mask
        #    mask_filename = f"mask_{image_no}.png"
        #    mask_path = os.path.join(masks_dir, mask_filename)
        #    #print(f"Checking path for mask: {mask_path}")

         #   if os.path.exists(mask_path):
                #print(f"Found mask file: {mask_path}")
        #        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Load mask
        #        if mask is not None:
        #            masks.append(mask.astype(np.bool_))
        #            class_ids.append(class_id)
        #    else:
        #        print(f"Mask file not found: {mask_path}")
        
        
        return masks, class_ids

    def _load_manifest(self):
        """ Load manifest file metadata (scale, source, segmenter, etc)"""

        self.manifest = {}
        self.scales = {}
        
        manifest_path = os.path.join(self.images_dir, "manifest.csv")
        if not os.path.exists(manifest_path):
            print(f"[WARN] manifest.csv not found in {self.images_dir}")
            return

        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)

            #minimal column check
            required_cols = ["new_image_id", "original_image_id","scale", "source","segmenter"]
            for col in required_cols:
                if col not in reader.fieldnames:
                    print(f"[WARN] Manifest missing required column: {col}")


            for row in reader:
                key = row["new_image_id"].strip() #must match image basename

                #original scale
                try:
                    original_scale = float(row.get("scale",1.0))
                except ValueError:
                    original_scale = 1.0
                    print(f"[WARN] Invalid scale for image {key}: {row['scale']}")

                # new scale (fallback to original if missing)
                try:
                    new_scale = float(row.get("new_scale", original_scale))
                except ValueError:
                    new_scale = original_scale
                    print(f"[WARN] Invalid new_scale for image {key}: {row.get('new_scale')}")

                # Rescale factor (default 1.0)
                try:
                    rescale_factor = float(row.get("rescale_factor", 1.0))
                except ValueError:
                    rescale_factor = 1.0
                    print(f"[WARN] Invalid rescale_factor for image {key}: {row.get('rescale_factor')}")

                # Crop coordinates (keep as string, parse later if needed)
                crop_coords = row.get("crop_coords", None)

                #save into manifest
                self.manifest[key] = row
                self.manifest[key]["_original_scale"] = original_scale
                self.manifest[key]["_rescale_factor"] = rescale_factor
                self.manifest[key]["_crop_coords"] = crop_coords

                if new_scale is None:
                    print("Scale fallback: setting to OG scale")
                    new_scale=original_scale
                self.scales[key] = new_scale
    
    def get_class_labels(self, class_ids):
        """ Return class labels for given ID's, skipping background (0)"""
        return [self.class_names[c] for c in np.unique(class_ids) if c != 0]
                                 
        
                     


def create_dataset_results_dirs(dataset_name, results_dir):
    """creates the full results directory subfolders for given dataset
    Params:
    - dataset_name(str): name of dataset (e.g. D1e1_test)
    - results_dir (str): path to main results folder
    
    returns:
    -str: path to dataset's results directories
    """
    
    dataset_dir = os.path.join(results_dir,dataset_name)
    
    subdirs = [
        os.path.join(dataset_dir, "PP_Info"),
        os.path.join(dataset_dir, "Visualizations"),
        os.path.join(dataset_dir, "IoUs")
    ]
    
    for path in subdirs:
        os.makedirs(path, exist_ok=True)
    #print("Directories Created")
    return dataset_dir

def load_and_register_dataset(dataset_name, ROOT_DIR, results_dir, 
                              load_particle=True, load_cluster=False, load_cluster_as_input=False, 
                              create_dirs=True):
  
    images_anlyz_dir =  os.path.join(ROOT_DIR, 'data', dataset_name)    
    particle_masks_anlyz_dir = os.path.join(images_anlyz_dir, 'particle')
    cluster_masks_anlyz_dir = os.path.join(images_anlyz_dir, 'cluster')
    
    if create_dirs:                      
        dataset_results_dir = create_dataset_results_dirs(dataset_name, results_dir)
    #initalize and load dataset
    
    dataset_analyze = SAGEDataset(images_anlyz_dir,particle_masks_anlyz_dir, cluster_masks_anlyz_dir,
                                load_particle=load_particle, load_cluster=load_cluster, 
                                load_cluster_as_input=load_cluster_as_input)
    
    dataset_analyze.load_dataset(dataset_name=dataset_name)
    dataset_analyze.prepare()
    
    #add loaded dataset to dictionary
    #datasets[dataset_name] = dataset_analyze
    
    return dataset_analyze



def print_loaded_datasets(loaded_datasets):
    """
    Prints the names of all the loaded datasets.
    
    Args:
        loaded_datasets (dict): Dictionary of loaded datasets where keys are dataset names.
        
    Usage:
        print_loaded_datasets(loaded_datasets)
    """
    if not loaded_datasets:
        print("No datasets loaded.")
        return
    
    print("Loaded Datasets:")
    for dataset_name, dataset in loaded_datasets.items():
        print(f"- {dataset_name}")
        if hasattr(dataset,'class_names') and dataset.class_names:
            print("   Classes and IDs:")
            for idx, class_name in enumerate(dataset.class_names):
                print(f"    ID {idx}: {class_name}")
        else: 
            print("   No class information available")
        print("-" * 40)


def inspect_dataset_samples(dataset, num_images=3, show_mask_info=True):
    """
    Inpect Manifest information for a few samples from a dataset
    
    Args: 
        dataset: Dataset object
        num_images (int): Number of images to inspect
        show_mask_info (bool): Whether to print mask/class details

    """

    image_ids = dataset.image_ids

    for image_id in image_ids[:num_images]:
        image_info = dataset.image_info[image_id]

        print(f"Image ID: {image_id}")

        print(f"Basename: {image_info.get('basename')}")
        print(f"Original Image ID: {image_info.get('og_image_id')}")
        print(f"Source Dataset: {image_info.get('source_dataset')}")
        print(f"Segmenter: {image_info.get('segmenter')}")
        print(f"Scale: {image_info.get('scale')}")
        print(f"Dimensions (WxH): {image_info.get('width')} x {image_info.get('height')}")

        if show_mask_info and hasattr(dataset, "load_mask"):
            mask, class_ids = dataset.load_mask(image_id)

            print(f"class ids: {class_ids}")

            if hasattr(dataset,"get_class_labels"):
                labels = dataset.get_class_labels(class_ids)
                print(f"Class labels: {labels}")

        print("")
        

   


        
def print_active_models(model_dict):
    """
    Prints the names of active models
    Args: model_list (list): list of model names (strings)
    usage:
        print_active_models(model_list)
    """
    
    if not model_dict:
        print("No active models found")
        return
    for idx, (model_name, model_info) in enumerate(model_dict.items(), start=1):
        model = model_info.get('model')  # Get the actual model object
        threshold = model_info.get('confidence')  # Get the confidence threshold (if available)
        target_class = model_info.get('target_class', None)
        model_config = model_info.get('config', None)
        image_size = model_config.IMAGE_MAX_DIM if model_config else "Unknown"
        
        config_name = type(model_config).__name__ if model_config else "Unknown"
        print(f"Model {idx}: {model_name}")
        #print(f"  Model Object: {model}")  # Prints the model instance, could be customized further to show more details
        print(f"  Confidence Threshold: {threshold}")
        print(f"  Target Class: {target_class}")
        print(f"  Config: {config_name}")
        print(f"  Config IMAGE_MIN_DIM: {model_config.IMAGE_MIN_DIM if model_config else 'Unknown'}")
        #print(f"  Molded Image Size (IMAGE_MAX_DIM): {molded_size}")
        print("-" * 40)  # Separating line for readability
    

    
def print_iou_summary(iou_df, model_name):
    print(f"Model: {model_name}")


    if iou_df is None or len(iou_df) ==0:
        print(f"No IoU data available (likely no Ground Truth or no matches computes)")
        return
    

    print(iou_df.head())
    print(f"Count: {len(iou_df)}")

    if "iou" not in iou_df.columns:
        print("IoU column not present (GT not available or IoU not computed)")
        return

    print(f"Range: {iou_df['iou'].min():.4f} - {iou_df['iou'].max():.4f}")
    print(f"Mean: {iou_df['iou'].mean():.4f}")


    iou_vals = iou_df["iou"].dropna()

    if len(iou_vals) ==0:
        print("IoU column exists but contains no valid values")
        return
    
    count_50 = (iou_df['iou'] >= 0.5).sum()
    count_75 = (iou_df['iou'] >= 0.75).sum()
    
    print(f"IoU ≥ 0.50: {count_50}")
    print(f"IoU ≥ 0.75: {count_75}")


class ResultsCache: #### EDIT PRINTOUTS AND IMPROVE VERBOSITY STUFF
    """ 
    Central location to house loaded GT and prediction masks, as well as other info"""

    def __init__(self, datasets: dict, model_dict: dict, results_dir: str, verbose:int=1):
        self.datasets = datasets or {} # {dataset_name: dataset_obj}
        self.model_dict = model_dict or {}  # {model_name: model_info dict}
        self.results_dir= results_dir
        self.verbose = verbose
        
        self.gt_data = {}               # {dataset_name: {image_id: (gt_class_ids, gt_bbox, gt_mask)}}
        self.gt_scale = {}   
        self.gt_cluster_data = {}       # {dataset_name: {image_id: gt_cluster_masks}}
        self.model_preds = {}           # {model_name: {dataset_name: {image_id: result_dict}}}
        self.processed_matches = {}     # {cache_key: filtered_matches_df}
    
    def _log(self, msg, level=1):
        if self.verbose >= level:
            print(f"[ResultsCache] {msg}")

    def register_datasets_and_models(self, new_datasets=None, new_models=None, update=False):
        """
        Safely add or update multiple datasets or models.
        - new_datasets: dict {dataset_name: dataset_obj}
        - new_models: dict {model_name: model_info}
        - update: if True, replaces objects with same name but keeps existing cached GT/predictions
        """
        if new_datasets:
            for name, ds in new_datasets.items():
                if name in self.datasets:
                    if update:
                        self._log(f"Updating dataset '{name}' object but keeping cached GT/preds.", level=1)
                        self.datasets[name] = ds
                    else:
                        self._log(f"Dataset '{name}' already exists. Skipping.", level=1)
                else:
                    self.datasets[name] = ds
                    self._log(f"Added new dataset '{name}' to cache.", level=1)

        if new_models:
            for name, mi in new_models.items():
                if name in self.model_dict:
                    if update:
                        self._log(f"Updating model '{name}' object but keeping cached predictions.", level=1)
                        self.model_dict[name] = mi
                    else:
                        self._log(f"Model '{name}' already exists. Skipping.", level=1)
                else:
                    self.model_dict[name] = mi
                    self._log(f"Added new model '{name}' to cache.", level=1)

                    
    def requires_gt(self, dataset_name):
        dataset = self.datasets.get(dataset_name, None)
        if dataset is None:
            return False
        return getattr(dataset, "gt_available", False)

    def get_method_info(self, key):
        """
        Return standard metadata info about model/dataset for lookup purposes
        """

        if key in self.model_dict:
            
            model_info = self.model_dict[key]

            return {
                "method": key,
                "is_model": True,
                "confidence": model_info.get("confidence",None),
                "target_class": model_info.get("target_class",None),
                "sort_method": "confidence",

            }

        elif key in self.datasets:

            ds = self.datasets[key]

            return {
                "method": key,
                "is_model":False,
                "confidence":"N/A",
                "target_class": getattr("target_class", None),
                "sort_method": "iou",
            }

        raise ValueError(
            f"'{key}' not found in model_dict or datasets"
        )


    def _make_match_key(self, dataset_name, model_name=None, gt2_name=None, sort_method='iou',iou_threshold=0):
        # unique key for caching processed matches
        return f"{dataset_name}|{model_name}|{gt2_name}|{sort_method}|{iou_threshold}"
    
    #Load/cache GT masks
    def get_gt(self, dataset_name, config, image_id):
        if dataset_name not in self.gt_data:
            self.gt_data[dataset_name] = {}

        if image_id in self.gt_data[dataset_name]:
            self._log(f"GT cache hit -> {dataset_name}, image_id={image_id}", level=2)
            return self.gt_data[dataset_name][image_id]

        dataset = self.datasets[dataset_name]

        #Check if it has GT masks or not

        if not self.requires_gt(dataset_name):

            self._log(f"No GT masks available for {dataset_name}, image_id={image_id}, Loading Image Only", level=1)
            image = dataset.load_image(image_id)
            image_meta = np.array([])
            gt_class_ids = np.array([])
            gt_bbox = np.zeros((0, 4))
            gt_mask = np.zeros((0, 0, 0))
            self.gt_data[dataset_name][image_id] = (image, image_meta, gt_class_ids, gt_bbox, gt_mask)
            self.gt_scale.setdefault(dataset_name, {})[image_id] = 1.0
            
            #self.gt_scale[dataset_name][image_id] = 1.0  # default scale
            return image, image_meta, gt_class_ids, gt_bbox, gt_mask


        
        self._log(f"Loading GT for {dataset_name}, image_id={image_id}", level=1)
        image, image_meta, gt_class_ids, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id)


         #compute effective scale
        info = dataset.image_info[image_id]
        original_height = info.get('height', image.shape[0])
        original_width = info.get('width', image.shape[1])
        original_scale = info.get('scale')
        if original_scale is None:
            original_scale = 1.0


        #model input often is resized; calc effective scale
        resized_height, resized_width = image.shape[:2]
        effective_scale = original_scale * (original_height / resized_height)
        self.gt_scale.setdefault(dataset_name, {})[image_id] = effective_scale
        self.gt_data[dataset_name][image_id] = (image,image_meta, gt_class_ids, gt_bbox, gt_mask)

        return image, image_meta, gt_class_ids, gt_bbox, gt_mask
    
    def get_gt_cluster_masks(self, dataset_name, image_id):
        """Load or return cached GT cluster masks"""
        dataset=self.datasets[dataset_name]

        if dataset_name not in self.gt_cluster_data:
            self.gt_cluster_data[dataset_name] = {}
        #Cache hit
        if image_id in self.gt_cluster_data[dataset_name]:
            #self._log(f"GT cluster cache hit → {dataset_name}, image_id={image_id}", level=2)
            return self.gt_cluster_data[dataset_name][image_id]

        if dataset.load_cluster_as_input:
            cluster_masks = dataset.load_cluster_input_masks(image_id)
            #self._log(f"Loaded GT cluster input masks for {dataset_name}, image_id = {image_id}", level=1)
        else:
            cluster_masks = np.zeros((0,0,0), dtype=np.bool_)
            #self._log(f"No GT input masks available for {dataset_name}, image_id = {image_id}", level=1)

        self.gt_cluster_data[dataset_name][image_id] = cluster_masks
        return cluster_masks 

    #Load/cache model predictions
    def get_model_pred(self, model_name,dataset_name, config, image_id, verbose=0):

       
        if model_name not in self.model_preds:
            self.model_preds[model_name]= {}
        if dataset_name not in self.model_preds[model_name]:
            self.model_preds[model_name][dataset_name] = {}
        if image_id in self.model_preds[model_name][dataset_name]:
            self._log(f"Prediction cache hit → {model_name} on {dataset_name}, image={image_id}", level=2)
            return self.model_preds[model_name][dataset_name][image_id]
        
        #self._log(f"Running inference for {model_name} on {dataset_name}")
        dataset = self.datasets[dataset_name]

        # Load image and imag meta
        if not getattr(dataset, 'load_particle',False) and not getattr(dataset,'load_cluster', False):
            image = dataset.load_image(image_id)
            
            #Minimal image meta
            image_meta = modellib.compose_image_meta(
                image_id, image.shape, image.shape, [0,0,image.shape[0], image.shape[1]],
                1.0, np.zeros([dataset.num_classes], dtype=np.int32)
            )
        else:
            image, image_meta, _,_,_ = modellib.load_image_gt(dataset, config,image_id)

        #image = dataset.load_image(image_id)

        #compute effective scale
        info = dataset.image_info[image_id]
        original_height = info.get('height', image.shape[0])
        original_width = info.get('width', image.shape[1])
        original_scale = info.get('scale', 1.0)
       


        #model input often is resized; calc effective scale
        resized_height, resized_width = image.shape[:2]
        effective_scale = original_scale * (original_height /resized_height)
        #prepare optional gt clusters
        gt_clusters = None
        if getattr(dataset, "load_cluster_as_input", False) and getattr(config, "PP_USE_GT_CLUSTERS", False):


            #only load if they exist in dataset and toggled by config
            gt_clusters = self.get_gt_cluster_masks(dataset_name, image_id)


            if gt_clusters.size != 0:
                
                og_img = dataset.load_image(image_id)
                #compute scale, window, padding, crop
                _, window, scale, padding, crop = resize_image(
                    og_img,
                    min_dim=config.IMAGE_MIN_DIM,
                    min_scale = config.IMAGE_MIN_SCALE,
                    max_dim = config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE
                )            

                gt_clusters = resize_mask(gt_clusters, scale, padding, crop)


            else:
                gt_clusters = None
           
 
        model = self.model_dict[model_name]['model']
        result = model.detect([image],
                              gt_cluster_masks=[gt_clusters] if gt_clusters is not None else None,
                               verbose=verbose)[0]

        result['effective_scale'] = effective_scale

        self.model_preds[model_name][dataset_name][image_id] = result

        

        return result

    #Cache/Retrieve filtered matches
    def get_processed_matches(self, dataset_name, model_name=None, gt2_name=None, 
                              sort_method='iou', iou_threshold=0, recompute=False, **kwargs):
        key  = self._make_match_key(dataset_name, model_name, gt2_name, sort_method, iou_threshold)
        print(f"Computed cache key: {key}")
        print(f"Keys currently in cache: {list(self.processed_matches.keys())}")
        
        if not recompute and key in self.processed_matches:
            self._log(f"Match cache hit → key={key}", level=1)
            return self.processed_matches[key]

        self._log(f"Computing new matches → key={key}", level=1)

        #Get datasets(s)
        dataset = self.datasets[dataset_name]
        gt2_dataset = self.datasets.get(gt2_name) if gt2_name else None

        if not self.requires_gt(dataset_name):
            self._log(f"Skipping matches: no GT available for {dataset_name}", level=1)
            return pd.DataFrame()
        #pass kwargs to process matches 
        filtered_df = process_matches(
            self,
            dataset_name,
            inference_config = kwargs.get('inference_config', None),
            model_name = model_name,
            gt2_name = gt2_name,
            sort_method = sort_method,
            iou_threshold = iou_threshold,
            verbose = kwargs.get('verbose', False),
            filter = kwargs.get('filter', False)
            )
        
        self._log(f"Caching processed matches → {key}", level=1)
        self.processed_matches[key] = filtered_df
        return filtered_df



    
def get_overlaps(cache, model_name, dataset_name, config, 
                 iou_threshold=0, sort_method='iou', verbose=False,
                 gt2_name =None, filter=False):
    
    
    iou_values = [] #list to store IoU values
    dataset = cache.datasets[dataset_name]
    model = cache.model_dict[model_name]['model'] if model_name else None
    image_ids = dataset.image_ids
    if not verbose:
        image_ids = tqdm(image_ids, desc="Computing Overlaps", unit="img")
        
    for image_id in image_ids:
        
        #load GT data
        image, image_meta, gt_class_ids, gt_bbox, gt_mask =cache.get_gt(dataset_name,config, image_id)
        #\
            
            #modellib.load_image_gt(dataset, config, image_id)
        
        #if no model is passed, load second set of GT data
        if model is None:
            if gt2_name is None:
                raise ValueError("Either a model or second ground truth set must be provided")
           
           
           
            
            _,_,gt2_class_ids, gt2_bbox, gt2_mask = cache.get_gt(gt2_name, config, image_id)
            #\
            #    modellib.load_image_gt(gt_set2, config, image_id)
            if filter:
                gt2_mask = filter_mask_size(gt2_mask, min_dp_pix = 18)
            #calculate IoUs between GT masks    
            overlaps = compute_overlaps_masks(gt_mask, gt2_mask)
            
            if verbose:
                print(f"Image ID: {image_id}")
                print(f"GT Set 1 Bboxes: {gt_bbox.shape[0]} (GT Boxes), GT Set 2 Bboxes: {gt2_bbox.shape[0]} (Pred Boxes)")
                print(f"Overlaps matrix shape: {overlaps.shape}")
                #print(f"Overlaps: {overlaps}")
                #print(f"Predicted Scores: {len(r['scores'])} scores")
                #print(f"gt_match: {gt_match}, pred_match: {pred_match}")
                
                
        # if model is passed
        else:
            #print('Model Passed')
            #run detection
            result = cache.get_model_pred(model_name, dataset_name,config,  image_id) #model.detect([image], verbose=0)
            r= result #results[0]

        
            if sort_method == 'confidence':
            #print(f"ROIs: {len(r['rois'])}, Scores: {len(r['scores'])}")
                gt_match, pred_match, overlaps = compute_matches(gt_bbox, gt_class_ids, gt_mask, 
                                                             r["rois"], #pred bboxes
                                                             r["class_ids"], #pred class ids
                                                             r["scores"], #pred scores
                                                             r["masks"], #pred masks
                                                             iou_threshold=iou_threshold
                                                            )
            if sort_method == 'iou':
                overlaps1 = compute_overlaps_masks(gt_mask, r['masks'])
                
                pseudo_scores = np.max(overlaps1,axis=0)
                sorted_ix = np.argsort(pseudo_scores)[::-1]
                
                pred_boxes= r['rois'][sorted_ix]
                pred_class_ids= r['class_ids'][sorted_ix]
                pseudo_scores = pseudo_scores[sorted_ix]
                pred_masks = r['masks'][..., sorted_ix]
                
                gt_match, pred_match, overlaps = compute_matches(gt_bbox, gt_class_ids, gt_mask, 
                                                             pred_boxes, #pred bboxes
                                                             pred_class_ids, #pred class ids
                                                             pseudo_scores, #pred scores
                                                             pred_masks, #pred masks
                                                             iou_threshold=iou_threshold
                                                            )
        
            if verbose:
                print(f"Image ID: {image_id}")
                print(f"Ground Truth Bboxes: {gt_bbox.shape[0]} (GT Boxes), Predicted Bboxes: {r['rois'].shape[0]} (Pred Boxes)")
                print(f"Overlaps matrix shape: {overlaps.shape}")
                #print(f"Overlaps: {overlaps}")
                #print(f"Predicted Scores: {len(r['scores'])} scores")
                #print(f"gt_match: {gt_match}, pred_match: {pred_match}")
            max_iou_per_pred = {}
        
        for i in range(overlaps.shape[0]): #loop through pred boxes
            for j in range(overlaps.shape[1]): #loop through gt boxes
                if overlaps[i,j] >= iou_threshold:
                    entry = {
                        'image_id':dataset.image_info[image_id]['id'],
                        'gt_index':j,
                        'pred_index': i,
                        'iou': overlaps[i,j]
                        
                    }
                    
                        
                    iou_values.append(entry)
                        #'confidence score': r['scores'][i] 
                
                        
        #print(iou_values)
    return iou_values


def process_matches(cache,  dataset_name, inference_config, 
                    model_name=  None, 
                    gt2_name=None, 
                    sort_method = 'iou',
                    iou_threshold=0, verbose=False, filter=False): 
    """
    Processes prediction/ground truth matches for each image based on IoU scores, and returns a filtered DataFrame of the matches
    
    Parameters:
     
    """
    if model_name:
        print(f"[Cache] Using model: {model_name}")
        print(f"--> {sort_method.capitalize()} sort")
            
            #ious = get_overlaps(model=model,dataset=dataset_analyze, 
            #                    config=inference_config,sort_method='confidence',
            #                     iou_threshold=iou_threshold,verbose=verbose)
        ious = get_overlaps(cache, model_name, dataset_name, inference_config, 
                            iou_threshold=iou_threshold, sort_method=sort_method, 
                            verbose=verbose, gt2_name = None, filter=filter
                            )
        #else:
        #    print("--> IoU sort")
        #    ious = get_overlaps(model=model,dataset=dataset_analyze, 
        #                        config=inference_config,sort_method='iou',
        #                         iou_threshold=iou_threshold,verbose=verbose)
            
    elif gt2_name:
        print(f"[Cache] Comparing GT sets: {dataset_name} vs {gt2_name}")
        ious = get_overlaps(
            cache, None, dataset_name, inference_config,
            iou_threshold=iou_threshold, sort_method=sort_method,
            verbose=verbose, gt2_name=gt2_name, filter=filter)
        #ious = get_overlaps(model=None, dataset=dataset_analyze, 
        #                    config=inference_config, iou_threshold=iou_threshold, gt_set2=dataset_analyze2, filter = filter)
    else:
        raise ValueError("Either a model or second dataset must be provided.")
    iou_df = pd.DataFrame(ious)
    #sort by image_id, gt_index, and iou(descending) to prioritize best IoU matches
    sorted_df = iou_df.sort_values(by=['image_id','iou'], ascending=[True, False])
    #print(sorted_df)
        


    #list to store filtered results
    unique_matches = []
    
    #iterate over each image id
    image_ids = sorted_df['image_id'].unique()
    if not verbose:
        image_ids = tqdm(image_ids, desc="Filtering matches", unit="img")
            
    for image_id in image_ids:
        if verbose:
            print(f"Processing image_id:{image_id}")
            
        #get rowsfor current image id
        image_df = sorted_df[sorted_df['image_id']==image_id]
    
        if verbose:
            print(f"Dataframe for image_id {image_id}")
            print(image_df.head(10))
    
        #track which GT and preds are already matched
        matched_gt=set()
        matched_preds=set() 
    
        match_counter = 0
        skip_counter = 0
    
        for _, row in image_df.iterrows():
            #print(f"checking row:{row.to_dict()}")
            #print(f" Pred: {row['pred_index']} and GT: {row['gt_index']} IoU: {row['iou']}")
            #if current GT box has not been matched and prediction has not been used
            if row['pred_index'] not in matched_preds and row['gt_index'] not in matched_gt:
                if verbose:
                    print(f"--> Match found: Prediction {row['pred_index']} with GT {row['gt_index']}  (IoU: {row['iou']:.4f})")
                match_counter +=1
                
                #add to list of unique matches
                unique_matches.append(row)
                #mark GT as matched
                matched_gt.add(row['gt_index'])
                #mark pred as matched
                matched_preds.add(row['pred_index'])
                
                
            elif row['pred_index'] in matched_preds:
                #print(f"--> skipped: Prediction {row['pred_index']} already matched")
                skip_counter +=1
            elif row['gt_index'] in matched_gt:
                #print(f"--> skipped: GT {row['gt_index']} already matched")
                skip_counter +=1
        if verbose:
            print(f"\n Image {image_id} processing complete")
            print(f"Matches:{match_counter}")
            print(f"skips: {skip_counter}")
        
            
    #convert list of unique matches to df
    filtered_df = pd.DataFrame(unique_matches)
    #sort by pred index
    filtered_df = filtered_df.sort_values(by=['image_id', 'pred_index'])

    if verbose:
        print("\nFinal Filtered DataFrame:")
        print(filtered_df)

    
    return filtered_df


 
def print_verbose(message, verbose=1, level=1):
    """Prints the message only if verbose_level is greater than 0."""
    if verbose >= level:
        print(message)
    
    
def load_model(model_path, model_dict, model_list, model_dir, 
               config, target_class=None, image_size=None, 
               custom_name=None, append_epoch=False):
    """ Helper function to load models based on paths given in input"""
    
    model = modellib.MaskRCNN(mode="inference", 
                          config=config,
                          model_dir=model_dir)
    
    print(f"Loading model weights from: {model_path}")
    
    model.load_weights(model_path, by_name=True)
    
    base_name = os.path.basename(os.path.dirname(model_path))
    model_name = base_name

    #determine model name

    #add custom descriptor
    if custom_name is not None:
        model_name = f"{model_name}_{custom_name}"

    #optionally append epoch
    if append_epoch:
        match = re.search(r'_(\d+)\.h5$', model_path)
        if match:
            epoch = match.group(1)
            model_name = f"{model_name}_E{epoch}"
        
        
    
    #model_name = os.path.basename(os.path.dirname(model_path))
    
    model_dict[model_name] = {
        "model": model,
        "confidence": config.DETECTION_MIN_CONFIDENCE, 
        "target_class": target_class if target_class is not None else None,
        "path": model_path,
        "config": config,
        "image_size": image_size if image_size is not None else 1024
        
        }
    model_list.append(model_name)
    print(f"Loaded model: {model_name}")
    
    return model
    
    
####Postprocessing additions

class MaskAnalyzer:
    """Class for analyzing masks and computing overlaps"""
    def __init__(self, cache: ResultsCache):
        self.cache = cache
        
    def get_masks(self, ref_name, key, image_id, inference_config, verbose=1):
        """
        Fetch masks for a dataset or model prediction:
        -ref_name: The dataset containing image to be analyzed
        -key: name of model or analysis dataset (such as CHT or EDMWS or GT)
        """
        # Always load the reference image:
        image, _,_,_,_ = self.cache.get_gt(ref_name, inference_config, image_id)

        print_verbose(f"[get masks] GT Image size: {image.shape}",verbose,2)

        dataset = self.cache.datasets[ref_name]
        image_info = dataset.image_info[image_id]

        #Default Scale from dataset
        scale = image_info.get('scale',1.0)
        print_verbose(f"[get masks] Dataset '{ref_name}' image '{image_id}' default scale: {scale}", verbose, level=2)
        #Load masks
        if key in self.cache.model_dict:
            #Model Predictions
            result= self.cache.get_model_pred(key, ref_name, inference_config, image_id, verbose=verbose)
            
            masks = result['masks']

            
            # override scale with effective scale 
            scale = result.get('effective_scale', scale)

            print_verbose(f"[get masks] Using model '{key}' effective scale: {scale}", verbose, level=2)

        elif key in self.cache.datasets:
            #Ground truth masks or other anaylsis dataset
            _,_,_,_,masks =  self.cache.get_gt(key, inference_config, image_id)
            scale = self.cache.gt_scale[ref_name][image_id]

            print_verbose(f"[get_masks] Using dataset '{key}' masks (from dataset '{ref_name}'), scale: {scale}", verbose, level=2)
            print_verbose(f"[get_masks] Masks shape: {masks.shape}", verbose, level=2)
            #use default dataset scale, not overriden effective scale
        else:
            raise ValueError(f"Key '{key}' not found in models or datasets.")
        

        print_verbose(f"[get_masks] Image shape: {image.shape}, Masks shape: {masks.shape if masks is not None else None}", verbose, level=3)
        return image, masks, scale
    #-----------------
    #Particle/mask analysis
    def compute_equi_diam_pix(self, masks):
        """Computes the equivalent diameter of particle mask
        masks: [Height, Width, instances] """
        
        if masks.shape[-1] ==0:
            return np.zeros((masks.shape[-1]))
        #flatten masks and compute area
        masks = np.reshape(masks > .5, (-1,masks.shape[-1])).astype(np.float32)
        area = np.sum(masks, axis=0)
    
        dp_pix = np.sqrt((4*area)/np.pi)
        
        return dp_pix, area

    def compute_feret(self, masks):
        """Computes the feret diameter, feret x, feret y, feret angle, and min feret"""
        #get number of masks
        n_instances = masks.shape[-1]
        if n_instances ==0:
            return (np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,)))
        
        #initialize lists to store results
        feret = np.zeros(n_instances)
        feret_x , feret_y= np.zeros(n_instances) , np.zeros(n_instances)
        feret_angle = np.zeros(n_instances)
        min_feret = np.zeros(n_instances)

        for i in range(n_instances):
            mask = (masks[...,i] > 0.5).astype(np.uint8) #ensure 8 bit binary mask
            #find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                print(f"No contours found for mask {i}")
                continue
            cnt = max(contours, key =cv2.contourArea)
            if cv2.contourArea(cnt) ==0:
                print(f"Warning: Largest contour has zero area for mask {i}")
            hull = cv2.convexHull(cnt) #compute convex hull (removes convex)

            #compute max feret (largest distance between any two points on contour)
            pts = hull[:,0,:] #extract points

            #compute pairwise distances
            diff = pts[:, np.newaxis,:] - pts[np.newaxis,:,:] #shape (N,N, 2)
            dists = np.sqrt((diff**2).sum(axis=2)) #euclidean distance

            #max distance is feret diameter
            feret[i] = np.max(dists)

            # compute feret angle (angle between two farthest points)
            ij = np.unravel_index(np.argmax(dists), dists.shape) #indices of max dist
            dx, dy = pts[ij[1]] - pts[ij[0]]
            feret_angle[i] = np.degrees(np.arctan2(dy, dx)) #angle in degrees

            #compute feret x and y (using bbox)
            x,y,w,h = cv2.boundingRect(cnt)
            feret_x[i] = w
            feret_y[i] = h

            # compute min feret (smallest area bounding rectangle)
            rect = cv2.minAreaRect(hull)
            (_,_), (width, height),_ = rect
            min_feret[i] = min(width, height)

            if (feret[i] == 0 or feret_x[i] == 0 or feret_y[i] == 0 or min_feret[i] == 0):
                print(f"Warning: Zero Feret found for mask {i} (feret={feret[i]}, feret_x={feret_x[i]}, "
                    f"feret_y={feret_y[i]}, min_feret={min_feret[i]})")



        return feret, feret_x, feret_y, feret_angle, min_feret

    def filter_mask_size(self, masks, min_dp_pix = 18):
        N = masks.shape[-1]
        if N == 0:
            return masks
        # Ensure binary
        masks = masks > 0.5
        #compute diameters and areas
        dp_pix, area = self.compute_equi_diam_pix(masks)
        print("Min dp: (pre-filter)", min(dp_pix))
        keep = np.ones(masks.shape[-1], dtype=bool)
        keep &= dp_pix >= min_dp_pix
        
        #filter
        filtered_masks = masks [:,:,keep ]
        dp_pix_filtered = dp_pix[keep]
        print("Min dp: (filtered):", min(dp_pix_filtered))
        return filtered_masks

    def compute_mask_centroids(self, masks, scale=None):
        N = masks.shape[-1]
        #print("N", N)
        centroids = []
        pixel_counts = []
        
        for i in range(N):
            mask = masks[:, :, i]  # Select the i-th mask

            # Get coordinates of all pixels where the mask is True (nonzero)
            y_coords, x_coords = np.where(mask)

            # Calculate average x and y coordinates (centroid)
            x_avg = np.mean(x_coords)
            y_avg = np.mean(y_coords)

            # Count the number of pixels in the mask
            pixel_count = len(x_coords)

        # Append results
            centroids.append((x_avg, y_avg))
            pixel_counts.append(pixel_count)
        centroids = np.array(centroids)
        pixel_counts = np.array(pixel_counts)

        return centroids, pixel_counts

    def calculate_radius_of_gyration(self, df_main):
        df = df_main.copy()
        
        #Use particle area as weight
        weights = df['PP area (nm^2)']
        
        #compute weights centroids of aggregate
        x_centroid = (df['x_avg'] * weights).sum() / weights.sum()
        y_centroid = (df['y_avg'] * weights).sum() / weights.sum()
        
        #Compute square distances of each particle from aggregate centroid
        df['distance_sq'] = (df['x_avg'] - x_centroid)**2 + (df['y_avg'] - y_centroid)**2
        
        #Caluclate weighted mean of squared distances
        weighted_mean_sq_distance = (weights * df['distance_sq']).sum() / weights.sum()

        #calculate aggregate radius of gyration
        Rg = np.sqrt(weighted_mean_sq_distance)*df['scale_length (nm)'].mean()
        
        return Rg

    def calc_cluster_Rg(self, mask, scale = 1.0):
        centroids, _ = self.compute_mask_centroids(mask[:,:,np.newaxis])

        x_avg, y_avg = centroids[0]

        pixel_indices = np.argwhere(mask)
        if pixel_indices.size == 0:
            return 0.0, 0.0
        
        y_coords, x_coords = pixel_indices[:,0], pixel_indices[:,1]

        dx = x_coords - x_avg
        dy = y_coords - y_avg
        rg_pix = np.sqrt(np.mean(dx**2+dy**2))
        rg_nm = rg_pix * scale
        return rg_pix, rg_nm

    def compute_fractal_dimension(self, masks, save_binary=True, save_path=None, show=False, plot=0,verbose=0):

        analyzer = SF.StereoFractAnalyzer()

        binary_mask = 255 - np.any(masks, axis=-1).astype(np.uint8)*255
        if save_binary:
            if save_path is None:
                raise ValueError("save path must be provided")
            plt.figure(figsize = (binary_mask.shape[1]/100, binary_mask.shape[0]/100),dpi=100)
            plt.imshow(binary_mask, cmap='gray')
            plt.axis('off')
            plt.tight_layout(pad=0.0)
            plt.savefig(save_path, format="png",bbox_inches='tight', pad_inches=0,orientation= 'landscape')
            plt.close()
            plt.tight_layout(pad=0.0)

            if show:
                plt.imshow(binary_mask, cmap='gray')
                plt.axis('off')
                plt.show()
                
            image_path = save_path
        
        else: 
            #save as temp image if not saving
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_filename = temp_file.name   
            Image.fromarray(binary_mask).save(temp_filename)
            image_path = temp_filename
    
        fractal_dimension = analyzer.get_image_fractal_dimension(image_path, plot=plot, verbose=verbose)
        
        if not save_binary:
            os. remove(temp_filename)
        
        return fractal_dimension

    def compute_pp_coverage_score(self, particle_masks, cluster_mask, verbose):
        print_verbose("Calculating Coverage Fraction...", verbose, level=2)

        if particle_masks is None or particle_masks.size == 0:
            return np.nan
        if cluster_mask is None or cluster_mask.size == 0:
            return np.nan

        

        # Union of all primary particles
        pp_union = np.any(particle_masks > 0, axis=-1)  # HxW boolean

        # Ensure cluster mask is binary and 2D
        cluster_bin = np.squeeze(cluster_mask > 0)

        # Resize if needed
        if pp_union.shape != cluster_bin.shape:
            # you can resize cluster_bin or pp_union to match the other
            raise ValueError(f"Shape mismatch: pp_union {pp_union.shape} vs cluster_mask {cluster_bin.shape}")

        # Compute coverage
        covered_pixels = np.logical_and(pp_union, cluster_bin).sum()
        total_cluster_pixels = cluster_bin.sum()

        coverage_fraction = covered_pixels / (total_cluster_pixels + 1e-6)
        return coverage_fraction        
            

    def compute_pp_leakage_score(self, particle_masks, cluster_mask, verbose):
        print_verbose("Calculating Leakage Fraction...", verbose, level=2)

        if particle_masks is None or particle_masks.size ==0:
            return np.nan

        if cluster_mask is None or cluster_mask.size==0:
            return np.nan

        pp_union = np.any(particle_masks >0, axis=-1) #HxW boolean

        cluster_bin = np.squeeze(cluster_mask > 0)

        if pp_union.shape != cluster_bin.shape:
            raise ValueError(f"Shape mismatch: pp_union {pp_union.shape} vs cluster_mask {cluster_bin.shape}")

        #compute leakage
        leak_pixels = np.logical_and(pp_union, np.logical_not(cluster_bin)).sum()

        total_pred_pixels = pp_union.sum()

        leakage_fraction = leak_pixels / (total_pred_pixels + 1e-6)

        return leakage_fraction


    def analyze_image(self, ref_name, key, image_id, inference_config=None, 
                    metrics=None,save_binary=False, show=False, 
                      verbose=1, mask_filter=None, cluster_mode=False):
        
        if metrics is None:
            metrics = ['dp', 'feret', 'centroids', 'fractal', 'Rg', 'coverage']
     
        results = {"particle": {}, "aggregate": {},"meta": {}}


        #Load image and masks

        image, masks, scale = self.get_masks(ref_name, key, image_id, inference_config,verbose=verbose)

        
        # Verbose tracing
        print_verbose(f"[analyze_image] Processing image {image_id} in {('cluster' if cluster_mode else 'particle')} mode", verbose, level=2)
        print_verbose(f"[analyze_image] Image shape: {image.shape}, Masks shape: {masks.shape if masks is not None else None}", verbose, level=2)
        print_verbose(f"[analyze_image] Using scale: {scale} nm/pix", verbose, level=2)

        # Apply mask filtering if specified
        if mask_filter is not None and masks is not None and masks.size >0:
            masks=  self.filter_mask_size(masks, **mask_filter)
            print_verbose(f"[analyze_image] After filtering, masks shape: {masks.shape}", verbose, level=3)

        #store meta info
        original_filename = self.cache.datasets[ref_name].image_info[image_id]['basename']
        base_filename = os.path.splitext(os.path.basename(original_filename))[0]

        mode_str = "Cluster/Aggregate" if cluster_mode else "Particle"
    
        print_verbose(
            f"↳Processing Image {image_id} ({base_filename}) in {mode_str} Mode: "
            f"{masks.shape[-1] if masks is not None else 0} masks found",
            verbose,
            level=1
        )
        results["meta"] = {
            "image_id": image_id,
            "original_filename": original_filename,
            "base_filename": base_filename,
            "scale_length (nm)": scale,
            "scale_area (nm^2)": scale**2 if scale is not None else None
        }
        #store masks in results as well.
        results["masks"] = masks

        N = masks.shape[-1] if masks is not None else 0
        results["aggregate"]["num_particles" if not cluster_mode else "num_aggregates"] = N
        print_verbose(f"[analyze_image] Number of {'particles' if not cluster_mode else 'aggregates'}: {N}", verbose, level=2)
       
        #---------------------------------------------------
        # Particle-level metrics (skip if in cluster mode)
        #---------------------------------------------------

        if not cluster_mode and N>0:
            #--- Equivalent Diameter
            
            if "dp" in metrics:
                dp_pix, area_pix = self.compute_equi_diam_pix(masks)
                dp_nm = dp_pix * scale if scale else dp_pix
                area_nm2 = 0.25 * np.pi * dp_nm**2 if scale else area_pix

                results["particle"]["dp (pix)"] = dp_pix
                results["particle"]["dp (nm)"] = dp_nm
                results["particle"]["PP area (pix)"] = area_pix
                results["particle"]["PP area (nm^2)"] = area_nm2


            #--- Feret Properties
            if "feret" in metrics:
                feret, feret_x, feret_y, feret_angle, min_feret = self.compute_feret(masks)
                results["particle"].update({
                    "feret (pix)": feret,
                    "feret (nm)": feret * scale if scale else feret,
                    "feret_x (pix)": feret_x,
                    "feret_x (nm)": feret_x * scale if scale else feret_x,
                    "feret_y (pix)": feret_y,
                    "feret_y (nm)": feret_y * scale if scale else feret_y,
                    "feret_angle (degrees)": feret_angle,
                    "min_feret (pix)": min_feret,
                    "min_feret (nm)": min_feret * scale if scale else min_feret
                })

            #--- Centroids
            if "centroids" in metrics:
                centroids, pixel_counts = self.compute_mask_centroids(masks, scale=scale)
                results["particle"]["centroids"] = centroids
                results["particle"]["num_pixels"] = pixel_counts
        
        # ------ Aggregate Level metrics
        if N>0:
            if "coverage" in metrics or "leakage" in metrics:

                #load cluster (from dataset cache for now
                dataset = self.cache.datasets[ref_name]
                gt_clusters= self.cache.get_gt_cluster_masks(ref_name, image_id)
            
                if gt_clusters is None:
                    print_verbose(f"gt_clusters: None", verbose, level=3)
                else:
                    print_verbose(f"gt_clusters.shape: {gt_clusters.shape}", verbose, level=3)
                if gt_clusters is not None and gt_clusters.size > 0:

                    
                    og_img = dataset.load_image(image_id)
                    #compute scale, window, padding, crop
                    _, window, cluster_scale, padding, crop = resize_image(
                        og_img,
                        min_dim=inference_config.IMAGE_MIN_DIM,
                        min_scale = inference_config.IMAGE_MIN_SCALE,
                        max_dim = inference_config.IMAGE_MAX_DIM,
                        mode=inference_config.IMAGE_RESIZE_MODE
                    )            

                    gt_clusters = resize_mask(gt_clusters, cluster_scale, padding, crop)
                    
                #print(f"Reshaped: {gt_clusters.shape}")

                #coverage
                if "coverage" in metrics:
                    if gt_clusters is not None:
                        coverage = self.compute_pp_coverage_score(particle_masks=masks,cluster_mask=gt_clusters, verbose=verbose)  
                    else:
                        coverage=np.nan
                    
                    print_verbose(f"Coverage score for image {image_id}: {coverage}", verbose, level=2)
                    results["aggregate"]["coverage_score"] = coverage
                if "leakage" in metrics:
                    if gt_clusters is not None:
                        leakage = self.compute_pp_leakage_score(particle_masks=masks, cluster_mask = gt_clusters, verbose = verbose)
                    else:
                        leakage=np.nan

                    print_verbose(f"Leakage fraction for image {image_id}: {leakage}", verbose, level=2)
                    results["aggregate"]["leakage_frac"] = leakage


            #compute centroids and areas for clusters
            if "centroids" in metrics:
                centroids, pixel_counts = self.compute_mask_centroids(masks, scale=scale)
                areas_nm2 = [pix * scale**2 if scale else pix for pix in pixel_counts]
                results["aggregate"]["centroids"] = centroids
                results["aggregate"]["num_pixels"] = pixel_counts
                results["aggregate"]["area (nm^2)"] = areas_nm2
            #Fractal Dimension
            if "fractal" in metrics:
                if cluster_mode:
                    #compute per cluster rather than union of particle masks
                    fractal_list = []
                    for i in range(N):
                        mask = masks[:,:,i]
                        fractal_dim = self.compute_fractal_dimension(
                            mask[:,:,np.newaxis], 
                            save_binary=save_binary, 
                            save_path=None,
                            show=show, 
                            plot=0,
                            verbose=verbose
                        )
                        fractal_list.append(fractal_dim)
                    results["aggregate"]["fractal_dimension"] = fractal_list
                else:
                    fractal_dim = self.compute_fractal_dimension(
                        masks, 
                        save_binary=save_binary, 
                        save_path=None,
                        show=show, 
                        plot=0,
                        verbose=verbose
                    )
                    results["aggregate"]["fractal_dimension"] = fractal_dim
            #Radius of Gyration
            if "Rg" in metrics:
                if cluster_mode:
                    #Cluster level Rg calculation (using cluster mask rather than particle masks)
                    Rg_list = []
                    for i in range(N):
                        mask = masks[:,:,i]
                        Rg_pix, Rg_nm = self.calc_cluster_Rg(mask, scale=scale)
                        Rg_list.append(Rg_nm)
                    results["aggregate"]["radius_of_gyration (nm)"] = Rg_list
                
                else:
                    #Particle level Rg calculartion
                    if "centroids" not in results["particle"] or results["particle"]["centroids"] is None:
                        centroids, pixel_counts = self.compute_mask_centroids(masks, scale=scale)
                        results["particle"]["centroids"] = centroids
                        results["particle"]["num_pixels"] = pixel_counts

                    # Ensure areas are available
                    if "PP area (pix)" not in results["particle"]:
                        dp_pix, area_pix = self.compute_equi_diam_pix(masks)
                        results["particle"]["PP area (pix)"] = area_pix

                    #build temporary df
                    df_temp = pd.DataFrame({
                        "x_avg": [c[0] for c in results["particle"]["centroids"]],
                        "y_avg": [c[1] for c in results["particle"]["centroids"]],
                        "PP area (nm^2)": results["particle"]["PP area (pix)"] * scale**2,
                        "scale_length (nm)": scale
                    })

                    Rg = self.calculate_radius_of_gyration(df_temp)
                    results["aggregate"]["radius_of_gyration (nm)"] = Rg

        
        return results

    def process_particles(self, dataset_name, key=None, inference_config = None, 
                          save_binary=False, show_binary_union=False, plot_df=0, 
                          mask_filter = None, metrics=None,
                          verbose=0):


        #Default: compute all metrics
        if metrics is None: 
            metrics = ['dp', 'feret', 'centroids', 'fractal', 'Rg','coverage', 'leakage']

        #load dataset (need to use images even if using model)
        dataset = self.cache.datasets.get(dataset_name)

        if dataset is None: 
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        particle_records = []
        aggregate_records = []
        fractal_dims = []
        Rgs = []

        dp_pix_all, dp_nm_all, feret_pix_all, feret_nm_all, min_feret_pix_all, min_feret_nm_all = [],[],[], [], [], []  

        for image_id in dataset.image_ids:
            image_info = dataset.image_info[image_id]
            original_filename = image_info['basename']
            base_filename = os.path.splitext(os.path.basename(original_filename))[0] 
            #scale = image_info.get('scale', 1.0) #fallback to 1.0 if missing
            # scale = image_scales[base_filename] if isinstance(image_scales, dict) else image_scales

            

            #Call analyze image function
            results = self.analyze_image(
                ref_name = dataset_name, 
                key = key,
                image_id = image_id, 
                inference_config = inference_config,
                #scale = scale,
                metrics = metrics,
                save_binary=save_binary, 
                show = show_binary_union,
                verbose=verbose,
                mask_filter=mask_filter
                
            )

            scale = results['meta']['scale_length (nm)']
            scale_area = scale**2

            N_particles = results["aggregate"].get("num_particles", 0)
            if N_particles == 0:
                continue


            #Extract particle-level data
            centroids = results["particle"].get("centroids", [None]*N_particles)
            pixel_counts = results["particle"].get("num_pixels", [None]*N_particles)
            dp_pix = results["particle"].get("dp (pix)", [None]*N_particles)
            dp_nm = results["particle"].get("dp (nm)", [None]*N_particles)
            pp_area_pix =results["particle"].get("PP area (pix)", [None]*N_particles)

            feret_pix = results["particle"].get("feret (pix)", [None]*N_particles)
            feret_nm = results["particle"].get("feret (nm)", [None]*N_particles)
            feret_x_pix = results["particle"].get("feret_x (pix)", [None]*N_particles)
            feret_x_nm = results["particle"].get("feret_x (nm)", [None]*N_particles)
            feret_y_pix = results["particle"].get("feret_y (pix)", [None]*N_particles)
            feret_y_nm = results["particle"].get("feret_y (nm)", [None]*N_particles)
            feret_angle = results["particle"].get("feret_angle (degrees)", [None]*N_particles)
            min_feret_pix = results["particle"].get("min_feret (pix)", [None]*N_particles)
            min_feret_nm = results["particle"].get("min_feret (nm)", [None]*N_particles)

            Rg_nm = results["aggregate"].get("radius_of_gyration (nm)")
            fractal_dim = results["aggregate"].get("fractal_dimension")
            coverage_score = results["aggregate"].get("coverage_score")
            leakage_frac = results["aggregate"].get("leakage_frac")

            fractal_dims.append({'image': base_filename, 'fractal dimension': fractal_dim})
            Rgs.append({'image': base_filename, 'Rg (nm)': Rg_nm})

            #Build particle level records
            for i in range(N_particles):
                x_avg, y_avg = centroids[i] if centroids[i] is not None else (None,None)
                record = {
                    "image": base_filename,
                    "scale_length (nm)": scale,
                    "scale_area (nm^2)": scale_area,
                    "PP #": i+1
                    }
                if "centroids" in metrics:
                    #x_avg, y_avg = centroids[i] if (centroids is not None and len(centroids) > 0) else (None, None)
                    record.update({
                        "x_avg": x_avg,
                        "y_avg": y_avg,
                        "num_pixels": pixel_counts[i] 
                    })
            
                if "dp" in metrics:
                    record.update({
                        "dp (pix)": dp_pix[i] ,
                        "dp (nm)" : dp_nm[i] ,
                        "PP area (pix)":pp_area_pix[i],
                        "PP area (nm^2)": 0.25 * np.pi * dp_nm[i]**2 if dp_nm is not None else None
                    })
                if "feret" in metrics:
                    record.update({
                        "feret (pix)": feret_pix[i], 
                        "feret (nm)": feret_nm[i], 
                        "feret_x (pix)": feret_x_pix[i],
                        "feret_x (nm)": feret_x_nm[i],
                        "feret_y (pix)": feret_y_pix[i], 
                        "feret_y (nm)": feret_y_nm[i],
                        "feret_angle (degrees)": feret_angle[i],
                        "min_feret (pix)": min_feret_pix[i],
                        "min_feret (nm)": min_feret_nm[i],
                    })

                particle_records.append(record)


            #Store aggregate level records
            agg_record = {
                "image": base_filename,
                "num_particles": N_particles,
                "scale_length (nm)": scale,
                "scale_area (nm^2)": scale_area
                }

            if 'fractal' in metrics:
                agg_record["fractal_dimension"] = fractal_dim
            if 'Rg' in metrics:
                agg_record["Rg (nm)"] = Rg_nm
            if 'coverage' in metrics:
                agg_record["coverage_score"] =coverage_score
            if 'leakage' in metrics:
                agg_record["leakage_frac"] = leakage_frac
            aggregate_records.append(agg_record)

        df_particles = pd.DataFrame(particle_records)
        df_aggregate = pd.DataFrame(aggregate_records)
        

        return df_particles, df_aggregate
        
            

    def process_clusters(self, dataset_name, key=None, inference_config=None, save_binary=False,
                         show_binary_union=False, plot_df=0, metrics=None, verbose = 0):
        

        #Default metrics for cluster processing:
        if metrics is None:
            metrics = ['centroids','fractal', 'Rg']

        cluster_records = []
        fractal_dims = []
        Rg_records = []

        dataset=  self.cache.datasets.get(dataset_name)
        if dataset is None: 
            raise ValueError(f"Dataset '{dataset_name}' not found.")
        
        for image_id in dataset.image_ids:
            image_info = dataset.image_info[image_id]
            original_filename = image_info['basename']
            base_filename = os.path.splitext(os.path.basename(original_filename))[0] 
            #scale = image_info.get('scale',1.0)
            # scale = image_scales[base_filename] if isinstance(image_scales, dict) else image_scales

            #call analyze_image function in cluster mode
            results = self.analyze_image(
                ref_name = dataset_name, 
                key = key,
                image_id = image_id,
                inference_config = inference_config,
                #scale=scale,
                metrics= metrics, 
                save_binary= save_binary,
                show= show_binary_union,
                verbose= verbose,
                cluster_mode= True
            )

            scale = results['meta']['scale_length (nm)']
            scale_area = scale**2
            N_clusters = results["aggregate"].get("num_aggregates", 0)

            # fractal_list = results["aggregate"].get("fractal_dimension", [])
            # if N_clusters == 1 and not isinstance(fractal_list, list):
            #     fractal_list = [fractal_list]
            # for fd in fractal_list:
            #     fractal_dims.append({'image': base_filename, 'fractal_dimension': fd})
            fractal_list = results["aggregate"].get("fractal_dimension", [])
            if not isinstance(fractal_list, list):
                fractal_list = [fractal_list]

            centroids_list = results["aggregate"].get("centroids", [])
            pixel_counts = results["aggregate"].get("num_pixels", [])
            areas_nm2 = results["aggregate"].get("area (nm^2)", [])
            Rg_list = results["aggregate"].get("radius_of_gyration (nm)", [])

            for i in range(N_clusters):
                cent = centroids_list[i] if i < len(centroids_list) else (None, None)
                x_avg, y_avg = cent
                area_pix = pixel_counts[i] if i < len(pixel_counts) else None
                area_nm2= areas_nm2[i] if i < len(areas_nm2) else None

                #radius of gyration
                
                Rg_nm = Rg_list[i] if i < len(Rg_list) else None
                Rg_pix= Rg_nm / scale if Rg_nm is not None and scale is not None else None
                fd = fractal_list[i] if i < len(fractal_list) else None

                cluster_records.append({
                    "image": base_filename,
                    "cluster #": i+1,
                    "scale_length (nm)": scale,
                    "scale_area (nm^2)": scale_area,
                    "Cluster area (pix)": area_pix,
                    "Cluster area (nm^2)": area_nm2,
                    "Fractal dimension": fd,
                    "Rg (pix)": Rg_pix,
                    "Rg (nm)": Rg_nm,
                    "x_avg": x_avg,
                    "y_avg": y_avg  
                })

                Rg_records.append({'image': base_filename,
                                    'cluster #': i+1, 
                                    'Rg (nm)': Rg_nm,
                                    'Rg (pix)': Rg_pix})

                fractal_dims.append({'image': base_filename, 'fractal_dimension': fd})
        #convert to dfs
        df_clusters = pd.DataFrame(cluster_records)
        df_fractal = pd.DataFrame(fractal_dims)
        df_Rg = pd.DataFrame(Rg_records)

        return df_clusters, df_fractal, df_Rg

    def gather_aggregate_morphology(self, ref_set, key, settings=None, 
                                save_binary=False, show_binary=False, plot=False):
        """wrapper function to gather and save aggregate morphology information"""
        
        # --- Defaults
        if settings is None:
            settings = {}
        save_results = settings.get('save_results', True)
        verbose = settings.get('verbose', 1)
        inference_config = settings.get('inference_config', None)

        datasets = self.cache.datasets
        model_dict = self.cache.model_dict
        Results_DIR = self.cache.results_dir
        
        # --- Load reference dataset
        ref_dataset = datasets.get(ref_set)
        if ref_dataset is None:
            raise ValueError(f"Reference dataset '{ref_set}' not available.")
        
        # --- Determine Source type (model or dataset)

        if key in model_dict:
            mode = "model"
            model_entry = model_dict[key]
            model = model_entry['model']
            confidence = model_entry.get('confidence', float('nan'))
            target_class = model_entry.get('target_class', 'particle')
            method = key

            pp_save_name = os.path.join(Results_DIR,ref_set, 'PP_Info',f"{method}_{confidence}_pp_info.csv")
            #add save path

        elif key in datasets:
            mode = "dataset"
            pred_dataset = datasets[key]
            model=None
        
            #infer target class from dataset lables
            confidence = float('nan')
            labels = pred_dataset.get_class_labels(range(len(pred_dataset.class_names)))
            target_class = labels[0] if labels else 'particle'
            method = key
            pp_save_name = os.path.join(Results_DIR,ref_set, 'PP_Info', f"{key}_pp_info.csv")

        else: 
            raise ValueError(f"Key '{key}' not found in models or datasets.")
        
        # --- Verbose information
        print_verbose(f"\n[Aggregate Morphology]", verbose, level=1)
        print_verbose(f"Reference Dataset: {ref_set}", verbose, level=1)
        print_verbose(f"Prediction Source: {key} ({mode})", verbose, level=1)
        print_verbose(f"Target Class: {target_class}", verbose, level=1)
        


        #----Ensure output directories exist ----
        save_dir = os.path.join(Results_DIR, ref_set)
        os.makedirs(save_dir, exist_ok=True)

        
        # -----Particle based model path:
        if target_class == 'particle':
            pp_output_dir = os.path.dirname(pp_save_name)
            os.makedirs(pp_output_dir, exist_ok=True)


        #Process individual particles (give df with dp info)
      
            df_particles, df_aggregate = self.process_particles(
                dataset_name=ref_set,
                key = key,
                inference_config = inference_config,
                save_binary= save_binary,
                show_binary_union= show_binary,
                plot_df= plot,
                mask_filter= None,
                metrics = None,
                verbose = verbose)

                                                                
        
                #print_verbose(f" Fractal Dimension df {fractal_dims}", verbose)
                #print_verbose(f"", verbose)
                #print_verbose(f"{df_particles.head(5)}", verbose)
        
            # --- Compute aggregate statistics per image

            #improved for scalability using groupby for aggregating stats:

            particles_by_image = df_particles.groupby('image')
            image_ids = list(particles_by_image.groups.keys())

            # tqdm iterator
            if verbose < 2:
                image_iter = tqdm(image_ids, desc=f"Processing images ({method})", leave=True, unit='image')
            else:
                image_iter = image_ids



            aggregates_list = []

            for image_name in image_iter:
                image_particles = particles_by_image.get_group(image_name)
                scale_length = image_particles['scale_length (nm)'].mean()
                N_pp = len(image_particles)

                mean_dp_pix = image_particles['dp (pix)'].mean()
                SEM_dp_pix = image_particles['dp (pix)'].sem()
                STD_dp_pix = image_particles['dp (pix)'].std()

                mean_dp_nm = image_particles['dp (nm)'].mean()
                SEM_dp_nm = image_particles['dp (nm)'].sem()
                STD_dp_nm = image_particles['dp (nm)'].std()

                #Get Rg and fractal dim directly from df_aggregate
                agg_row = df_aggregate.loc[df_aggregate['image'] == image_name]
                Rg_nm = agg_row['Rg (nm)'].values[0]
                Rg_pix = Rg_nm / scale_length if scale_length else None

                fractal_dim = agg_row['fractal_dimension'].values[0]

                coverage_score = agg_row.get('coverage_score', [float('nan')]).values[0]
                leakage_frac = agg_row.get('leakage_frac', [float('nan')]).values[0]

                aggregates_list.append({
                    "image": image_name,
                    "method": method,
                    "conf_threshold": confidence,
                    "target_class": target_class,
                    "length scale [nm/pix]": scale_length,
                    "# of PP": N_pp,
                    "Mean dp [pix]": mean_dp_pix,
                    "mdp SEM [pix]": SEM_dp_pix,
                    "mdp STD [pix]": STD_dp_pix,
                    "Mean dp [nm]": mean_dp_nm,
                    "mdp SEM [nm]": SEM_dp_nm,
                    "mdp STD [nm]": STD_dp_nm,
                    "Rg [pix]": Rg_pix,
                    "Rg [nm]": Rg_nm,
                    "fractal_dim": fractal_dim,
                    "coverage_score": coverage_score,
                    "leakage_frac": leakage_frac
                })

                 # --- Verbose per-image info
                # if verbose >=2 :
                #     print_verbose(f"Processed image '{image_name}': #PP={N_pp}, Mean dp={mean_dp_nm:.2f} nm, Rg={Rg_nm:.2f} nm, Fractal Dim={fractal_dim:.3f}", verbose, level=2)
                if verbose >= 2:
                    tqdm.write(f"Processed image '{image_name}': #PP={N_pp}, Mean dp={mean_dp_nm:.2f} nm, Rg={Rg_nm:.2f} nm, Fractal Dim={fractal_dim:.3f}")

            aggregates = pd.DataFrame(aggregates_list)



        #------------------------
        #Cluster-based-model path
        #-------------------------
        elif target_class == 'cluster':
            #here, just extract cluster maskls and compute aggregate level Rg and dF
            print(f" Running in cluster mode for model: {key}")

            df_clusters,_,_  = self.process_clusters(dataset_name=ref_set,
                                                    key=key,
                                                    inference_config=inference_config,
                                              
                                                    save_binary=save_binary,
                                                    show_binary_union=show_binary,
                                                    plot_df=plot,
                                                    metrics=None,
                                                    verbose=verbose)
            

            cluster_ids = df_clusters['image'].unique()
            if verbose < 2:
                cluster_iter = tqdm(cluster_ids, desc=f"Processing clusters ({method})", leave=True, unit='cluster')
            else:
                cluster_iter = cluster_ids
            aggregates_records = []
            
            for cluster_id in cluster_iter:
                row = df_clusters.loc[df_clusters['image'] == cluster_id].iloc[0]
                fd = row['fractal_dimension']

            # for idx, row in df_clusters.iterrows():
            #     fd = df_clusters.loc[df_clusters['image'] == row['image'], 'fractal_dimension'].values[0]

                aggregates_records.append({
                    "image": row['image'],
                    "method": method,
                    "target_class": target_class,
                    "length scale [nm/pix]": row['scale_length (nm)'],
                    "Rg [pix]": row['Rg (pix)'],
                    "Rg [nm]": row["Rg (nm)"],
                    "fractal_dim": fd

                })

                # --- Verbose per-cluster info
                #print_verbose(f"Processed cluster '{row['image']}': Rg={row['Rg (nm)']:.2f} nm, Fractal Dim={fd:.3f}", verbose, level=2)
                if verbose >= 2:
                    tqdm.write(f"Processed cluster '{row['image']}': Rg={row['Rg (nm)']:.2f} nm, Fractal Dim={fd:.3f}")
            aggregates = pd.DataFrame(aggregates_records)
            df_particles = pd.DataFrame()
            df_clusters_return = df_clusters
            clust_output_dir = os.path.join(Results_DIR, ref_set, 'Cluster_Info')
            os.makedirs(clust_output_dir, exist_ok=True)
            clust_save_name = os.path.join(clust_output_dir, f"{method}_cluster_info.csv")

        else:
            raise ValueError(f"Unknown target class '{target_class}' for key '{key}'.")
        
        
       
        
      

        # -- Save outputs
        if save_results:
            if target_class == 'particle' and not df_particles.empty:
                df_particles.to_csv(pp_save_name, index=False)
                print(f"Saved primary particle info to {pp_save_name}")
            elif target_class =='cluster' and not df_clusters.empty:
                df_clusters.to_csv(clust_save_name, index=False)
                print(f"Saved cluster info to {clust_save_name}")

            aggregates_save_name = os.path.join(save_dir, "Aggregate_info.csv") if target_class == 'particle' else os.path.join(save_dir, "Aggregate_info_clusters.csv")

            if os.path.exists(aggregates_save_name):
                aggregates.to_csv(aggregates_save_name, mode='a', header=False, index=False)
                print(f"Appended aggregate summary to {aggregates_save_name}")
            else:
                aggregates.to_csv(aggregates_save_name, index=False)
                print(f"Created new aggregate summary for {ref_set}: {aggregates_save_name}")

        # --- Return dataframes


        if target_class == 'cluster':
            return aggregates, df_clusters_return                      
        else:
            return aggregates, df_particles

        
    

        

            


    

def compute_equi_diam_pix(masks):
    """Computes the equivalent diameter of particle mask
    masks: [Height, Width, instances] """
    
    if masks.shape[-1] ==0:
        return np.zeros((masks.shape[-1]))
    #flatten masks and compute area
    masks = np.reshape(masks > .5, (-1,masks.shape[-1])).astype(np.float32)
    area = np.sum(masks, axis=0)
   
    dp_pix = np.sqrt((4*area)/np.pi)
      
    return dp_pix, area


def compute_feret(masks):
    """Computes the feret diameter, feret x, feret y, feret angle, and min feret"""
    #get number of masks
    n_instances = masks.shape[-1]
    if n_instances ==0:
        return (np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,)))
    
    #initialize lists to store results
    feret = np.zeros(n_instances)
    feret_x , feret_y= np.zeros(n_instances) , np.zeros(n_instances)
    feret_angle = np.zeros(n_instances)
    min_feret = np.zeros(n_instances)

    for i in range(n_instances):
        mask = (masks[...,i] > 0.5).astype(np.uint8) #ensure 8 bit binary mask
        #find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            print(f"No contours found for mask {i}")
            continue
        cnt = max(contours, key =cv2.contourArea)
        if cv2.contourArea(cnt) ==0:
            print(f"Warning: Largest contour has zero area for mask {i}")
        hull = cv2.convexHull(cnt) #compute convex hull (removes convex)

        #compute max feret (largest distance between any two points on contour)
        pts = hull[:,0,:] #extract points

        #compute pairwise distances
        diff = pts[:, np.newaxis,:] - pts[np.newaxis,:,:] #shape (N,N, 2)
        dists = np.sqrt((diff**2).sum(axis=2)) #euclidean distance

        #max distance is feret diameter
        feret[i] = np.max(dists)

        # compute feret angle (angle between two farthest points)
        ij = np.unravel_index(np.argmax(dists), dists.shape) #indices of max dist
        dx, dy = pts[ij[1]] - pts[ij[0]]
        feret_angle[i] = np.degrees(np.arctan2(dy, dx)) #angle in degrees

        #compute feret x and y (using bbox)
        x,y,w,h = cv2.boundingRect(cnt)
        feret_x[i] = w
        feret_y[i] = h

        # compute min feret (smallest area bounding rectangle)
        rect = cv2.minAreaRect(hull)
        (_,_), (width, height),_ = rect
        min_feret[i] = min(width, height)

        if (feret[i] == 0 or feret_x[i] == 0 or feret_y[i] == 0 or min_feret[i] == 0):
            print(f"Warning: Zero Feret found for mask {i} (feret={feret[i]}, feret_x={feret_x[i]}, "
                  f"feret_y={feret_y[i]}, min_feret={min_feret[i]})")



    return feret, feret_x, feret_y, feret_angle, min_feret



def filter_mask_size(masks, min_dp_pix = 18):
    N = masks.shape[-1]
    if N == 0:
        return masks
    # Ensure binary
    masks = masks > 0.5
    #compute diameters and areas
    dp_pix, area = compute_equi_diam_pix(masks)
    print("Min dp: (pre-filter)", min(dp_pix))
    keep = np.ones(masks.shape[-1], dtype=bool)
    keep &= dp_pix >= min_dp_pix
    
    #filter
    filtered_masks = masks [:,:,keep ]
    dp_pix_filtered = dp_pix[keep]
    print("Min dp: (filtered):", min(dp_pix_filtered))
    return filtered_masks
    

def compute_mask_centroids(masks, scale=None):
    N = masks.shape[-1]
    #print("N", N)
    centroids = []
    pixel_counts = []
    
    for i in range(N):
        mask = masks[:, :, i]  # Select the i-th mask

        # Get coordinates of all pixels where the mask is True (nonzero)
        y_coords, x_coords = np.where(mask)

        # Calculate average x and y coordinates (centroid)
        x_avg = np.mean(x_coords)
        y_avg = np.mean(y_coords)

        # Count the number of pixels in the mask
        pixel_count = len(x_coords)

    # Append results
        centroids.append((x_avg, y_avg))
        pixel_counts.append(pixel_count)
    centroids = np.array(centroids)
    pixel_counts = np.array(pixel_counts)

    return centroids, pixel_counts

def calculate_radius_of_gyration(df_main):
    df = df_main.copy()
    
    #Use particle area as weight
    weights = df['PP area (nm^2)']
    
    #compute weights centroids of aggregate
    x_centroid = (df['x_avg'] * weights).sum() / weights.sum()
    y_centroid = (df['y_avg'] * weights).sum() / weights.sum()
    
    #Compute square distances of each particle from aggregate centroid
    df['distance_sq'] = (df['x_avg'] - x_centroid)**2 + (df['y_avg'] - y_centroid)**2
    
    #Caluclate weighted mean of squared distances
    weighted_mean_sq_distance = (weights * df['distance_sq']).sum() / weights.sum()

    #calculate aggregate radius of gyration
    Rg = np.sqrt(weighted_mean_sq_distance)*df['scale_length (nm)'].mean()
    
    return Rg

def calc_cluster_Rg(mask, scale = 1.0):
    centroids, _ = compute_mask_centroids(mask[:,:,np.newaxis])

    x_avg, y_avg = centroids[0]

    pixel_indices = np.argwhere(mask)
    if pixel_indices.size == 0:
        return 0.0, 0.0
    
    y_coords, x_coords = pixel_indices[:,0], pixel_indices[:,1]

    dx = x_coords - x_avg
    dy = y_coords - y_avg
    rg_pix = np.sqrt(np.mean(dx**2+dy**2))
    rg_nm = rg_pix * scale
    return rg_pix, rg_nm
    


def compute_fractal_dimension(masks, save_binary=True, save_path=None, show=False, plot=0,):

    analyzer = SF.StereoFractAnalyzer()

    binary_mask = 255 - np.any(masks, axis=-1).astype(np.uint8)*255
    if save_binary:
        if save_path is None:
            raise ValueError("save path must be provided")
        plt.figure(figsize = (binary_mask.shape[1]/100, binary_mask.shape[0]/100),dpi=100)
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0.0)
        plt.savefig(save_path, format="png",bbox_inches='tight', pad_inches=0,orientation= 'landscape')
        plt.close()
        plt.tight_layout(pad=0.0)

        if show:
            plt.imshow(binary_mask, cmap='gray')
            plt.axis('off')
            plt.show()
            
        image_path = save_path
    
    else: 
        #save as temp image if not saving
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_filename = temp_file.name   
        Image.fromarray(binary_mask).save(temp_filename)
        image_path = temp_filename
 
    fractal_dimension = analyzer.get_image_fractal_dimension(image_path, plot=plot)
    
    if not save_binary:
        os. remove(temp_filename)
    
    return fractal_dimension


def process_particles(dataset_name, datasets, model_dict, Results_DIR, model_name=None, 
                      image_scales=None, verbose = 0, save_binary=False, show_binary_union=False, plot_df=0):
    dp_pix_all = []
    dp_nm_all = []
    records = []
    fractal_dims = []
    feret_pix_all = []
    feret_nm_all = []
    
    #load dataset (need to use images even if using model)
    dataset = datasets.get(dataset_name, None)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found.")
    
    #run for each image in dataset
    image_ids = dataset.image_ids
    #progress_bar = tqdm(image_ids, desc=f"Processing '{dataset_name}' images", dynamic_ncols=True)
    #progress bar is WIP
    
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        original_filename = dataset.image_info[image_id]['basename']
        base_filename = os.path.splitext(os.path.basename(original_filename))[0] 
        
        
        print(f"↳Processing Image {image_id} ({base_filename})")
        

        #get image scales from either dict or float
        if isinstance(image_scales,dict):
            scale = image_scales.get(base_filename)
            if scale is None:
                print(f"No Scale found for {base_filename}, skipping")
        else:
            scale = image_scales
        
       
        
        if model_name:
            model = model_dict[model_name]['model'] #load model
            confidence = model_dict[model_name]['confidence'] #get conf threshold
            print(f"Using Model {model_name} (conf_thresh = {confidence})")
            binary_dir = os.path.join(Results_DIR, dataset_name,'Visualizations', f"{model_name}_{confidence}", 'Binary_Unions' )
            results = model.detect([image], verbose=0)
            r = results[0]
            masks = r['masks']
                
        else:
            print(f"Analyzing Dataset Masks: {dataset_name}")
            model=None
            binary_dir = os.path.join(Results_DIR, dataset_name, 'Visualizations', dataset_name,'Binary_Unions' )
            masks,_ = dataset.load_mask(image_id)
        
        #optional: set path for saving binary unions
        binary_filename = f"{base_filename}_binary.png"  # Name the file based on the original image
        binary_path = os.path.join(binary_dir, binary_filename)
        if save_binary:
            print_verbose(f"{binary_path}", verbose)
        
        if dataset_name == 'PROCI_EDMWS':
            masks = filter_mask_size(masks,min_dp_pix = 18)
        
        #compute fractal dimension (once per image)
        fractal_dim = compute_fractal_dimension(masks, save_binary=save_binary, save_path=binary_dir, show=show_binary_union, plot=plot_df)
        fractal_dims.append({'image': base_filename, 'fractal dimension': fractal_dim})
        
        #compute particle diameters
        dp_pix, area_pix = compute_equi_diam_pix(masks)

        feret, feret_x, feret_y, feret_angle, min_feret = compute_feret(masks)
        
        #compute centroids and pixel counts
        centroids, pixel_counts = compute_mask_centroids(masks)
        
        
        
        for i in range(len(dp_pix)):
            x_avg, y_avg = centroids[i]
            pixel_count = pixel_counts[i]
            dp_nm = dp_pix[i]*scale
            
            records.append({
                "image": base_filename,
                "scale_length (nm)": scale,
                "scale_area (nm^2)": scale**2,
                "PP #": i+1,
                "dp (pix)": dp_pix[i],
                "dp (nm)" : dp_nm,
                "PP area (pix)": area_pix[i],
                "PP area (nm^2)": 0.25*np.pi*dp_nm**2,
                "x_avg": x_avg,
                "y_avg": y_avg,
                "num_pixels": pixel_count,
                "feret (pix)": feret[i],
                "feret (nm)": feret[i]*scale,
                "feret_x (pix)": feret_x[i],
                "feret_x (nm)": feret_x[i]*scale,
                "feret_y (pix)": feret_y[i],
                "feret_y (nm)": feret_y[i]*scale,
                "feret_angle (degrees)": feret_angle[i],
                "min_feret (pix)": min_feret[i],
                "min_feret (nm)": min_feret[i]*scale
                
            })
            dp_pix_all.append(dp_pix[i])
            dp_nm_all.append(dp_nm)
            feret_pix_all.append(feret[i])
            feret_nm_all.append(feret[i]*scale)

            
    df =pd.DataFrame(records)
    df_fd = pd.DataFrame(fractal_dims)
    
    return df, df_fd, dp_pix_all, dp_nm_all, feret_pix_all, feret_nm_all

def process_clusters(dataset_name, datasets, model_dict, Results_DIR, model_name=None, 
                      image_scales=None, verbose = 0, save_binary=False, show_binary_union=False, plot_df=0):

    cluster_records = []
    fractal_dims = []

    dataset = datasets.get(dataset_name, None)
    if dataset is None:
        raise ValueError(f"Datasets '{dataset_name}' not found.")
    
    image_ids = dataset.image_ids

    for image_id in image_ids:
        image = dataset.load_image(image_id)
        original_filename = dataset.image_info[image_id]['basename']
        base_filename = os.path.splitext(os.path.basename(original_filename))[0]

        
        print(f"↳Processing Image {image_id} ({base_filename})")

        #get image scales from either dict or float
        if isinstance(image_scales, dict):
            scale = image_scales.get(base_filename)
            if scale is None: 
                print(f"No scale found for {base_filename}, skipping")
        else:
            scale = image_scales
        
        #Load masks (model or dataset)
        if model_name:
            model = model_dict[model_name]['model']
            confidence = model_dict[model_name]['confidence']
            print(f"Using Model {model_name} (conf_thresh = {confidence})")
            binary_dir =  os.path.join(Results_DIR, dataset_name, 'Visualizations', f"{model_name}_{confidence}", 'Binary_Unions')
            results = model.detect([image], verbose=0)
            r = results[0]
            masks = r['masks']
        else:
            print(f"Analyzing Dataset Masks: {dataset_name}")
            binary_dir = os.path.join(Results_DIR, dataset_name, 'Visualizations', dataset_name, 'Binary_Unions')
            masks, _ = dataset.load_mask(image_id)

       

        #optional: set path for saving binary uniions
        binary_filename = f"{base_filename}_binary.png"  # Name the file based on the original image
        binary_path = os.path.join(binary_dir, binary_filename)
        if save_binary:
            print_verbose(f"{binary_path}", verbose)

        #compute fractal dims (once per image)
        fractal_dim = compute_fractal_dimension(masks, save_binary=save_binary, 
                                                save_path = binary_dir, show = show_binary_union,
                                                plot = plot_df)
        
        fractal_dims.append({'image': base_filename, 'fractal_dimension': fractal_dim})
        
        centroids, pixel_counts = compute_mask_centroids(masks)

        #Loop over clusters
        N_clusters = masks.shape[-1] #get number of clusters in image
        for i in range(N_clusters):
            mask = masks[:,:,i]
            x_avg, y_avg = centroids[i]
            area_pix = pixel_counts[i]
            area_nm2 = area_pix *(scale**2)

            #compute Rg
            Rg_pix, Rg_nm = calc_cluster_Rg(mask, scale=scale)

            cluster_records.append({
                "image": base_filename,
                "Cluster #": i + 1,
            "scale_length (nm)": scale,
            "Cluster area (pix)": area_pix,
            "Cluster area (nm^2)": area_nm2,
            "Radius of gyration (pix)": Rg_pix,
            "Radius of gyration (nm)": Rg_nm,
            "x_avg": x_avg,
            "y_avg": y_avg
            })
    
    #convert to dfs
    df_clusters = pd.DataFrame(cluster_records)
    df_fd = pd.DataFrame(fractal_dims)

    return df_clusters, df_fd



def gather_aggregate_morphology(ref_set, scales, settings, 
                               save_binary=False, show_binary=False, plot=False):
    """wrapper function to gather and save aggregate morphology information"""
    
    dataset_name = settings['dataset_name']
    datasets = settings['datasets']
    model_dict = settings['model_dict']
    Results_DIR = settings['Results_DIR']
    model_name = settings.get('model_name')
    save_results = settings.get('save_results', 0)
    verbose = settings.get('verbose', 0)
    
    #calculate dp and get mean for each image
    dataset = datasets.get(dataset_name, None)
    pp_output_dir = os.path.join(Results_DIR, ref_set, 'PP_Info')
    os.makedirs(pp_output_dir, exist_ok=True)
    if model_name:
        model = model_dict[model_name]['model']
        confidence = model_dict[model_name]['confidence']
        method = model_name
        pp_save_name = os.path.join(pp_output_dir,f"{model_name}_{confidence}_pp_info.csv")
        #add save path
    else:
        model=None
        method = dataset_name
        pp_save_name = os.path.join(pp_output_dir, f"{dataset_name}_pp_info.csv")
    #g -- Process individual particles and fractal dimension
    
    #(dataset_name, model_name=None, image_scales=None, save_binary=False, show_binary_union=False, plot_df=0):
    
    #Process individual particles (give df with dp info)
    df_particles, fractal_dims, dp_pix_all, dp_nm_all ,feret_pix_all, feret_nm_all= process_particles(dataset_name, datasets, model_dict, Results_DIR, model_name, scales, verbose, 
                                                                          save_binary, show_binary, plot)
    
    #print_verbose(f" Fractal Dimension df {fractal_dims}", verbose)
    #print_verbose(f"", verbose)
    #print_verbose(f"{df_particles.head(5)}", verbose)
    
    particles = df_particles.image.unique()
    
    N_pp = []
    mean_dp_pix = []
    SEM_mdp_pix = []
    STD_mdp_pix = []
    mean_dp_nm = []
    SEM_mdp_nm = []
    STD_mdp_nm = []
    Rg_pix = []
    Rg_nm = []
    methods = []
    scale_list = []

    for i, particle in enumerate(particles):
        particle_data = df_particles[df_particles['image']==particle].copy()
        #print(particle_data)
        scale_length = particle_data['scale_length (nm)'].mean()
        scale_list.append(scale_length)
        print_verbose(f"Processing Image: {particle}", verbose)
        print_verbose(f" ", verbose)
        
        # --Primary particle count
        N = len(particle_data)
        print_verbose(f"---> # of PP: {N}", verbose) 
        N_pp.append(N)
        print_verbose(f" ", verbose)
        
        # -- dp statistics (pixels)
        mdp_pix = particle_data['dp (pix)'].mean()
        sem_dp_pix = particle_data['dp (pix)'].sem()
        std_dp_pix =  particle_data['dp (pix)'].std()
        print_verbose(f"---> Mean dp(pix): {mdp_pix} ||| SEM: {sem_dp_pix} & STD: {std_dp_pix}", verbose)
        mean_dp_pix.append(mdp_pix)
        SEM_mdp_pix.append(sem_dp_pix)
        STD_mdp_pix.append(std_dp_pix)
        
        # -- dp statistics (nm)
        mdp_nm = particle_data['dp (nm)'].mean()
        sem_dp_nm = particle_data['dp (nm)'].sem()
        std_dp_nm =  particle_data['dp (nm)'].std()
        print_verbose(f"---> Mean dp(nm): {mdp_nm} ||| SEM: {sem_dp_nm} & STD: {std_dp_nm}", verbose)
        mean_dp_nm.append(mdp_nm)
        SEM_mdp_nm.append(sem_dp_nm)
        STD_mdp_nm.append(std_dp_nm)
        
        
        # -- Radius of Gyration
        radius = calculate_radius_of_gyration(particle_data)
        print_verbose(f"---> Radius of Gyration (nm) for {particle}: {radius}", verbose)
        Rg_nm.append(radius)
        print_verbose(f"---> Radius of Gyration (pix) for {particle}: {radius/scale_length}", verbose)
        Rg_pix.append(radius/scale_length)
        
        # -- Fractal Dimension
        fractal_dim = fractal_dims.iloc[i]['fractal dimension']
        print_verbose(f"---> Fractal Dimension for {particle}: {fractal_dim}", verbose)
        print_verbose(f"", verbose)
        methods.append(method)
        print(f"--> N: {N} | Mean dp (nm): {mdp_nm:.3f} (SEM: {sem_dp_nm:.3f}, STD: {std_dp_nm}) | Rg (nm): {radius:.3f} | Fractal Dim: {fractal_dim:.3f}")
        
        print_verbose(f"", verbose)
    # --Final Aggregate Summary                  
    aggregates = pd.DataFrame({'image': particles, 
                               "method": methods,
                               "length scale [nm/pix]": scale_list,
                               "# of PP": N_pp,
                               "Mean dp [pix]": mean_dp_pix,
                               "mdp SEM [pix]": SEM_mdp_pix,
                               "mdp STD [pix]": STD_mdp_pix,
                               "Mean dp [nm]": mean_dp_nm,
                               "mdp SEM [nm]": SEM_mdp_nm,
                               "mdp STD [nm]": STD_mdp_nm,
                               "Rg [pix]": Rg_pix,
                               "Rg [nm]": Rg_nm,
                               "fractal_dim": fractal_dims['fractal dimension'],
                               
                              })
    
    # -- Create save path based on reference dataset
    save_dir = os.path.join(Results_DIR, ref_set)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,"Aggregate_info.csv")
    
    # -- check if file exists to append if it does
    if save_results:
        df_particles.to_csv(pp_save_name, index=False)
        print(f"Saved primary particle info to {pp_save_name}")
        if os.path.exists(save_path):
            aggregates.to_csv(save_path, mode='a', header=False, index=False)
            print(f"Appended aggregate summary to {save_path}")
        else:
            aggregates.to_csv(save_path, index=False)
            print(f"Created new aggregate summary for {ref_set}: {save_path}")
                          
    return aggregates, df_particles

def gather_aggregate_morphology2(ref_set, scales, settings, 
                               save_binary=False, show_binary=False, plot=False):
    """wrapper function to gather and save aggregate morphology information"""
    
    dataset_name = settings['dataset_name']
    datasets = settings['datasets']
    model_dict = settings['model_dict']
    Results_DIR = settings['Results_DIR']
    model_name = settings.get('model_name')
    save_results = settings.get('save_results', 0)
    verbose = settings.get('verbose', 0)
    
    #calculate dp and get mean for each image
    dataset = datasets.get(dataset_name, None)
    pp_output_dir = os.path.join(Results_DIR, ref_set, 'PP_Info')
    os.makedirs(pp_output_dir, exist_ok=True)

    # --- Load Model Info
    if model_name:
        model_entry = model_dict[model_name]
        model = model_entry['model']
        confidence = model_entry['confidence']
        target_class = model_entry.get('target_class', 'particle')
        method = model_name
        pp_save_name = os.path.join(pp_output_dir,f"{model_name}_{confidence}_pp_info.csv")
        #add save path
    else:
        model=None
        all_class_ids = range(len(dataset.class_names))
        labels = dataset.get_class_labels(all_class_ids)
        if labels:
            target_class = labels[0]
        else:
            target_class = 'particle'
        method = dataset_name
        pp_save_name = os.path.join(pp_output_dir, f"{dataset_name}_pp_info.csv")
    #g -- Process individual particles and fractal dimension
    
    #(dataset_name, model_name=None, image_scales=None, save_binary=False, show_binary_union=False, plot_df=0):
    print(f"Target Class: {target_class}")

    # -----Particle based model path:
    #Process individual particles (give df with dp info)
    if target_class == 'particle':
        df_particles, fractal_dims, dp_pix_all, dp_nm_all, feret_pix, feret_nm = process_particles(dataset_name, datasets, model_dict, Results_DIR, model_name, scales, verbose, 
                                                                            save_binary, show_binary, plot)
    
    #print_verbose(f" Fractal Dimension df {fractal_dims}", verbose)
    #print_verbose(f"", verbose)
    #print_verbose(f"{df_particles.head(5)}", verbose)
    
        particles = df_particles.image.unique()
    
        N_pp, mean_dp_pix, SEM_mdp_pix, STD_mdp_pix = [], [], [] , []
        mean_dp_nm, SEM_mdp_nm, STD_mdp_nm = [],[],[]
        Rg_pix, Rg_nm , methods, scale_list = [],[],[],[]
    

        for i, particle in enumerate(particles):
            particle_data = df_particles[df_particles['image']==particle].copy()
            #print(particle_data)
            scale_length = particle_data['scale_length (nm)'].mean()
            scale_list.append(scale_length)
            print_verbose(f"Processing Image: {particle}", verbose)
            print_verbose(f" ", verbose)
            
            # --Primary particle count
            N = len(particle_data)
            print_verbose(f"---> # of PP: {N}", verbose) 
            N_pp.append(N)
            print_verbose(f" ", verbose)
            
            # -- dp statistics (pixels)
            mdp_pix = particle_data['dp (pix)'].mean()
            sem_dp_pix = particle_data['dp (pix)'].sem()
            std_dp_pix =  particle_data['dp (pix)'].std()
            print_verbose(f"---> Mean dp(pix): {mdp_pix} ||| SEM: {sem_dp_pix} & STD: {std_dp_pix}", verbose)
            mean_dp_pix.append(mdp_pix)
            SEM_mdp_pix.append(sem_dp_pix)
            STD_mdp_pix.append(std_dp_pix)
            
            # -- dp statistics (nm)
            mdp_nm = particle_data['dp (nm)'].mean()
            sem_dp_nm = particle_data['dp (nm)'].sem()
            std_dp_nm =  particle_data['dp (nm)'].std()
            print_verbose(f"---> Mean dp(nm): {mdp_nm} ||| SEM: {sem_dp_nm} & STD: {std_dp_nm}", verbose)
            mean_dp_nm.append(mdp_nm)
            SEM_mdp_nm.append(sem_dp_nm)
            STD_mdp_nm.append(std_dp_nm)
            
            
            # -- Radius of Gyration
            radius = calculate_radius_of_gyration(particle_data)
            print_verbose(f"---> Radius of Gyration (nm) for {particle}: {radius}", verbose)
            Rg_nm.append(radius)
            print_verbose(f"---> Radius of Gyration (pix) for {particle}: {radius/scale_length}", verbose)
            Rg_pix.append(radius/scale_length)
            
            # -- Fractal Dimension
            fractal_dim = fractal_dims.iloc[i]['fractal dimension']
            print_verbose(f"---> Fractal Dimension for {particle}: {fractal_dim}", verbose)
            print_verbose(f"", verbose)
            methods.append(method)
            print(f"--> N: {N} | Mean dp (nm): {mdp_nm:.3f} (SEM: {sem_dp_nm:.3f}, STD: {std_dp_nm}) | Rg (nm): {radius:.3f} | Fractal Dim: {fractal_dim:.3f}")
            
            print_verbose(f"", verbose)
        # --Final Aggregate Summary                  
        aggregates = pd.DataFrame({'image': particles, 
                                "method": methods,
                                "target_class": target_class,
                                "length scale [nm/pix]": scale_list,
                                "# of PP": N_pp,
                                "Mean dp [pix]": mean_dp_pix,
                                "mdp SEM [pix]": SEM_mdp_pix,
                                "mdp STD [pix]": STD_mdp_pix,
                                "Mean dp [nm]": mean_dp_nm,
                                "mdp SEM [nm]": SEM_mdp_nm,
                                "mdp STD [nm]": STD_mdp_nm,
                                "Rg [pix]": Rg_pix,
                                "Rg [nm]": Rg_nm,
                                "fractal_dim": fractal_dims['fractal dimension'],
                                
                                })
    #------------------------
    #Cluster-based-model path
    #-------------------------
    elif target_class == 'cluster':
        #here, just extract cluster maskls and compute aggregate level Rg and dF
        print(f" Running in cluster mode for model: {model_name}")

        df_clusters, fractal_dims = process_clusters(dataset_name, datasets, 
                                                     model_dict, Results_DIR, 
                                                     model_name, 
                                                     scales, verbose,
                                                     save_binary, show_binary, plot)
        
        aggregates_records = []
        
        for idx, row in df_clusters.iterrows():
            fd = fractal_dims.loc[fractal_dims['image'] == row['image'], 'fractal_dimension'].values[0]

            aggregates_records.append({
                "image": row['image'],
                "method": method,
                "target_class": target_class,
                "length scale [nm/pix]": row['scale_length (nm)'],
                "Rg [pix]": row['Radius of gyration (pix)'],
                "Rg [nm]": row['Radius of gyration (nm)'],
                "fractal_dim": fd

            })

        aggregates = pd.DataFrame(aggregates_records)
        df_particles = pd.DataFrame()
        df_clusters_return = df_clusters
        clust_output_dir = os.path.join(Results_DIR, ref_set, 'Cluster_Info')
        os.makedirs(clust_output_dir, exist_ok=True)
        clust_save_name = os.path.join(clust_output_dir, f"{dataset_name}_cluster_info.csv")


    # -- Create save path based on reference dataset
    save_dir = os.path.join(Results_DIR, ref_set)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,"Aggregate_info.csv")
    
    # -- check if file exists to append if it does
    if save_results:
        if not df_particles.empty:
            df_particles.to_csv(pp_save_name, index=False)
            print(f"Saved primary particle info to {pp_save_name}")
        else:
            df_clusters.to_csv(clust_save_name, index=False)
            print(f"Saved cluster info to {clust_save_name}")


        aggregates_save_name = os.path.join(save_dir, "Aggregate_info.csv") if target_class == 'particle' else os.path.join(save_dir, "Aggregate_info_clusters.csv")

        if os.path.exists(aggregates_save_name):
            aggregates.to_csv(aggregates_save_name, mode='a', header=False, index=False)
            print(f"Appended aggregate summary to {save_path}")
        else:
            aggregates.to_csv(aggregates_save_name, index=False)
            print(f"Created new aggregate summary for {ref_set}: {save_path}")


    if target_class == 'cluster':
        return aggregates, df_clusters_return                      
    else:
        return aggregates, df_particles




def calc_performance_metrics(cache, ref_name, key, inference_config, settings,sort_method='iou', **kwargs):
    
    
    datasets = cache.datasets
    model_dict = cache.model_dict
    Results_DIR = cache.results_dir
    
    save_results = settings.get('save_results', 0)
    verbose = settings.get('verbose', 0)
    kwargs['inference_config'] = inference_config
    
    ref_data = datasets.get(ref_name, None)
    #metrics_list = []
    
    if key in model_dict:
        model_entry = model_dict[key]
        model = model_entry.get('model', None)
        confidence = model_entry['confidence']
        #model = model_dict[model_name]['model']
        #confidence = model_dict[model_name]['confidence']
        method = key
        
        filtered_df = cache.get_processed_matches(ref_name, model_name=key, gt2_name=None, sort_method="confidence", iou_threshold=0, recompute=False, **kwargs)
        
        #process_matches(model, ref_data, inference_config, sort_method=sort_method,iou_threshold=0, verbose=False)
        #pp_save_name = os.path.join(pp_output_dir,f"{model_name}_{confidence}_pp_info.csv")
        #add save path

    elif key in datasets:
        model=None
        method = key
        dataset = datasets.get(key, None)

        filter_flag = key in ['PROCI_EDMWS', 'PROCI_EDMWS2']

        print(f"dataset_name: {key}, loaded dataset: {dataset}")
        
        filtered_df = cache.get_processed_matches(dataset_name=ref_name,
                                                  gt2_name=key,
                                                  sort_method=sort_method,
                                                  iou_threshold=0,
                                                  recompute=False, 
                                                  **kwargs)
        
    else:
        raise ValueError(f"Key {key} not found in model_dicts or datasets")
    
    #print(filtered_df)
    
    gt_tot = np.array([])
    pred_tot = np.array([])
    mAP_conf = []
    mAP_iou = []
    metrics_list = []
    mAP_ = []
    AP75_ = []
    mAP_range_ = []
    precision_range_ = []
    recall_range_ = []
    
    
    print_verbose(f"Analyzing {method} performance on {ref_name}", verbose, 1)
    
    print_verbose(f"Sorting {method} predictions by {sort_method}", verbose, 1)
    
    tp_list = []
    fp_list = []
    fn_list = []
    prec_list = []
    
    for image_id in ref_data.image_ids:
        #get data from reference (ground truth) dataset
        image, image_meta, gt_class_id, gt_bbox, gt_mask = cache.get_gt(ref_name, inference_config, image_id)
        info = ref_data.image_info[image_id]
        original_filename = ref_data.image_info[image_id]['basename']
        base_filename = os.path.splitext(os.path.basename(original_filename))[0]
      
        
        tp =0
        fp=0
        fn=0
        
        if model:
            #results = model.detect([image], verbose=0)
            #r = results[0]
            r = cache.get_model_pred(key, ref_name, inference_config, image_id)
            
            print_verbose(f" Processing image {base_filename} (ID: {image_id})", verbose,1)
            print_verbose(f"Detected {len(r['class_ids'])} predictions for {base_filename}", verbose, 1)
            
            gt, pred = gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
            gt = np.array(gt).astype(int)
            pred = np.array(pred).astype(int)
    
            gt_tot = np.append(gt_tot, gt)
            pred_tot = np.append(pred_tot, pred)
            #print("the actual len of the gt vect is : ", len(gt_tot))
            #print("the actual len of the pred vect is : ", len(pred_tot))
            print_verbose(f"Ground Truth: {len(gt_class_id)} objects | Predictions: {len(r['class_ids'])} objects", verbose,1)
            
            if sort_method =='confidence':
                '''calcualates metrics using confidence scores to order and decide matches'''
                
                AP_, precision_, recall_, overlap_= compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
                mAP_.append(AP_)
                
                AP75, _, _, _ = compute_ap(gt_bbox,gt_class_id,gt_mask,r['rois'],
                                                 r['class_ids'], r['scores'], r['masks'], iou_threshold=0.75)
                
                AP75_.append(AP75)
                mean_ap_ = compute_ap_range(gt_bbox, gt_class_id, gt_mask, 
                                                                          r['rois'], r['class_ids'], r['scores'], r['masks'], verbose=verbose)
                
                mAP_range_.append(mean_ap_)
                
                print_verbose(f"AP (conf sort) for {base_filename}: {AP_}",verbose ,2)
                print_verbose(f"AP 0.75: {AP75}", verbose,2)
                print_verbose(f"", verbose,2)
                print_verbose(f"AP(ranged) (conf sort) for {base_filename}: {mean_ap_}",verbose,2)
                #print_verbose(f"Precision (conf sort) for {base_filename}: {precision_conf}", verbose)
                #print_verbose(f"Recall (conf sort) for {base_filename}: {recall_conf}", verbose)
                
            if sort_method =='iou':
                '''use IoU as a 'pseudo score' and sort the same way confidence scores 
                sorted for use in matching and AP calculations'''
                overlaps = compute_overlaps_masks(gt_mask, r['masks'])
                
                pseudo_scores = np.max(overlaps,axis=0)
                sorted_ix = np.argsort(pseudo_scores)[::-1]
                
                pred_boxes= r['rois'][sorted_ix]
                pred_class_ids= r['class_ids'][sorted_ix]
                pseudo_scores = pseudo_scores[sorted_ix]
                pred_masks = r['masks'][..., sorted_ix]
                
                AP_, precision_, recall_, overlap_ = compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          pred_boxes, pred_class_ids, pseudo_scores, pred_masks)
                mAP_.append(AP_)
                AP75, _, _, _= compute_ap(gt_bbox, gt_class_id, gt_mask, pred_boxes, 
                                            pred_class_ids, pseudo_scores, pred_masks, iou_threshold=0.75)
                
                AP75_.append(AP75)
                mean_ap_  = compute_ap_range(gt_bbox, gt_class_id, gt_mask, 
                                            pred_boxes, pred_class_ids, pseudo_scores, pred_masks, verbose=verbose)
                
                mAP_range_.append(mean_ap_)
                
                
                print_verbose(f"AP (iou sort) for {base_filename}: {AP_}", verbose,2)
                print_verbose(f"AP 0.75: {AP75}", verbose,2)
                print_verbose(f"AP(ranged) (conf sort) for {base_filename}: {mean_ap_}",verbose,2)
                
                #print_verbose(f"Precision (iou sort) for {base_filename}: {precision_iou}", verbose)
                #print_verbose(f"Recall (iou sort) for {base_filename}: {recall_iou}", verbose)
                
                
                print_verbose(f"", verbose,2)
                
        else:
            # dataset to dataset comparisn
            #get GT masks for second set from cache
            _,_, data_class_id,data_bbox, data_mask = cache.get_gt(key, inference_config, image_id)


            print_verbose(f"[Before filtering] {base_filename}: {data_mask.shape[-1]} masks", verbose,2)
            if key in ['PROCI_EDMWS', 'PROCI_EDMWS2']:
                data_mask = filter_mask_size(data_mask, min_dp_pix = 18)
                
            print_verbose(f"[After filtering] {base_filename}: {data_mask.shape[-1]} masks", verbose,2)
            print_verbose(f"\n🔍 Processing image {base_filename} (ID: {image_id})", verbose,2)
            print_verbose(f"Detected {len(data_class_id)} predictions for {base_filename}", verbose,2)
            
            gt, pred = gt_pred_lists(gt_class_id, gt_bbox, data_class_id, data_bbox)
            gt = np.array(gt).astype(int)
            pred = np.array(pred).astype(int)

            gt_tot = np.append(gt_tot, gt)
            pred_tot = np.append(pred_tot, pred)
            #print("the actual len of the gt vect is : ", len(gt_tot))
            #print("the actual len of the pred vect is : ", len(pred_tot))
            print_verbose(f"Ground Truth: {len(gt_class_id)} objects | Predictions: {len(data_class_id)} objects", verbose,2)
                
            overlaps = compute_overlaps_masks(gt_mask, data_mask)
                
            pseudo_scores = np.max(overlaps,axis=0)
            sorted_ix = np.argsort(pseudo_scores)[::-1]
                
            pred_boxes=data_bbox[sorted_ix]
            pred_class_ids= data_class_id[sorted_ix]
            pseudo_scores = pseudo_scores[sorted_ix]
            pred_masks = data_mask[..., sorted_ix]
                
            AP_, precision_, recall_, overlap_ = compute_ap(gt_bbox, gt_class_id, gt_mask,
                                      pred_boxes, pred_class_ids, pseudo_scores, pred_masks)
            print_verbose(f"{AP_}", verbose,2)
            mAP_.append(AP_)
            
            AP75, _, _, _= compute_ap(gt_bbox, gt_class_id, gt_mask, pred_boxes, 
                                            pred_class_ids, pseudo_scores, pred_masks, iou_threshold=0.75)
                
            AP75_.append(AP75)
            
             
            mean_ap_ = compute_ap_range(gt_bbox, gt_class_id, gt_mask, pred_boxes, pred_class_ids, pseudo_scores, pred_masks, verbose=verbose)
                
            mAP_range_.append(mean_ap_)
            
            
            
            print_verbose(f"AP (iou sort) for {base_filename}: {AP_}", verbose,2)
            print_verbose(f"AP(ranged) (conf sort) for {base_filename}: {mean_ap_}",verbose,2)
            
            #print_verbose(f"{mAP_}",2)
        #try ap through the df
        image_df = filtered_df[filtered_df['image_id']==image_id]
            
        tp = len(image_df[image_df['iou']>=0.5])
        fp = len(image_df[image_df['iou']<0.5])
        gtlen = len(gt_class_id)
        #print("tot_gt", gtlen)
            
        fn = gtlen - tp - fp
        
        prec_2 = tp/(tp+fp+fn)
        print_verbose(f"AP from TP,etc on {base_filename}: {prec_2}", verbose,2)
        tp_list.append(tp)
        fp_list.append(fp) 
        fn_list.append(fn)
        prec_list.append(prec_2)
            
    print_verbose(f"TP list {tp_list}", verbose,2)
    print_verbose(f"FP list {fp_list}", verbose,2)
    print_verbose(f"FN list {fn_list}",verbose,2)
    
    tp = np.sum(tp_list)  
    fp = np.sum(fp_list)  
    fn = np.sum(fn_list)
    
    print(f"CM Values TP: {tp}, FP {fp}, FN: {fn}")
    #prec_tot = tp_tot/(tp_tot+fp_tot+fn_tot)
    #print(prec_tot)
    prec_ave = np.mean(prec_list)
    #print(prec_ave)        
    
                
    mean_iou = filtered_df['iou'].mean()
    print("Mean IoU: ",mean_iou)           
    final_mAP = sum(mAP_)/len(mAP_)    
    print(f"Final AP_0.5: {final_mAP}")
    final_AP75 = sum(AP75_)/len(mAP_)
    print(f"Final AP_0.75: {final_AP75}") 
    final_mAP_range = sum(mAP_range_)/len(mAP_range_)
    print(f"Final mAP (ranged): {final_mAP_range}")
    
    accuracy = (tp) / (tp+fp+fn)
    print('Accuracy:', accuracy)
    precision = tp/(tp+fp)
    print('Precision:', precision)
    recall = tp/(tp+fn)
    print('Recall:', recall)
    F1 = tp/(tp+0.5*(fp+fn))
    print('F1 Score:',F1)
            
        
    metrics_dict = {
    "Method" : method,
    "Test Dataset": ref_name,
    "Confidence Threshold": model_dict[key]['confidence'] if model else 'N/A',
    "TP": int(tp),
    "FP": int(fp),
    "FN": int(fn),
    "Accuracy": round(accuracy,4),
    "Precision": round(precision,4),
    "Recall": round(recall,4),
    "F1": round(F1,4),
    "AP_50":round(final_mAP,4),
    "AP_75":round(final_AP75,4),
    "AP_range": round(final_mAP_range,4),
    "Mean IoU": round(mean_iou,4)
    }
    metrics_list.append(metrics_dict)
    metrics_df = pd.DataFrame(metrics_list)

    print(metrics_df.head(5))

    if save_results:
        output_path = os.path.join(Results_DIR, "Metrics") if sort_method == 'confidence' else os.path.join(Results_DIR, "Metrics", "IoUSort" )
        csv_path = os.path.join(output_path, f"{method}_metrics.csv")

        os.makedirs(output_path, exist_ok=True)
        
        if os.path.exists(csv_path):
            print(f"appended to: {csv_path}")
            metrics_df.to_csv(csv_path, mode='a',header=False, index=False)
        else:
            print(f"created at: {csv_path}")
            metrics_df.to_csv(csv_path, mode='w', header=True, index=False)

            #pp_save_name = os.path.join(pp_output_dir, f"{dataset_name}_pp_info.csv")
        
    return metrics_df



# def calc_performance_metrics2(ref_name, inference_config, settings,sort_method='iou'):
    
#     dataset_name = settings['dataset_name']
#     datasets = settings['datasets']
#     model_dict = settings['model_dict']
#     Results_DIR = settings['Results_DIR']
#     model_name = settings.get('model_name')
#     save_results = settings.get('save_results', 0)
#     verbose = settings.get('verbose', 0)
    
    
#     ref_data = datasets.get(ref_name, None)
#     if model_name:
#         model = model_dict[model_name]['model']
#         confidence = model_dict[model_name]['confidence']
#         method = model_name
        
#         filtered_df = process_matches(model, ref_data, inference_config, sort_method=sort_method,iou_threshold=0, verbose=False)
#         #pp_save_name = os.path.join(pp_output_dir,f"{model_name}_{confidence}_pp_info.csv")
#         #add save path
#     else:
#         model=None
#         method = dataset_name
#         dataset = datasets.get(dataset_name, None)
#         print(f"dataset_name: {dataset_name}, loaded dataset: {dataset}")
#         if dataset_name =='PROCI_EDMWS' or dataset_name == 'PROCI_EDMWS2':
#             filter = True
#         else: 
#             filter = False
#         filtered_df = process_matches(model=None, 
#                                        dataset_analyze=ref_data,
#                                       inference_config=inference_config,
#                                       iou_threshold=0, sort_method=sort_method,
#                                       verbose=False,
#                                       dataset_analyze2=dataset, filter=filter)
    
#     #print(filtered_df)
    
#     gt_tot = np.array([])
#     pred_tot = np.array([])
#     mAP_conf = []
#     mAP_iou = []
#     metrics_list = []
#     mAP_ = []
#     AP75_ = []
#     mAP_range_ = []
#     precision_range_ = []
#     recall_range_ = []
    
    
#     print_verbose(f"Analyzing {method} performance on {ref_name}", verbose)
    
#     print_verbose(f"Sorting {method} predictions by {sort_method}", verbose)
    
#     tp_list = []
#     fp_list = []
#     fn_list = []
#     prec_list = []
    
#     for image_id in ref_data.image_ids:
#         #get data from reference (ground truth) dataset
#         image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#             modellib.load_image_gt(ref_data, inference_config, image_id)#, use_mini_mask=False)
#         info = ref_data.image_info[image_id]
#         original_filename = ref_data.image_info[image_id]['basename']
#         base_filename = os.path.splitext(os.path.basename(original_filename))[0]
      
        
#         tp =0
#         fp=0
#         fn=0
        
#         if model:
#             results = model.detect([image], verbose=0)
#             r = results[0]
            
#             print_verbose(f"\n🔍 Processing image {base_filename} (ID: {image_id})", verbose)
#             print_verbose(f"Detected {len(r['class_ids'])} predictions for {base_filename}", verbose)
            
#             gt, pred = gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
#             gt = np.array(gt).astype(int)
#             pred = np.array(pred).astype(int)
    
#             gt_tot = np.append(gt_tot, gt)
#             pred_tot = np.append(pred_tot, pred)
#             #print("the actual len of the gt vect is : ", len(gt_tot))
#             #print("the actual len of the pred vect is : ", len(pred_tot))
#             print_verbose(f"Ground Truth: {len(gt_class_id)} objects | Predictions: {len(r['class_ids'])} objects", verbose)
            
#             if sort_method =='confidence':
#                 '''calcualates metrics using confidence scores to order and decide matches'''
                
#                 AP_, precision_, recall_, overlap_= compute_ap(gt_bbox, gt_class_id, gt_mask,
#                                           r['rois'], r['class_ids'], r['scores'], r['masks'])
#                 mAP_.append(AP_)
                
#                 AP75, _, _, _ = compute_ap(gt_bbox,gt_class_id,gt_mask,r['rois'],
#                                                  r['class_ids'], r['scores'], r['masks'], iou_threshold=0.75)
                
#                 AP75_.append(AP75)
#                 mean_ap_ = compute_ap_range(gt_bbox, gt_class_id, gt_mask, 
#                                                                           r['rois'], r['class_ids'], r['scores'], r['masks'])
                
#                 mAP_range_.append(mean_ap_)
                
#                 print_verbose(f"AP (conf sort) for {base_filename}: {AP_}",verbose )
#                 print_verbose(f"AP 0.75: {AP75}", verbose)
#                 print_verbose(f"", verbose)
#                 print_verbose(f"AP(ranged) (conf sort) for {base_filename}: {mean_ap_}",verbose)
#                 #print_verbose(f"Precision (conf sort) for {base_filename}: {precision_conf}", verbose)
#                 #print_verbose(f"Recall (conf sort) for {base_filename}: {recall_conf}", verbose)
                
#             if sort_method =='iou':
#                 '''use IoU as a 'pseudo score' and sort the same way confidence scores 
#                 sorted for use in matching and AP calculations'''
#                 overlaps = compute_overlaps_masks(gt_mask, r['masks'])
                
#                 pseudo_scores = np.max(overlaps,axis=0)
#                 sorted_ix = np.argsort(pseudo_scores)[::-1]
                
#                 pred_boxes= r['rois'][sorted_ix]
#                 pred_class_ids= r['class_ids'][sorted_ix]
#                 pseudo_scores = pseudo_scores[sorted_ix]
#                 pred_masks = r['masks'][..., sorted_ix]
                
#                 AP_, precision_, recall_, overlap_ = compute_ap(gt_bbox, gt_class_id, gt_mask,
#                                           pred_boxes, pred_class_ids, pseudo_scores, pred_masks)
#                 mAP_.append(AP_)
#                 AP75, _, _, _= compute_ap(gt_bbox, gt_class_id, gt_mask, pred_boxes, 
#                                             pred_class_ids, pseudo_scores, pred_masks, iou_threshold=0.75)
                
#                 AP75_.append(AP75)
#                 mean_ap_  = compute_ap_range(gt_bbox, gt_class_id, gt_mask, 
#                                                                          pred_boxes, pred_class_ids, pseudo_scores, pred_masks)
                
#                 mAP_range_.append(mean_ap_)
                
                
#                 print_verbose(f"AP (iou sort) for {base_filename}: {AP_}", verbose)
#                 print_verbose(f"AP 0.75: {AP75}", verbose)
#                 print_verbose(f"AP(ranged) (conf sort) for {base_filename}: {mean_ap_}",verbose)
                
#                 #print_verbose(f"Precision (iou sort) for {base_filename}: {precision_iou}", verbose)
#                 #print_verbose(f"Recall (iou sort) for {base_filename}: {recall_iou}", verbose)
                
                
#                 print_verbose(f"", verbose)
                
#         else:
#             _,_, data_class_id,data_bbox, data_mask =\
#                 modellib.load_image_gt(dataset, inference_config, image_id)#, use_mini_mask=False)
#             print_verbose(f"[Before filtering] {base_filename}: {data_mask.shape[-1]} masks", verbose)
#             if dataset_name == 'PROCI_EDMWS' or dataset_name == 'PROCI_EDMWS2':
#                 data_mask = filter_mask_size(data_mask, min_dp_pix = 18)
                
#             print_verbose(f"[After filtering] {base_filename}: {data_mask.shape[-1]} masks", verbose)
#             print_verbose(f"\n🔍 Processing image {base_filename} (ID: {image_id})", verbose)
#             print_verbose(f"Detected {len(data_class_id)} predictions for {base_filename}", verbose)
            
#             gt, pred = gt_pred_lists(gt_class_id, gt_bbox, data_class_id, data_bbox)
#             gt = np.array(gt).astype(int)
#             pred = np.array(pred).astype(int)

#             gt_tot = np.append(gt_tot, gt)
#             pred_tot = np.append(pred_tot, pred)
#             #print("the actual len of the gt vect is : ", len(gt_tot))
#             #print("the actual len of the pred vect is : ", len(pred_tot))
#             print_verbose(f"Ground Truth: {len(gt_class_id)} objects | Predictions: {len(data_class_id)} objects", verbose)
                
#             overlaps = compute_overlaps_masks(gt_mask, data_mask)
                
#             pseudo_scores = np.max(overlaps,axis=0)
#             sorted_ix = np.argsort(pseudo_scores)[::-1]
                
#             pred_boxes=data_bbox[sorted_ix]
#             pred_class_ids= data_class_id[sorted_ix]
#             pseudo_scores = pseudo_scores[sorted_ix]
#             pred_masks = data_mask[..., sorted_ix]
                
#             AP_, precision_, recall_, overlap_ = compute_ap(gt_bbox, gt_class_id, gt_mask,
#                                       pred_boxes, pred_class_ids, pseudo_scores, pred_masks)
#             print_verbose(f"{AP_}", verbose)
#             mAP_.append(AP_)
            
#             AP75, _, _, _= compute_ap(gt_bbox, gt_class_id, gt_mask, pred_boxes, 
#                                             pred_class_ids, pseudo_scores, pred_masks, iou_threshold=0.75)
                
#             AP75_.append(AP75)
            
             
#             mean_ap_ = compute_ap_range(gt_bbox, gt_class_id, gt_mask, 
#                                                                          pred_boxes, pred_class_ids, pseudo_scores, pred_masks)
                
#             mAP_range_.append(mean_ap_)
            
            
            
#             print_verbose(f"AP (iou sort) for {base_filename}: {AP_}", verbose)
#             print_verbose(f"AP(ranged) (conf sort) for {base_filename}: {mean_ap_}",verbose)
            
#             print_verbose(f"{mAP_}")
#         #try ap through the df
#         image_df = filtered_df[filtered_df['image_id']==image_id]
            
#         tp = len(image_df[image_df['iou']>=0.5])
#         fp = len(image_df[image_df['iou']<0.5])
#         gtlen = len(gt_class_id)
#         #print("tot_gt", gtlen)
            
#         fn = gtlen - tp - fp
        
#         prec_2 = tp/(tp+fp+fn)
#         print_verbose(f"AP from TP,etc on {base_filename}: {prec_2}", verbose)
#         tp_list.append(tp)
#         fp_list.append(fp) 
#         fn_list.append(fn)
#         prec_list.append(prec_2)
            
#     print_verbose(f"TP list {tp_list}", verbose)
#     print_verbose(f"FP list {fp_list}", verbose)
#     print_verbose(f"FN list {fn_list}",verbose)
    
#     tp = np.sum(tp_list)  
#     fp = np.sum(fp_list)  
#     fn = np.sum(fn_list)
    
#     print(f"CM Values TP: {tp}, FP {fp}, FN: {fn}")
#     #prec_tot = tp_tot/(tp_tot+fp_tot+fn_tot)
#     #print(prec_tot)
#     prec_ave = np.mean(prec_list)
#     #print(prec_ave)        
    
                
#     mean_iou = filtered_df['iou'].mean()
#     print("Mean IoU: ",mean_iou)           
#     final_mAP = sum(mAP_)/len(mAP_)    
#     print(f"Final AP_0.5: {final_mAP}")
#     final_AP75 = sum(AP75_)/len(mAP_)
#     print(f"Final AP_0.75: {final_AP75}") 
#     final_mAP_range = sum(mAP_range_)/len(mAP_range_)
#     print(f"Final mAP (ranged): {final_mAP_range}")
    
#     accuracy = (tp) / (tp+fp+fn)
#     print('Accuracy:', accuracy)
#     precision = tp/(tp+fp)
#     print('Precision:', precision)
#     recall = tp/(tp+fn)
#     print('Recall:', recall)
#     F1 = tp/(tp+0.5*(fp+fn))
#     print('F1 Score:',F1)
            
        
#     metrics_dict = {
#     "Method" : method,
#     "Test Dataset": ref_name,
#     "Confidence Threshold": model_dict[model_name]['confidence'] if model else 'N/A',
#     "TP": int(tp),
#     "FP": int(fp),
#     "FN": int(fn),
#     "Accuracy": round(accuracy,4),
#     "Precision": round(precision,4),
#     "Recall": round(recall,4),
#     "F1": round(F1,4),
#     "AP_50":round(final_mAP,4),
#     "AP_75":round(final_AP75,4),
#     "AP_range": round(final_mAP_range,4),
#     "Mean IoU": round(mean_iou,4)
#     }
#     metrics_list.append(metrics_dict)
#     metrics_df = pd.DataFrame(metrics_list)

#     print(metrics_df.head(5))

#     if save_results:
#         output_path = os.path.join(Results_DIR, "Metrics") if sort_method == 'confidence' else os.path.join(Results_DIR, "Metrics", "IoUSort" )
#         csv_path = os.path.join(output_path, f"{method}_metrics.csv")

#         os.makedirs(output_path, exist_ok=True)
        
#         if os.path.exists(csv_path):
#             print(f"appended to: {csv_path}")
#             metrics_df.to_csv(csv_path, mode='a',header=False, index=False)
#         else:
#             print(f"created at: {csv_path}")
#             metrics_df.to_csv(csv_path, mode='w', header=True, index=False)

#             #pp_save_name = os.path.join(pp_output_dir, f"{dataset_name}_pp_info.csv")
        
#     return metrics_df

def full_summary(aggs_df, metrics_df, info_df, settings, cache):
  
 
    Results_DIR = cache.results_dir
    save_results = settings.get('save_results', 0)
    verbose = settings.get('verbose', 0)
    
    summary_dict = {
        "Method": metrics_df['Method'].iloc[0],
        "Conf Thresh": metrics_df['Confidence Threshold'].iloc[0],
        "Target Class": aggs_df['target_class'].iloc[0],
        "TP": metrics_df['TP'].iloc[0],
        "FP": metrics_df['FP'].iloc[0],
        "FN": metrics_df['FN'].iloc[0],
        "Acc.": metrics_df['Accuracy'].iloc[0],
        "Prec.": metrics_df['Precision'].iloc[0],
        "Rec.": metrics_df['Recall'].iloc[0],
        "F1": metrics_df['F1'].iloc[0],
        "AP50": metrics_df['AP_50'].iloc[0],
        "AP75": metrics_df['AP_75'].iloc[0],
        "mAP": metrics_df['AP_range'].iloc[0],
        "Mean IoU": metrics_df['Mean IoU'].iloc[0],
        "Mean dF": round(aggs_df['fractal_dim'].mean(),4),
        "dF SEM": round(aggs_df['fractal_dim'].sem(),4),
        "df STD": round(aggs_df['fractal_dim'].std(),4),
        "Mean Rg [nm]": round(aggs_df['Rg [nm]'].mean(),4),
        "Rg SEM": round(aggs_df['Rg [nm]'].sem(),4),
        "Rg STD": round(aggs_df['Rg [nm]'].std(),4),
        
    }

    #add coverage score if available
    if 'coverage_score' in aggs_df.columns:
        summary_dict.update({
            "Mean Coverage Score": round(aggs_df['coverage_score'].mean(),4),
            "Coverage SEM": round(aggs_df['coverage_score'].sem(),4),
            "Coverage STD": round(aggs_df['coverage_score'].std(),4)

        })

    if 'leakage_frac' in aggs_df.columns:
        summary_dict.update({
            "Mean Leakage Fraction": round(aggs_df['leakage_frac'].mean(),4),
            "Leakage SEM": round(aggs_df['leakage_frac'].sem(),4),
            "Leakage STD": round(aggs_df['leakage_frac'].std(),4)
        })
    #Particle only stats (if pp_df exists)
    if 'dp (nm)' in info_df.columns:
        summary_dict.update({
            "Mean dp [nm]": round( info_df['dp (nm)'].mean(),4),
            "dp SEM": round( info_df['dp (nm)'].sem(),4),
            "dp STD": round( info_df['dp (nm)'].std(),4),
        })

        filename = "Full Summary.csv"
    else:
        summary_dict.update({
            "Mean dp [nm]": None,
            "dp SEM": None,
            "dp STD": None
        })
        filename = "Full Summary_clusters.csv"

   
    if verbose:
        for k,v in summary_dict.items():
            print(f"{k}: {v}")
            
    summary_df = pd.DataFrame([summary_dict])
    if save_results:
        output_dir = os.path.join(Results_DIR,metrics_df['Test Dataset'].iloc[0])
        #print(output_dir)
        output_path = os.path.join(output_dir, "Full Summary.csv")
        if os.path.exists(output_path):
            print(f"appended to: {output_path}")
            summary_df.to_csv(output_path, mode='a',header=False, index=False)
        else:
            print(f"created at: {output_path}")
            summary_df.to_csv(output_path, mode='w', header=True, index=False)

            #pp_save_name = os.path.join(pp_output_dir, f"{dataset_name}_pp_info.csv")
        
        
        
    return summary_df
                  
def process_method_analysis(
        ref_name, 
        keys=None, 
        cache=None, 
        save_results=False, 
        verbose=1,
        default_config=None, 
        compute_ML_metrics=False):

    #safeguard against no cache or noninitialized cache.model_dict    
    if cache is None:
        raise ValueError("cache must be provided")

    if cache.model_dict is None:
        raise ValueError("cache.model_dict is not initialized")


    #auto fill if no keys passed
    if keys is None:
        keys = list(cache.model_dict.keys())
        if verbose:
            
            print(f"[Auto] Processing all loaded methods: {keys}")
    
    all_summaries={}
    analyzer=MaskAnalyzer(cache)

    ref_has_gt = cache.requires_gt(ref_name)
   
    for key in keys:
        print(f"\nProcessing: {key}")

        settings={'save_results': save_results,
                  'verbose': verbose,
                  }
        

        info = cache.get_method_info(key)

        if info is None:
            raise ValueError(f"Unknown method: {key}")
        
        is_model = info["is_model"]
        sort_method = info["sort_method"]


        #config handling


        if is_model:
            settings['inference_config']= cache.model_dict[key]['config']
      
        else:
            if default_config is None:
                raise ValueError(
                    "default_config must be provided for dataset keys (non-model inputs)"
                )
            
            settings['inference_config'] = default_config
           

        
        #Step 1: Aggregate morphology
        agg_summary, pp_info = analyzer.gather_aggregate_morphology(ref_name, key, settings)

        if verbose:
            mean_cov = agg_summary['coverage_score'].mean()
            print(f"Mean Coverage Score: {mean_cov:.3f}")

        #Step 2: ML metrics (only if provided GT dataset and compute_ML_metrics=True)
        metrics_df = None
        if compute_ML_metrics and ref_has_gt:
            metrics_df = calc_performance_metrics(cache, ref_name, key, inference_config=settings['inference_config'], settings=settings, sort_method=sort_method)
        
        #SAFE metrics df fallback
        if metrics_df is None:
            metrics_df = pd.DataFrame([{
                "Method": info['method'],
                "Confidence Threshold": info.get("confidence", "N/A"),
                "Target Class": info['target_class'],
                "TP": np.nan,
                "FP": np.nan,
                "FN": np.nan,
                "Accuracy": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1": np.nan,
                "AP_50": np.nan,
                "AP_75": np.nan,
                "AP_range": np.nan,
                "Mean IoU": np.nan,
            }])


        #Step 3: compile into full summary
        summary = full_summary(agg_summary, metrics_df, pp_info, settings, cache)

        all_summaries[key] = summary

    return all_summaries

####### Confusion matrix ########

#function 1 to be added to your utils.py
def get_iou(a, b, epsilon=1e-5):
    """ 
    Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args: 
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


#function 2 to be added to your utils.py
def gt_pred_lists(gt_class_ids, gt_bboxes, pred_class_ids, pred_bboxes, iou_tresh = 0.5):

    """ 
        Given a list of ground truth and predicted classes and their boxes, 
        this function associates the predicted classes to their gt classes using a given Iou (Iou>= 0.5 for example) and returns 
        two normalized lists of len = N containing the gt and predicted classes, 
        filling the non-predicted and miss-predicted classes by the background class (index 0).

        Args    :
            gt_class_ids   :    list of gt classes of size N1
            pred_class_ids :    list of predicted classes of size N2
            gt_bboxes      :    list of gt boxes [N1, (x1, y1, x2, y2)]
            pred_bboxes    :    list of pred boxes [N2, (x1, y1, x2, y2)]
            
        Returns : 
            gt             :    list of size N
            pred           :    list of size N 

    """

    #dict containing the state of each gt and predicted class (0 : not associated to any other class, 1 : associated to a class)
    gt_class_ids_ = {'state' : [0*i for i in range(len(gt_class_ids))], "gt_class_ids":list(gt_class_ids)}
    pred_class_ids_ = {'state' : [0*i for i in range(len(pred_class_ids))], "pred_class_ids":list(pred_class_ids)}

    #the two lists to be returned
    pred=[]
    gt=[]

    for i, gt_class in enumerate(gt_class_ids_["gt_class_ids"]):
        for j, pred_class in enumerate(pred_class_ids_['pred_class_ids']): 
            #check if the gt object is overlapping with a predicted object
            if get_iou(gt_bboxes[i], pred_bboxes[j])>=iou_tresh:
                #change the state of the gt and predicted class when an overlapping is found
                gt_class_ids_['state'][i] = 1
                pred_class_ids_['state'][j] = 1
                #gt.append(gt_class)
                #pred.append(pred_class)
                
                #chack if the overlapping objects are from the same class
                if (gt_class == pred_class):
                    gt.append(gt_class)
                    pred.append(pred_class)
                #if the overlapping objects are not from the same class 
                else : 
                    gt.append(gt_class)
                    pred.append(pred_class)
                
    #look for objects that are not predicted (gt objects that dont exists in pred objects)
    for i, gt_class in enumerate(gt_class_ids_["gt_class_ids"]):
        if gt_class_ids_['state'][i] == 0:
            gt.append(gt_class)
            pred.append(0)
            #match_id += 1
    #look for objects that are mispredicted (pred objects that dont exists in gt objects)
    for j, pred_class in enumerate(pred_class_ids_["pred_class_ids"]):
        if pred_class_ids_['state'][j] == 0:
            gt.append(0)
            pred.append(pred_class)
    return gt, pred



#########  Print confusion matrix for the whole dataset and return tp,fp and fn ##########
#########  The style of this confusion matrix is inspired from https://github.com/wcipriano/pretty-print-confusion-matrix ##########

def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1
#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0, show_percentages=False, show_totals = False, show_total_percentages=False):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)
    
    
    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        """
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

        #calc luminence
        r, g, b, _ = carr
        luminance = r * 0.3 + g * 0.59 + b * 0.11

        if luminance <0.5:
            text_color = 'w'
        else:
            text_color = 'k'

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color = text_color, ha="center", va="center", gid='sum', fontproperties=font_prop)
        #text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)


        if show_total_percentages:
            lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        else: 
            lis_txt = ['%d'%(cell_val)]
        lis_kwa = [text_kwargs]


        #dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        #dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        #for i in range(len(lis_txt)):
         #   newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
          #  #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
           # text_add.append(newText)
        #print '\n'

        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);


        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=text_kwargs)
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)"""
    else:
        if(per > 0):
            if show_percentages:
                txt = '%s\n%.2f%%' %(cell_val, per)
            else: txt = '%s' %(cell_val)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #calculate luminence based on the cell background color
        r, g, b, _ = facecolors[posi]
        luminence = r*0.3 + g*0.59 + b*0.11 
        if luminence < 0.5:
            oText.set_color('white') #use white text for dark backgrounds
        else:
            oText.set_color('black') #use black text for light backgrounds


        #main diagonal
        if(col == lin):

            #calculate luminence based on the cell background color
            r, g, b, _ = facecolors[posi]
            luminence = r*0.3 + g*0.59 + b*0.11 
            if luminence < 0.5:
                oText.set_color('white') #use white text for dark backgrounds
            else:
                oText.set_color('black') #use black text for light backgrounds

            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')


    return text_add, text_del
#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')
#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="viridis", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y,',show_totals=False,save_path=None):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    if show_totals:
        # create "Total" column
        insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    sn.set(font_scale=1.8)
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)
    

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 26)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 26)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
    if save_path:
        plt.savefig(save_path)
    plt.show()
#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="viridis",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[36,36], show_null_values=0, pred_val_axis='lin',show_totals=False, save_path=None):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
        return the tp, fp and fn
    """

    #data
    if(not columns):
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:max(len(np.unique(y_test)),len(np.unique(predictions)))]]
    
    y_test = np.array(y_test)
    predictions = np.array(predictions)
    #confusion matrix 
    confm = confusion_matrix(y_test, predictions)
    num_classes = len(columns)
    
    #compute tp fn fp 
    
    fp=[0]*num_classes
    fn=[0]*num_classes
    tp=[0]*num_classes
    tn=[0]*num_classes
    for i in range(confm.shape[0]):
        fn[i]+=np.sum(confm[i])-np.diag(confm)[i]
        fp[i]+=np.sum(np.transpose(confm)[i])-np.diag(confm)[i]
        for j in range(confm.shape[1]):
            if i==j:
                tp[i]+=confm[i][j]
                #print(confm)
    #compute tn
    for i in range(num_classes):
        tn[i] = np.sum(confm) - (tp[i] + fp[i] + fn[i])  # Total - (TP + FP + FN)
    
    #print(confm)
    #plot
    df_cm = DataFrame(confm, index=columns, columns=columns)

    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, 
        pred_val_axis=pred_val_axis, lw=lw, fmt=fmt, show_totals=show_totals, save_path=save_path)
    
    return tp, fp, fn, tn
  