"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import random
import itertools
import colorsys
import io
import cv2


import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
import IPython.display
import seaborn as sns
from PIL import Image
from tqdm import tqdm


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils

from mrcnn.utils import print_verbose
import mrcnn.model as modellib

############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), figAx=None,
                      show_mask=True, show_bbox=True,
                      show_caption=True,
                      colors=None, captions=None, save_path=None, view=True, linewidth=2):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    """image copy for furthere analysis"""
    unmaskedimage = image.copy()
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not figAx:
        fig,ax = plt.subplots(1, figsize=figsize)
        auto_show = True
    else:
        fig,ax = figAx

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    # print("image_size is {}".format(image.shape))
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                   alpha=0.7, linestyle="dashed",
                                   edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if show_caption:
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color, linewidth = linewidth)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0,orientation= 'landscape')
        
    if view:
        if auto_show:
            plt.show()

    else:
        if figAx is None:
            plt.close(fig)
        

def display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.5, score_threshold=0.5):
    """Display ground truth and prediction instances on the same image."""
    # Match predictions to ground truth
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    # Ground truth = green. Predictions = red
    colors = [(0, 1, 0,.8)] * len(gt_match)\
           + [(1, 0, 0, 1)] * len(pred_match)
    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    # Captions per instance show score/IoU
    captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
        pred_score[i],
        (overlaps[i, int(pred_match[i])]
            if pred_match[i] > -1 else overlaps[i].max()))
            for i in range(len(pred_match))]
    # Set title if not provided
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    #create ax if not passed
    #if ax is None:
    #    fig, ax = plt.subplots(1, figsize=(12, 12))  # Define the figure and axis
    #    figAx = (fig, ax)  # Pass the figure and axis as a tuple to display_instances
    #else:
    #    figAx = None  # If ax is provided, don't create a new figure (pass None)
    
    # Display
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, figAx=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)


def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
                                  [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
                class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        print(class_id)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictions and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with different
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominent each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    display_table(table)

############ SAGE Visualizations Added ###############    
    
def pad_bbox(dataset, model, output_folder, inference_config, class_name="cluster", min_confidence = 0.5,
             image_ids=None, override_class_names=True, save_as_images=True, 
             save_as_matrix=False,pad_particle_masks=False,start_index=0, debug=False, pad_extend_factor=0.1):
        
    import mrcnn.model as modellib
    #create folder if it doesn't exist
    if save_as_images:
        os.makedirs(output_folder, exist_ok=True)
        
    #optionally limit to certain subset:
    if image_ids is None:
        image_ids = dataset.image_ids
        print("image_ids:", image_ids)
            
    #set detection threshold (maybe reduntant)
    model.config.DETECTION_MIN_CONFIDENCE = min_confidence
        
    padded_images = []
    #padded_masks = []

        
    for image_id in image_ids:
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config, image_id)
            
        #get original filename
        original_filename = dataset.image_info[image_id]['basename']
        
        if debug:
            print(f"\nProcessing Image ID: {image_id}")
            print(f"Original filename (raw): {original_filename}")
            print(f"Type of original filename: {type(original_filename)}")
                                                         
        base_filename = os.path.splitext(str(original_filename))[0] #remove extension
        
        #run object detection
        results = model.detect([original_image], verbose=0)
        r=results[0]
        
        if debug:
            print(f"\n--- Image ID: {image_id} ---")
            print(f"Detected Classes: {r['class_ids']}")
            print(f"Scores: {r['scores']}")
            print(f"ROIs: {r['rois']}")
            
        #create copy of original image to apply padding
        #padded_image = np.full_like(original_image,255) #start with white canvas ---Moved to right before copying the roi
        """if pad_particle_masks and gt_mask.size>0: 
            particle_mask_padded = np.zeros_like(gt_mask)
        else None"""
        
        
      
        boxes = r['rois']
        class_ids = r['class_ids']
        class_names = dataset.class_names
            
        cluster_count = 1
        
        for i, box in enumerate(boxes):
            class_id = class_ids[i]
            if override_class_names:
                label = class_name
            else:
                label = model.class_names[class_id]
            print(label)
                
            #check if detected class is cluster
            if label == class_name:
                y1, x1, y2, x2 = box #bbox coords
                if debug:
                    print(f"Original bbox for image {image_id}, cluster {cluster_count}: x1:{x1},y1:{y1}, x2:{x2},y2:{y2}")
                cluster_count += 1
                
                #if extended padding is specified, pad a little less (more unpadded region to avoid cutting off spots
                if pad_extend_factor:
                    #calc extra space
                    height = y2-y1
                    width = x2-x1
                    
                    #calc how much extra to pad
                    extend_y = int(height * pad_extend_factor)
                    extend_x = int(width * pad_extend_factor)
                    
                    #extend bbox coords 
                    y1 = max(y1 - extend_y, 0) #doesnt go below 0
                    x1 = max(x1 - extend_x, 0)
                    y2 = min(y2 + extend_y, original_image.shape[0]) #ensure it doesnt go beyond image heigt
                    x2 = min(x2 + extend_x, original_image.shape[1]) #same with width
                    
                    if debug:
                        print(f"new bbox for image {image_id}, cluster {cluster_count}: x1:{x1},y1:{y1}, x2:{x2},y2:{y2}")
                
                #create copy of original image to apply padding
                padded_image = np.full_like(original_image,255) #start with white canvas
                """if pad_particle_masks and gt_mask.size>0: 
                       particle_mask_padded = np.zeros_like(gt_mask)
                       else None"""
                #copy ROI from original image to padded image
                padded_image[y1:y2,x1:x2] = original_image[y1:y2,x1:x2]
                
                
                if save_as_images:
                    padded_filename = f"{base_filename}_{class_name}_{cluster_count:02d}.png"
                    padded_path = os.path.join(output_folder, padded_filename)
                    cv2.imwrite(padded_path, padded_image)
                    if debug:
                        print(f"Saved padded {class_name} image: {padded_filename}")
                    start_index+= 1
                    cluster_count +=1
                #append cropped image as matrix to the list
                if save_as_matrix:
                    padded_images.append(padded_image)
                    
    print(f"Finished processing {len(padded_images)} images.")
    
    if save_as_matrix:
        return padded_images, start_index
    else: 
        return start_index
        
                
def plot_rev_cum_iou(methods, model_dict, dataset_dict, ref_dataset,config,   save_dir = None, 
                     iou_threshold=0, iou_summary=False, method_styles=None, fill=True, title=False, verbose=False,):
    """Processes matches for multiple models, prints IoUs summaries, and plots the reverse cumulative IoU dist
    Args:
    models (dict): dictionary where keys are model names 
    dataset : dataset to analyze
    config: 
    save_dir (str): directory to save plot
    iou_threshold (float): IoU threhold for processing matches
    verbose (bool): verbosity flag for match processing
    iou_summary (bool): prints iou summary for each model if True
    """
    
    #dict to store IoU Dfs
    all_ious = {}
    
    
    ref_data = dataset_dict.get(ref_dataset, None)
    #process each model and print IoU summary and store results
    for method in methods:
        if method in model_dict:
            model_info = model_dict.get(method)
            model = model_info.get('model')
            confidence = model_info.get('confidence')
            print(f"Using Model: {method}")
            filtered_df = utils.process_matches(model, ref_data, config,sort_method="confidence",
                                           iou_threshold=iou_threshold, 
                                           dataset_analyze2 = None, verbose=verbose, filter=False)
            all_ious[method] = filtered_df
    
        elif method in dataset_dict:
            model=None
            dataset = dataset_dict.get(method, None)
            print(f"Using Dataset: {method}")
            if method == 'PROCI_EDMWS':
                filter = True
            else:
                filter = False
                
            filtered_df = utils.process_matches(model, ref_data, config, sort_method="iou",
                                          iou_threshold=iou_threshold, dataset_analyze2=dataset, filter=filter)
            all_ious[method] = filtered_df
            
                
        else:
            
            print(f"Method {method} not found in models or datasets")
        if iou_summary:
                utils.print_iou_summary(filtered_df, method)
            
        if verbose:
            print(f"{method} - Filtered Df shape: {filtered_df.shape}")   
                
            
            
            
    
        
    #plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(12,8))


    #default styles if none are passed
    if method_styles is None:
        colors = sns.color_palette("tab10", len(methods))
        method_styles = {
            method: {"label": method, "color": colors[i]} for i, method in enumerate(methods)
        }
    
    #loop through each model to plot ecdf and fill area
    for (method_name, method_ious) in all_ious.items():
        style = method_styles.get(method_name, {})
        label = style.get('label', method_name)
        color = style.get('color', None)

        method_ious['method']=method_name
        sns.ecdfplot(data=method_ious,
                    x='iou',
                    hue='method',
                    label=label,
                    stat='count',
                    complementary=True,
                    palette=[color] if color else None)
        if fill:
            #calculate sorted iou values and cumulative counts for filling
            sorted_iou = np.sort(method_ious['iou'])[::-1]
            cum_count=np.arange(1,len(sorted_iou)+1)
            #fill area under ecdf curve with approriate color
            
            plt.fill_between(sorted_iou, 
                             cum_count, 
                             color=color if color else 'gray',
                             alpha=0.5)
            
    #add titles and labels
    if title:
        plt.title('Reverse Cumulative IoU Distribution',fontsize='29', weight='bold')
    plt.xlabel('IoU Threshold',fontsize='29')
    plt.ylabel('# of predictions above IoU',fontsize='29')
    
    plt.tick_params(axis='both', which='both', direction='inout', 
                    length=6, labelsize=25)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=20)
    #add vertical line at iou=0.5
    plt.axvline(x=0.5, color= 'black', linestyle='--', label='IoU=0.5')
    
    plt.legend(loc='best', fontsize=25)
    
    plt.tight_layout
    
    method_names = "_".join(all_ious.keys())
    
   
    if save_dir:
        #create save dir if not exists
        os.makedirs(save_dir, exist_ok=True)
        savename = os.path.join(save_dir, f'IoU_dist_{method_names}_{iou_threshold}.png')
        plt.savefig(savename, dpi=300, bbox_inches='tight')
        
    plt.show()          
                
def display_inst_heat(image, boxes, masks, class_ids, class_names,
                      scores=None,
                      metric=None, metric_name="Metric",normalize_metric=False, norm_range=(0.5,1.0),
                      title="",
                      figsize=(16, 16), figAx=None,
                      show_mask=True, show_bbox=True,
                      show_caption=True, show_cbar=True,
                      colormap='viridis',
                      cbar_position = [0.85, 0.15, 0.03, 0.7], label_color='black', 
                      captions=None, save_path=None,
                      show_pred_idx = False,
                     unmatched_color = (30 / 255, 144 / 255, 1.0)  ):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    """image copy for furthere analysis"""
    unmaskedimage = image.copy()
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not figAx:
        fig,ax = plt.subplots(1, figsize=figsize)
        auto_show = True
    else:
        fig,ax = figAx
    
    colormap_func = plt.get_cmap(colormap)
    norm = None
    #if metric is not None:
    if metric is not None and normalize_metric:
        #if len(metric) !=N:
         #   raise ValueError(f"The lenght of the metric array ({len(metric)}) "
          #                   f" must match the number of instances ({N}).")
        
        if norm_range is not None:
            norm = Normalize(vmin=norm_range[0], vmax=norm_range[1])
        else:
            norm = Normalize(vmin=np.min(metric), vmax=np.max(metric))

        #normalize metric if required:
       # if normalize_metric:
            #norm = Normalize(vmin=norm_range[0], vmax=norm_range[1]) if norm_range is not None else Normalize(vmin=np.min(metric), vmax = np.max(metric))
 
        #else: 
            #norm = None
            
        #apply colormap to metric
        #colormap = plt.get_cmap(colormap)
        #colors = [colormap(norm(value) if norm else value) for value in metric]
    colors = []
    
    for i in range(N):
        if metric is not None and i < len(metric) and metric[i] is not None and not np.isnan(metric[i]):
            value = metric[i]
            colors.append(colormap_func(norm(value) if norm else value))
        else:
            colors.append(unmatched_color)
    

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    # print("image_size is {}".format(image.shape))
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        
        has_metric = metric is not None and i< len(metric) and metric[i] is not None
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                   alpha=0.7, linestyle="dashed",
                                   edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if show_caption:
            if not captions:
                class_id = class_ids[i]
                score = metric[i] if metric is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score is not None else label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)
            
            if show_pred_idx:
                y_indices, x_indices = np.where(mask)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    ax.text(center_x, center_y, str(i), color=label_color,
                        fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.3'))
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    
    #display colorbar
    if metric is not None and show_cbar:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes(cbar_position)
        cbar = fig.colorbar(sm,cax=cbar_ax, orientation='horizontal', label=metric_name, fraction=0.02, pad=0.04)
         # Adjust the position and size of the colorbar to fit in the image without padding
        
        cbar.ax.tick_params(labelsize=16, labelcolor=label_color)
        cbar.set_label(metric_name,color=label_color,fontsize=20)
        
        #fig.subplots_adjust(right=0.96)
        #cbar.ax.set_position(cbar_position)  # Adjust these values to fit the image better
    
    
    if save_path is not None:
        plt.tight_layout(pad=0.0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0,orientation= 'landscape')
        plt.close()
    plt.tight_layout(pad=0.0)

    if auto_show:
        plt.show()        
        
        
def get_filtered_df(method_name, ref_name, model_dict, datasets, config, filter_size=False, verbose=0):
    
    #load reference data (i.e. ground truth to check against)
    ref_data = datasets.get(ref_name, None)
    if ref_data is None:
        raise ValueError(f"Reference dataset '{ref_name}' not found.")
        
    if method_name in model_dict:
        model_info = model_dict.get(method_name)
        model = model_info.get('model')
        confidence = model_info.get('confidence')
        print_verbose(f"Using Model: {method_name}", verbose)
        filtered_df = utils.process_matches(model, ref_data, config,sort_method="confidence",
                                    iou_threshold=0, 
                                    dataset_analyze2 = None, verbose=0, filter=False)
        full_name = f"{method_name}_{confidence}"
        return filtered_df, full_name, model, None, ref_data
    
    elif method_name in datasets:
        model = None
        dataset = datasets.get(method_name, None)
        print_verbose(f"Using dataset: {method_name}", verbose)
        filtered_df = utils.process_matches(model, ref_data, config, sort_method="iou",
                                        iou_threshold=0, dataset_analyze2=dataset, filter=filter_size, verbose=verbose)
        full_name = f"{method_name}"
        return filtered_df, full_name, None,dataset, ref_data
    
    else:
        raise ValueError(f"'{method_name}' not found in models or datasets.")
        

def get_iou_heatmaps(filtered_df, ref_data, config, model=None, dataset=None,
                     method_name = "", ref_name = "", Results_DIR=None, verbose=False, save_images = False, vis_settings=None):

    if verbose: 
        utils.print_iou_summary(filtered_df, method_name)
        
    #create vis dir
    vis_dir = os.path.join(Results_DIR, ref_name, "Visualizations", f"{method_name}","IoU_Heatmaps")
    os.makedirs(vis_dir, exist_ok=True)
    #maybe change
    #filtered_iou_values_new = filtered_df.to_dict(orient='records')
    
    average_ious = []
    
    image_ids=ref_data.image_ids
    for image_id in image_ids:
        image = ref_data.load_image(image_id)
        
        #maybe change
        #ious_for_image = [item for item in filtered_iou_values_new if item['image_id'] == ref_data.image_info[image_id]['id']]
        
        original_filename = ref_data.image_info[image_id]['basename']
        base_filename = os.path.splitext(os.path.basename(original_filename))[0] 
        print(f"Image ID: {image_id}, File Path: {base_filename}")
        
        iou_df = filtered_df[filtered_df['image_id']==image_id]
        
        if verbose:
            print(f"filtered dataframe for Image ID {image_id}")
            print(iou_df)
            
        if model:
            results = model.detect([image], verbose=1)

            r = results[0]
            scores = r['scores']
            print_verbose(("SCORES:",r['scores']),verbose)
            rois, masks, class_ids = r['rois'], r['masks'], r['class_ids']
            
        elif dataset:
             _,_, class_ids,rois, _ =\
                    modellib.load_image_gt(dataset, config, image_id)#, use_mini_mask=False)
    
             masks, _ = dataset.load_mask(image_id)   
        
        else:
            raise ValueError("Either model or dataset must be provided")
            
                      
        ious_for_image_values = np.full(len(class_ids), np.nan, dtype=np.float32)
        #ious_for_image_values = np.zeros(len(class_ids))  
        
        for pred_idx in range(len(class_ids)):
            iou_row = iou_df[iou_df['pred_index'] == pred_idx]
            if not iou_row.empty:
                ious_for_image_values[pred_idx] = iou_row['iou'].values[0]
                
        print_verbose(f"IoUs: {ious_for_image_values}", verbose)
        
        average_iou =  np.nanmean(ious_for_image_values)
        average_ious.append(average_iou)
        
        iou_gt_05 = np.sum(ious_for_image_values > 0.5)
        iou_lt_or_nan = np.sum((ious_for_image_values <= 0.5) | np.isnan(ious_for_image_values))  # Count of IoUs <= 0.5 or NaN
        
        
        print(f"Average IoU (ignoring NaNs): {average_iou}")
        print(f"Count of IoUs > 0.5: {iou_gt_05}")
        print(f"Count of IoUs <= 0.5 or NaN: {iou_lt_or_nan}")
        print("mean Average IoU of this set:", np.mean(average_ious))
        
        if save_images:
        
            heatmap_filename = f"{base_filename}_heatmap.png"
            heatmap_path = os.path.join(vis_dir, heatmap_filename)
            print(f"Heatmap path: {heatmap_path}")
        else:
            heatmap_path = None
            
            
        default_vis_settings = dict(
             metric_name="IoU",
            normalize_metric=False,
            show_caption=False,
            show_bbox=False,
            show_mask=True,
            show_cbar=False,
            colormap='RdYlGn',
            cbar_position=[0.1, 0.1, 0.8, 0.05],
            label_color='black',
            show_pred_idx=False,
            unmatched_color=(1.0, 105/255.0, 180/255.0)
            
            )
    
        if vis_settings is not None:
            default_vis_settings.update(vis_settings)

            
        display_inst_heat(image, rois, masks, class_ids, ref_data.class_names, 
                           metric=ious_for_image_values, 
                           save_path = heatmap_path,
                           **default_vis_settings
                          )
                           
                         
