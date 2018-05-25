
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import math
import argparse

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# ## Env setup

# In[2]:




# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:


# What model to download.
MODEL_NAME = 'ball_inference_graph_32744'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'object_detection/' + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1

"""
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
"""

# ## Load a (frozen) Tensorflow model into memory.

# In[5]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[6]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[7]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# In[ ]:

def run_inference_for_single_image(image, graph):
    with graph.device('/device:GPU:0'):
        with graph.as_default():
            config = tf.ConfigProto(allow_soft_placement = True)
            with tf.Session(config=config) as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                                tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                            detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                            detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                                             feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


import cv2

def preprocess(image_np):
  hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

  lower_red = np.array([0, 130, 33])
  upper_red = np.array([19, 255, 255])

  lower_red2 = np.array([164, 111, 0])
  upper_red2 = np.array([180, 255, 255])

  mask = cv2.inRange(hsv, lower_red, upper_red)
  mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
  mask_tot = mask | mask2

  mask_tot = cv2.erode(mask_tot, None, iterations=3)
  mask_tot = cv2.dilate(mask_tot, None, iterations=3)

  image_np = cv2.bitwise_and(image_np, image_np, mask=mask_tot)

  return image_np


def calculate_3d_cord(bbox, image):
    """
    :param bbox: Bounding box, normalized coordinates [y0,x0,y1,x1]
    :return: dict{"d": distance,
                    "center_x" : .. ,
                    "center_y" : ..}
    """
    #Get angles
    ang_hoz = math.radians(59.68)
    ang_ver = math.radians(36)

    h, w, _ = image.shape
    y0 = bbox[0]*h
    x0 = bbox[1]*w
    y1 = bbox[2]*h
    x1 = bbox[3]*w
    #size_bbox = (np.abs(y0-y1) + np.abs(x0-x1)) / 2
    center_y = y0 + np.abs(y0-y1)/2 - h/2
    center_x = x0 + np.abs(x0-x1)/2 - w/2

    d = find_distance(image, x1, x0, y1, y0)


    # x_meter = x_pixel * meter_hoz/pixel
    center_x = center_x * (2 * d * math.tan(ang_hoz / 2)) / w
    center_y = center_y * (2 * d * math.tan(ang_ver / 2)) / h

    return {"d" : d, "center_x": center_x, "center_y" : center_y}


def find_distance(image, xmax, xmin, ymax, ymin):
    """
    Finds the distance to the object given by the cords of the bounding box
    """
    resy, resx, _ = image.shape

    r = ((xmax - xmin) / resx + (ymax - ymin) / resy) / 2

    return 0.1168 / r

def plot_points(points, image, roll, lw, rx, ry):
    """
    :param points: [x_cord [m], y_cord [m], distance [m]]
    :param image: image [h, w, 3]
    :param roll: size of rolling window (int)
    :return: None - Plots points
    """
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D

    #Get angels
    ang_hoz = math.radians(59.68)
    ang_ver = math.radians(36)

    #Get image dimensions
    h, w, _ = image.shape

    #Convert to numpy matrix
    points = np.array(points)
    x = points[:,0]
    y = -points[:,1]
    z = points[:,2]

    #Smooth depth
    z = pd.DataFrame(z)
    z = np.array(z.rolling(roll).mean()).flatten()

    y = pd.DataFrame(y)
    y = np.array(y.rolling(ry).mean()).flatten()

    x = pd.DataFrame(x)
    x = np.array(x.rolling(rx).mean()).flatten()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot(x, y, z, '-b', zdir = 'y', label='3D Estimate', linewidth=lw)

    z_min = np.min(z[~np.isnan(z)])
    z_max = np.max(z[~np.isnan(z)])
    z_range = z_max-z_min
    ax.set_ylim(0, z_max + 0.5*z_range)

    lim = float((2*z_max*math.tan(ang_hoz/2)))
    ax.set_xlim(-lim/2, lim/2)
    ax.set_zlim(-lim/2, lim/2)


    ax.set_xlabel('X axis [M]')
    ax.set_ylabel('Depth [M]')
    ax.set_zlabel('Y [M]')
    ax.view_init(30, 230)


    """ Code for finding optimal angel
    for angle in range(150, 280):
        print(angle)
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.01)
    """
    
    plt.show()

def detect_vid(video, fraction):
    cap = cv2.VideoCapture(video)
    count = 0
    points = []
    image_np_org = None

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        while(cap.isOpened()):
          ret, image_np = cap.read()
          if not ret:
              break
          image_np_org = np.copy(image_np)
          count += 1
          if ret and (count % int(1/fraction) == 0):

              image_np = preprocess(image_np)

              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np_expanded = np.expand_dims(image_np, axis=0)
              image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
              boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
              scores = detection_graph.get_tensor_by_name('detection_scores:0')
              classes = detection_graph.get_tensor_by_name('detection_classes:0')
              num_detections = detection_graph.get_tensor_by_name('num_detections:0')
              # Actual detection.
              (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
              # Visualization of the results of a detection.

              vis_util.visualize_boxes_and_labels_on_image_array(
                image_np_org,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=.3)

              if scores[0][0] > 0.15:
                  point = calculate_3d_cord(boxes[0][0], image_np_org)
                  points.append([point["center_x"], point["center_y"], point["d"]])
                  print(point)
                  if count % 4 == 0:
                      #pass
                      cv2.imwrite("detections/" + video[16:-4] + "_{}".format(count) + ".png",
                                  image_np_org)
              cv2.imshow('object detection', cv2.resize(image_np_org, (1200,800)))
              if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        plot_points(points, image_np_org, 12, 3, 2, 2)
        cap.release()
        cv2.destroyAllWindows()
        
    cap.release()
    cv2.destroyAllWindows()
       


def detect_vid2(video, fraction):
    cap = cv2.VideoCapture(video)
    count = 0
    ret, image = cap.read()
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        while(cap.isOpened()):
          ret, image = cap.read()
          image_org = np.copy(image)
          image = preprocess(image)
          count += 1
          if ret and (count % int(1/fraction) == 0):
              #Run inference
              output_dict = sess.run(tensor_dict,
                                                           feed_dict={image_tensor: np.expand_dims(image, 0)})

              # all outputs are float32 numpy arrays, so convert types as appropriate
              output_dict['num_detections'] = int(output_dict['num_detections'][0])
              output_dict['detection_classes'] = output_dict[
                      'detection_classes'][0].astype(np.uint8)
              output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
              output_dict['detection_scores'] = output_dict['detection_scores'][0]
              if 'detection_masks' in output_dict:
                  output_dict['detection_masks'] = output_dict['detection_masks'][0]

              box = []
              
              
              for i, cat in enumerate(output_dict['detection_classes']):
                  s = output_dict['detection_scores'][i]
                  c = category_index[cat]['name']
                  if s < 0.02:
                      print("fail")
                      break
                  
                  if c in ["Ball"]:
                      box.append(output_dict['detection_boxes'][i])
                      break
                  
              if len(box) > 0:    
                  vis_util.draw_bounding_box_on_image_array(image_org, box[0][0], box[0][1],
                                                            box[0][2], box[0][3])


              cv2.imshow('object detection', cv2.resize(image_org, (1200,800)))
              if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        cap.release()
        cv2.destroyAllWindows()
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Video Detection')
  parser.add_argument('--video',  help='path to video file', default = "vid_data/videos/vid2.mov")
  parser.add_argument('--fraction', default=1, 
                      help='fraction of images processed')

  args = parser.parse_args()
  detect_vid(args.video, float(args.fraction))

