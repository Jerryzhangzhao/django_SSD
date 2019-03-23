# coding=UTF-8
# This Python file uses the following encoding: utf-8

import os
import math
import random
import grpc
import requests
import urllib.request
import numpy as np
import cv2
from tensorflow_serving.apis import model_service_pb2_grpc, model_management_pb2, get_model_status_pb2, predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.config import model_server_config_pb2
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.core.framework import types_pb2


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import namedtuple

import np_methods
import visualization
import SSD_utils

# 设置SSD的参数
SSDParams1 = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])

default_params1 = SSDParams(
        img_shape=(300, 300),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

print("++++++++++++++++++++++++++++++++++")


# Main image processing routine.
def process_image_1(rpredictions,rlocalisations,rbbox_img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def anchors1(img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      default_params.feat_shapes,
                                      default_params.anchor_sizes,
                                      default_params.anchor_ratios,
                                      default_params.anchor_steps,
                                      default_params.anchor_offset,
                                      dtype)

def ssd_anchor_one_layer1(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w

def ssd_anchors_all_layers1(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

def predict_test1(batch_size, serving_config,img):
    channel = grpc.insecure_channel(serving_config['hostport'], options=[('grpc.max_send_message_length', serving_config['max_message_length']), (
        'grpc.max_receive_message_length', serving_config['max_message_length'])])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    #img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #cv2.imshow("Image", image)

    # Test on some demo image and visualize output.
    #image_path = os.path.join(os.getcwd(),"images/")
    #image_names = sorted(os.listdir(image_path))
    #print(image_names)
    #img = mpimg.imread(image_path + image_names[1])

    # Creating random images for given batch size 
    #input_data=np.ones((500,500,3),dtype="uint8")
    input_data=img
    print(img.shape)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = serving_config['model_name']
    request.model_spec.signature_name = serving_config['signature_name']
    request.inputs['input0'].CopyFrom(make_tensor_proto(
        input_data, shape=img.shape, dtype=types_pb2.DT_UINT8))
    result = stub.Predict(request, serving_config['timeout'])
    channel.close()    
    return result

def parse_result_value(predict_result,name):
    raw_value = predict_result.outputs[name].float_val
    dims = predict_result.outputs[name].tensor_shape.dim
    shape=[]
    for dim in dims:
        shape.append(dim.size)
    #print(shape)
    reshaped_value = np.reshape(raw_value,shape)
    return reshaped_value


#net_shape = (300, 300)
#ssd_anchors = anchors(net_shape)

if __name__ == "__main__":
    net_shape = (300, 300)
    ssd_anchors = SSD_utils.get_ssd_anchor(net_shape)
    serving_config = {
        "hostport": "47.101.197.166:9000",
        "max_message_length": 10 * 1024 * 1024,
        "timeout": 30000,
        "signature_name": "serving_default", #
        "model_name": "SSD"
    }

    #load image from URL
    print("loading image")
    IMAGE_URL="http://img0.ph.126.net/YytwzUO2IPN3jHMu4r6wiw==/6597996654961386816.jpg"
    dl_request = requests.get(IMAGE_URL, stream=True)
    resp = urllib.request.urlopen(IMAGE_URL)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    #load image from local path
    #image_path = os.path.join(os.getcwd(),"images/")
    #image_names = sorted(os.listdir(image_path))
    #print(image_names)
    #img = mpimg.imread(image_path + image_names[1])

    predict_result = predict_test(1, serving_config,img)
    print("--------------------------------")

    
    predictions=[]
    localisations=[]
    # get result 
    predictions.append(parse_result_value(predict_result,'predictions0'))
    predictions.append(parse_result_value(predict_result,'predictions1'))
    predictions.append(parse_result_value(predict_result,'predictions2'))
    predictions.append(parse_result_value(predict_result,'predictions3'))
    predictions.append(parse_result_value(predict_result,'predictions4'))
    predictions.append(parse_result_value(predict_result,'predictions5'))
    localisations.append(parse_result_value(predict_result,'localisations0'))
    localisations.append(parse_result_value(predict_result,'localisations1'))
    localisations.append(parse_result_value(predict_result,'localisations2'))
    localisations.append(parse_result_value(predict_result,'localisations3'))
    localisations.append(parse_result_value(predict_result,'localisations4'))
    localisations.append(parse_result_value(predict_result,'localisations5'))
    bbox_img_0 = parse_result_value(predict_result,'bbox_img')

    rclasses, rscores, rbboxes = process_image_1(predictions,localisations,bbox_img_0)
    visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

    #print(predict_result.outputs['predictions0'].float_val)

#curl -d '{"instances": [[[1,1,1]]]}' -X POST http://localhost:8501/v1/models/testnet:predict


