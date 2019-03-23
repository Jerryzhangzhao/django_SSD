from django.http import HttpResponse
from django.shortcuts import render_to_response
import base64
import requests
import imagenet_id_to_name
import time
import shelve
import os

import math
import random
import urllib.request
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import namedtuple

from . import np_methods
from . import visualization
from . import SSD_utils

def write_request_meta_to_txt(request_info):
    txt_name = './log/log_request_META.txt'
    with open (txt_name,'a+') as f:
        f.write(str(time.asctime())+'\n')
        f.write(request_info+'\n\n')

# write access statistic info to log file
def write_to_txt(db_name,txt_name):
    rec = shelve.open(db_name)
    info_str = time.asctime()+'  search-form-count:'+str(rec['search-form-count'])+'  search-result-count:'+str(rec['search-result-count'])
    with open(txt_name,'a+') as f:
        f.write(info_str+'\n')

# record log info
def record_log(record_type):
    # initial
    if not (os.path.exists ('./log/record_log.db') or os.path.exists('./log/record_log.db.dat')):
        rec = shelve.open("./log/record_log.db")
        rec['search-form-count'] = 1
        rec['search-result-count'] = 1
        rec['last_log_time'] = time.time()
        rec.close()

    if record_type == "search-form":
        rec = shelve.open("./log/record_log.db")
        search_form_cuount = rec['search-form-count']
        search_form_cuount += 1
        rec['search-form-count'] = search_form_cuount

        if time.time() - rec['last_log_time'] > 3600*2:
            rec['last_log_time'] = time.time()
            write_to_txt("./log/record_log.db","./log/log_access.txt")
        rec.close()

    if record_type == "search-result":
        rec = shelve.open("./log/record_log.db")
        search_result_cuount = rec['search-result-count']
        search_result_cuount += 1
        rec['search-result-count'] = search_result_cuount

        if time.time() - rec['last_log_time'] > 3600*2:
            rec['last_log_time'] = time.time()
            write_to_txt("./log/record_log.db","./log/log_access.txt")
        rec.close()

def request_classification(image_url):
    SERVER_URL = 'http://47.101.197.166:8501/v1/models/resnet:predict'
    IMAGE_URL = image_url
    # Download the image
    print("download image...")
    try:
        dl_request = requests.get(IMAGE_URL, stream=True)
        dl_request.raise_for_status()    

        # Compose a JSON Predict request (send JPEG image in base64).
        print("Compose a JSON Predict request")
        predict_request = '{"instances" : [{"b64": "%s"}]}' % base64.b64encode(dl_request.content).decode('ascii')

        # Send few requests to warm-up the model.
        #print("warm-up the model")
        #for _ in range(3):
            #response = requests.post(SERVER_URL, data=predict_request)
            #response.raise_for_status()

        # Send few actual requests and report average latency.
        print("send requests")
        total_time = 0
        num_requests = 1
        for _ in range(num_requests):
            response = requests.post(SERVER_URL, data=predict_request)
            response.raise_for_status()
            total_time += response.elapsed.total_seconds()
            prediction = response.json()['predictions'][0]

        print('Prediction class: {},class name: {}, avg latency: {} ms'.format( prediction['classes'],imagenet_id_to_name.imagenet_name_dict[prediction['classes']], (total_time*1000)/num_requests))
    
        return imagenet_id_to_name.imagenet_name_dict[prediction['classes']] 
    except:
        return 'something wrong happend'

# start page
def search_form(request):
    record_log('search-form')
    write_request_meta_to_txt(str(request.META))
    return render_to_response('search_form.html')


# result page
def search(request):
    request.encoding = 'utf-8'
    print(request.GET)

    record_log('search-result')
    write_request_meta_to_txt(str(request.META))
   
    # gRPC config 
    serving_config = {
        "hostport": "47.101.197.166:9000",
        "max_message_length": 10 * 1024 * 1024,
        "timeout": 30000,
        "signature_name": "serving_default", #
        "model_name": "SSD"
    }

    if len(request.GET)>0:
        message = request.GET['info']+'\n\n'+ request_classification(request.GET['info'])
        #return HttpResponse(message)
        image_url = request.GET['info']
        
        #load image from URL
        print("loading image")
        IMAGE_URL=image_url  #"http://img0.ph.126.net/YytwzUO2IPN3jHMu4r6wiw==/6597996654961386816.jpg"
        dl_request = requests.get(IMAGE_URL, stream=True)
        resp = urllib.request.urlopen(IMAGE_URL)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #load image from local path
        #image_path = os.path.join(os.getcwd(),"images/")
        #image_names = sorted(os.listdir(image_path))
        #print(image_names)
        #img = mpimg.imread(image_path + image_names[1])

        predict_result = SSD_utils.predict_test(1, serving_config,img)
        #print("reshape result")

    
        predictions=[]
        localisations=[]
        # get result 
        predictions.append(SSD_utils.parse_result_value(predict_result,'predictions0'))
        predictions.append(SSD_utils.parse_result_value(predict_result,'predictions1'))
        predictions.append(SSD_utils.parse_result_value(predict_result,'predictions2'))
        predictions.append(SSD_utils.parse_result_value(predict_result,'predictions3'))
        predictions.append(SSD_utils.parse_result_value(predict_result,'predictions4'))
        predictions.append(SSD_utils.parse_result_value(predict_result,'predictions5'))
        localisations.append(SSD_utils.parse_result_value(predict_result,'localisations0'))
        localisations.append(SSD_utils.parse_result_value(predict_result,'localisations1'))
        localisations.append(SSD_utils.parse_result_value(predict_result,'localisations2'))
        localisations.append(SSD_utils.parse_result_value(predict_result,'localisations3'))
        localisations.append(SSD_utils.parse_result_value(predict_result,'localisations4'))
        localisations.append(SSD_utils.parse_result_value(predict_result,'localisations5'))
        bbox_img_0 = SSD_utils.parse_result_value(predict_result,'bbox_img')

        rclasses, rscores, rbboxes = SSD_utils.process_image_1(predictions,localisations,bbox_img_0)
        class_name =  visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        
        predict_name = class_name  # request_classification(request.GET['info'])
        image_url='result_image/result.png'
        return render_to_response('search_result.html',{'image_url':image_url,'predict_name':predict_name})
    else:
        return render_to_response('search_result.html',{'image_url':'incorrect','predict_name':'Please input right image URL'})

# traffic page
def search_traffic(request):
    rec = shelve.open("./log/record_log.db")
    search_form_visit_count = str(rec['search-form-count'])
    search_result_visit_count = str(rec['search-result-count'])
    rec.close()

    with open('./log/log_access.txt') as f:
        #search_detail_info = f.read()
        lines = f.readlines()
    search_detail_info = ""
    for line in lines:
        search_detail_info += line +" ---- " 

    return render_to_response('search_traffic.html',{'search_form_visit_count':search_form_visit_count,'search_result_visit_count':search_result_visit_count,'search_detail_info':search_detail_info})

if __name__ == '__main__':
    record("search-form")

