# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import os, copy
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from training.src.dataset_prepare_test import CocoPose
from training.src.handdetector import det, postprocess

import sys
caffe_root = '/home/inomjon/Projects/MobileNet-YOLO/python'  
sys.path.insert(0, caffe_root)
import caffe
import time
def expand_bounding_rect(bbox, image_dim, resize_dim):
    """
    :param bbox: [x, y, w, h]
    :param image_dim: [H, W]
    :param resize_dim: [H_r, W_r]
    :return: N x 4, [x, y, w, h]
    """
    place_ratio = 0.5
    bbox[3] = bbox[3]-bbox[1]
    bbox[2] = bbox[2]-bbox[0]
    bbox_expand = bbox
    
    if resize_dim[0] / bbox[3] > resize_dim[1] / bbox[2]:  # keep width
        bbox_expand[3] = resize_dim[0] * bbox[2] / resize_dim[1]
        bbox_expand[1] = max(min(bbox[1] - (bbox_expand[3] - bbox[3]) * place_ratio,
                                 image_dim[0] - bbox_expand[3]), 0.0)
    else:  # keep height
        bbox_expand[2] = resize_dim[1] * bbox[3] / resize_dim[0]
        bbox_expand[0] = max(min(bbox[0] - (bbox_expand[2] - bbox[2]) * place_ratio,
                                 image_dim[1] - bbox_expand[2]), 0.0)
    bbox_expand[3] = bbox_expand[3]+bbox_expand[1]
    bbox_expand[2] = bbox_expand[2]+bbox_expand[0]
    return bbox_expand
def padding(xmin, ymin, xmax, ymax, w, h, pad=20):
    xmin = xmin - pad
    ymin = ymin - pad
    xmax = xmax + pad
    ymax = ymax + pad
    if xmin<0:
        xmin = 0
    if ymin<0:
        ymin = 0

    if xmax>w:
        xmax = w
    if ymin>h:
        ymax = h

    return xmin, ymin, xmax, ymax

def run_with_frozen_pb_video(input_w_h, frozen_graph, output_node_names, net, test_img_path=None):

    with tf.gfile.GFile(frozen_graph, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
    )
    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name("image:0")
    output = graph.get_tensor_by_name("%s:0" % output_node_names)
    with tf.Session() as sess:
        if test_img_path is None:
            cap = cv2.VideoCapture(-1)
            # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = str(time.time())+'.mp4'
            name = "result.mp4"
            #out = cv2.VideoWriter(name,cv2.VideoWriter_fourcc("m", 'p', '4',"v"), 20, (352, 352))
            out = cv2.VideoWriter(name,cv2.VideoWriter_fourcc('M','J','P','G'), 20, (352, 352))
            while True:
                ret, frame = cap.read()
                if ret == False:
                    break
                frame = cv2.resize(frame, (352,352))
                img_w, img_h = frame.shape[0], frame.shape[1]
                result = det(frame,net)
                box, conf, cls = postprocess(frame, result)
                for i in range(len(box)):
                    # if int(cls[i])!=1:
                    #     continue
                    if conf[i]<0.4:
                        continue
                    xmin, ymin, xmax, ymax = padding(box[i][0],box[i][1],box[i][2],box[i][3],img_w,img_h)
                    pt = expand_bounding_rect([xmin, ymin, xmax, ymax], [352,352], [256, 256])
                    p1 = (xmin, ymin)
                    p2 = (xmax, ymax)
                    cv2.rectangle(frame, p1, p2, (0,255,0), 3)
                    image_ = frame[int(pt[1]):int(pt[3]), int(pt[0]):int(pt[2])]
                    image_ = cv2.resize(image_, (256, 256))
                    image_ = cv2.resize(image_, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)
                    start = time.time()
                    
                    heatmaps = sess.run(output, feed_dict={image: [image_]})
                    #print("inference time: ",time.time()-start)
                    frame = CocoPose.display_image_video(
                    frame,
                    pt,
                    heatmaps[0,:,:,:],
                    threshold=0.1,
                    filter=True,
                    swap=False
                    )
                    out.write(frame)
                    
                   # key1 = cv2.waitKey(0)
                    # cv2.imshow('hand', frame)
                    # if key1 == ord('w'):
                    #     hand = False
                    #     continue
                    #
                
                cv2.imshow('res', frame)
                
                
                key = cv2.waitKey(1) & 0xFF
                if key==ord('q'):
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    #out.release()
                    break
        else:
            image_0 = cv2.imread(test_img_path)
            image_ = cv2.resize(image_0, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)
            heatmaps = sess.run(output, feed_dict={image: [image_]})
            draw_img = CocoPose.display_image_video(
                image_0,
                heatmaps[0,:,:,:],
                threshold=0.1,
                filter=False,
                swap=False
            ) 
            cv2.imshow('res', draw_img)
            key = cv2.waitKey(0)
            if key==ord('q'):
                cv2.destroyAllWindows()

if __name__ == '__main__':


    caffe.set_mode_gpu()
    model_weights = 'yolo_hand/yolov3_lite_deploy_iter_164000.caffemodel'
    model_def = 'yolo_hand/yolov3_lite_deploy.prototxt'
    
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    run_with_frozen_pb_video(
        # "CPM/stage_1_out",
        192,
        "models/model-500000.pb",
        "Convolutional_Pose_Machine/stage_5_out",
        net
        #'test.jpg'
    )
    
