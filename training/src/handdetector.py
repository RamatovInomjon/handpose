# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import sys
import cv2
import argparse

import sys
caffe_root = '/home/inomjon/Projects/MobileNet-YOLO/python'  
sys.path.insert(0, caffe_root)
import os
import caffe
import math
from os import walk
from os.path import join
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

CLASSES = ('__background__',
           'hand', 'face')


def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def det(image,net):
    
    
    
    image = image - 127.5 #*
    image = image*0.007843
    image = np.asarray(image,np.float32)
    image = image.transpose(2,0,1) #3 channels
    net.blobs['data'].data[...] = image
    output = net.forward()

    return output


def main(args):    
 
    caffe.set_mode_gpu()
    model_def = args.model_def
    model_weights = args.model_weights
    
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame,(416, 416))
        result = det(frame,net)
        box, conf, cls = postprocess(frame, result)
        for i in range(len(box)):
            if conf[i]<0.6:
                continue
            if int(cls[i])!=1:
                continue
            p1 = (box[i][0], box[i][1])
            p2 = (box[i][2], box[i][3])
            cv2.rectangle(frame, p1, p2, (0,255,0), 3)
            p3 = (max(p1[0], 15), max(p1[1], 15))
            title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
            cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
        if ret==False:
            cap.release()
            cv2.destroyAllWindows()
            out.release()
            break

        cv2.imshow("Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
def parse_args():
    parser = argparse.ArgumentParser()
    '''parse args'''
    parser.add_argument('--model_def', '-md', default='yolo_hand/deploy.prototxt')
    parser.add_argument('--model_weights', '-mw', default='yolo_hand/handface.caffemodel')
    parser.add_argument('--video', '-v', default=0)
    return parser.parse_args()
    
if __name__ == '__main__':
    main(parse_args())
