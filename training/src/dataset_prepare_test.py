# -*- coding: utf-8 -*-
# @Time    : 18-3-6 3:20 PM
# @Author  : edvard_hua@live.com
# @FileName: data_prepare.py
# @Software: PyCharm

import numpy as np
import cv2
import struct
import math


class CocoPose:
    @staticmethod
    def get_bgimg(inp, target_size=None):
        inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
        return inp

    @staticmethod
    def display_image(inp, heatmap=None, pred_heat=None, as_numpy=False):
        global mplset
        mplset = True
        import matplotlib.pyplot as plt

        fig = plt.figure()
        if heatmap is not None:
            a = fig.add_subplot(1, 2, 1)
            a.set_title('True_Heatmap')
            plt.imshow(CocoPose.get_bgimg(inp, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
            tmp = np.amax(heatmap, axis=2)
            plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.7)
            plt.colorbar()
        else:
            a = fig.add_subplot(1, 2, 1)
            a.set_title('Image')
            plt.imshow(CocoPose.get_bgimg(inp))

        if pred_heat is not None:
            a = fig.add_subplot(1, 2, 2)
            a.set_title('Pred_Heatmap')
            plt.imshow(CocoPose.get_bgimg(inp, target_size=(pred_heat.shape[1], pred_heat.shape[0])), alpha=0.5)
            tmp = np.amax(pred_heat, axis=2)
            plt.imshow(tmp, cmap=plt.cm.gray, alpha=1)
            plt.colorbar()

        if not as_numpy:
            plt.show()
        else:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            fig.clear()
            plt.close()
            return data

    @staticmethod
    def swap(key_points):
        """
            args:
                key_points: a dict contains keypoints
        """
        SWAP_PAIRS = [[2,5],[3,6],[8,11],[9,12]]
        for pair in SWAP_PAIRS:
            pair_left, pair_right = pair
            if key_points[pair_left] is not None and key_points[pair_right] is not None:
                if key_points[pair_left][0] > key_points[pair_right][0]:
                    tmp = key_points[pair_left]
                    key_points[pair_left] = key_points[pair_right]
                    key_points[pair_right] = tmp
    
    @staticmethod
    def draw_img(img, key_points, gesture):
        """
            args:
                img: img to draw
                key_points: a dict contains keypoints
        """
        PAIRS = [[0,1], [1,2],[2,3], [3,4],
                [0,5], [5,6],[6,7], [7,8],
                [0,9], [9,10],[10,11],[11, 12],
                [0,13], [13,14],[14,15],[15,16],
                [0,17], [17, 18], [18,19], [19,20]
                 ]
        kk = 0
        finger_colors = [
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (255, 0, 255),
        ]
        p3 = (100,100)
        if None in key_points:
            return
        if gesture=="Unknown":
            return
        cv2.putText(img, gesture, p3, cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        for pair in PAIRS:
            pair_a, pair_b = pair
            #cv2.putText(img, str(kk),(key_points[kk][0], key_points[kk][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
            kk+=1
            color = finger_colors[int((kk-1)/4)]
            # print(pair)
            if key_points[pair_a] is not None and key_points[pair_b] is not None:
                point_a = (int(key_points[pair_a][0]), int(key_points[pair_a][1]))
                point_b = (int(key_points[pair_b][0]), int(key_points[pair_b][1]))
                #print(point_a, point_b)
                cv2.line(img, point_a, point_b, color, 1, lineType=cv2.LINE_AA)
                cv2.circle(img, point_a, 2, color, -1)
                cv2.circle(img, point_b, 2, color, -1)

    @staticmethod
    def get_keypoints(inp, heatmap=None, threshold=0.5, filter=False, swap=False):
        from scipy.ndimage.filters import gaussian_filter
        key_points = {}
        if heatmap is not None:
            image_h, image_w, _ = inp.shape
            heatmap_h, heatmap_w, heatmap_c = heatmap.shape
            x_ratio, y_ratio = image_w/heatmap_w, image_h/heatmap_h
            
            for c in range(heatmap_c):
                heatmap_tmp = heatmap[:,:,c]
                if filter:
                    heatmap_tmp = gaussian_filter(heatmap_tmp, sigma=5)
                ind = np.unravel_index(np.argmax(heatmap_tmp), heatmap_tmp.shape)
                print(ind)
                if heatmap_tmp[ind[0],ind[1]] < threshold:
                    print(heatmap_tmp[ind[0],ind[1]])
                    key_points[c] = None
                else:
                    print(heatmap_tmp[ind[0],ind[1]])
                    coord_x = int(ind[1]*x_ratio)
                    coord_y = int(ind[0]*y_ratio)
                    key_points[c] = [coord_x, coord_y]
        if swap: CocoPose.swap(key_points)
        return key_points
    @staticmethod
    def distance(a, b):
        dist = ((a[0]-b[0])**2+(a[1]-b[1])**2)
        return dist
    @staticmethod
    def isStraight(a,b,c):
        dist_ac = CocoPose.distance(a,c)
        dist_bc = CocoPose.distance(b,c)
        if dist_ac>dist_bc:
            return 1
        return 0


    @staticmethod
    def hand_gesture(key_points):
        gestures = ['One', 'Two', "Three", "Four", "Five","Six","Fist", "Yeah", "Ok","Great","Unknown"]
        #index = 0
        fingers = []
        gestures_eq = [[0,1,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[0,1,1,1,1],[1,1,1,1,1],[1,0,0,0,1],[0,0,0,0,0],[0,1,1,0,0],[0,0,1,1,1], [1,0,0,0,0]]
        if key_points[0] is not None and key_points[4] is not None and key_points[17] is not None:
            dist1 = CocoPose.distance(key_points[17], key_points[4])
            dist2 = CocoPose.distance(key_points[0], key_points[17])
            ratio = dist1/dist2
            
            if ratio<0.7:
                fingers.append(0)
            else:
                fingers.append(1)

        for i in range(1,5):
            if key_points[(i+1)*4] is not None and key_points[(i+1)*4-1] is not None and key_points[0] is not None:
                temp = CocoPose.isStraight(key_points[(i+1)*4], key_points[(i+1)*4-1], key_points[0])
                fingers.append(temp)
        if len(fingers)<5:
            #print("None")
            return gestures[10]
        print("Fingers: ", fingers)
        print("Big Finger: ",ratio)
        for index, gesture in enumerate(gestures_eq):
            if gesture == fingers:
                print("gesture is: ", gestures[index])
                return gestures[index]
        print("gesture is: ",gestures[9])
        return gestures[10]
    @staticmethod
    def display_image_video(inp, bbox, heatmap=None, threshold=0.5, filter=False, swap=False):
        import copy
        from scipy.ndimage.filters import gaussian_filter
        if heatmap is not None:
            #image_h, image_w, _ = inp.shape
            heatmap_h, heatmap_w, heatmap_c = heatmap.shape
            #x_ratio, y_ratio = image_w/heatmap_h, image_h/heatmap_w
            #print("image: ", image_h, "hetamap: ", heatmap_h)
            key_points = {}
            for c in range(heatmap_c):
                heatmap_tmp = heatmap[:,:,c]
                if filter:
                    heatmap_tmp = gaussian_filter(heatmap_tmp, sigma=6)
                ind = np.unravel_index(np.argmax(heatmap_tmp), heatmap_tmp.shape)
                #print(ind)
                if heatmap_tmp[ind[0],ind[1]] < threshold:
                    #print(heatmap_tmp[ind[0],ind[1]])
                    key_points[c] = None
                else:
                    #print(heatmap_tmp[ind[0],ind[1]])
                    coord_x = int(ind[1]/heatmap_w*(bbox[2]-bbox[0]) + bbox[0])
                    coord_y = int(ind[0]/heatmap_h*(bbox[3]-bbox[1]) + bbox[1])
                    key_points[c] = [coord_x, coord_y]
            if swap: CocoPose.swap(key_points)
            img_draw = copy.deepcopy(inp)
            gesture = CocoPose.hand_gesture(key_points)
            CocoPose.draw_img(img_draw, key_points, gesture)

            return img_draw

class CocoMetadata:
    __coco_parts = 21

    @staticmethod
    def parse_float(four_np):
        assert len(four_np) == 4
        return struct.unpack('<f', bytes(four_np))[0]

    @staticmethod
    def parse_floats(four_nps, adjust=0):
        assert len(four_nps) % 4 == 0
        return [(CocoMetadata.parse_float(four_nps[x * 4:x * 4 + 4]) + adjust) for x in range(len(four_nps) // 4)]

    def __init__(self, idx, img_path, img_meta, annotations, sigma):
        self.idx = idx
        self.img = self.read_image(img_path)
        self.sigma = sigma

        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])

        joint_list = []
        for ann in annotations:
            if ann.get('num_keypoints', 0) == 0:
                continue

            kp = np.array(ann['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]

            joint_list.append([(x, y) if v >= 1 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)])

        self.joint_list = []
        transform = list(zip(
            [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13,16, 18, 20, 15, 17, 19, 21],
            [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13,16, 18, 20, 15, 17, 19, 21]
        ))
        for prev_joint in joint_list:
            new_joint = []
            for idx1, idx2 in transform:
                j1 = prev_joint[idx1 - 1]
                j2 = prev_joint[idx2 - 1]

                if j1[0] <= 0 or j1[1] <= 0 or j2[0] <= 0 or j2[1] <= 0:
                    new_joint.append((-1000, -1000))
                else:
                    new_joint.append(((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2))
            # background
            # new_joint.append((-1000, -1000))
            self.joint_list.append(new_joint)

    def get_heatmap(self, target_size):
        heatmap = np.zeros((CocoMetadata.__coco_parts, self.height, self.width), dtype=np.float32)
        #print(heatmap.shape)
        for joints in self.joint_list:
            for idx, point in enumerate(joints):
                if point[0] < 0 or point[1] < 0:
                    continue
                CocoMetadata.put_heatmap(heatmap, idx, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        # heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        if target_size:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)
        #print(heatmap.shape)
        return heatmap.astype(np.float16)

    @staticmethod
    def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        
        th = 1.6052
        """
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        # gaussian filter
        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)
        """
        sigma2 = sigma*sigma*2
        h = np.arange(0, height, 1)
        w = np.arange(0, width, 1)
        ww, hh = np.meshgrid(w, h)
        ww1 = ww - center_x
        hh1 = hh - center_y
        dis2 = (ww1**2+hh1**2)/sigma2 
        heatmap_tmp = np.exp(-dis2)
        heatmap[plane_idx] = np.maximum(heatmap[plane_idx],heatmap_tmp)
	
    def read_image(self, img_path):
        img_path = img_path.replace('\\\\','/')
        img_str = open(img_path, "rb").read()
        if not img_str:
            print("image not read, path=%s" % img_path)
        nparr = np.fromstring(img_str, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
