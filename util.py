import torch.nn as nn
import cv2
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function as F
from parameters import Parameters
import math

p = Parameters()

def cross_entropy2d(inputs, target, weight=None, size_average=True):
    loss = torch.nn.CrossEntropyLoss()

    n, c, h, w = inputs.size()
    prediction = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    gt =target.transpose(1, 2).transpose(2, 3).contiguous().view(-1)

    return loss(prediction, gt)

###############################################################
##
## visualize
## 
###############################################################

def visualize_points(image, x, y):
    image = image
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    for k in range(len(y)):
        for i, j in zip(x[k], y[k]):
            if i > 0:
                image = cv2.circle(image, (int(i), int(j)), 2, p.color[1], -1)

    cv2.imshow("test2", image)
    cv2.waitKey(0)  

def visualize_points_origin_size(x, y, test_image, ratio_w, ratio_h):
    color = 0
    image = deepcopy(test_image)
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    image = cv2.resize(image, (int(p.x_size/ratio_w), int(p.y_size/ratio_h)))

    for i, j in zip(x, y):
        color += 1
        for index in range(len(i)):
            cv2.circle(image, (int(i[index]), int(j[index])), 10, p.color[color], -1)
    cv2.imshow("test2", image)
    cv2.waitKey(0)  

    return test_image

def visualize_gt(self, gt_point, gt_instance, ground_angle, image):
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    for y in range(self.p.grid_y):
        for x in range(self.p.grid_x):
            if gt_point[0][y][x] > 0:
                xx = int(gt_point[1][y][x]*self.p.resize_ratio+self.p.resize_ratio*x)
                yy = int(gt_point[2][y][x]*self.p.resize_ratio+self.p.resize_ratio*y)
                image = cv2.circle(image, (xx, yy), 10, self.p.color[1], -1)

    cv2.imshow("image", image)
    cv2.waitKey(0)

def visualize_regression(image, gt):
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    for i in gt:
        for j in range(p.regression_size):#gt
            y_value = p.y_size - (p.regression_size-j)*(220/p.regression_size)
            if i[j] >0:
                x_value = int(i[j]*p.x_size)
                image = cv2.circle(image, (x_value, y_value), 5, p.color[1], -1)
    cv2.imshow("image", image)
    cv2.waitKey(0)   

def draw_points(x, y, image):
    color_index = 0
    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            image = cv2.circle(image, (int(i[index]), int(j[index])), 5, p.color[color_index], -1)
            if index == 0: continue
            image = cv2.line(image, (int(i[index]), int(j[index])), (int(i[index-1]), int(j[index-1])), p.color[color_index], 2)

    return image

def get_egolane_from_points(x, y, image, intersector_mode=False):
    color_index = 3
    egoLeft_i = []
    egoLeft_j = []
    egoRight_i = []
    egoRight_j = []
    WIDTH = len(image[0])
    HEIGHT = len(image)
    laneImage = np.zeros_like(image)
    for i, j in zip(x, y):
        # egoLeft_i is defined by having x value less than middle of image while having maximum y value (toward the image bottom)
        # egoRight_i is defined by having x value less than middle of image while having maximum y value (toward the image bottom)
        if i[0] < WIDTH/2: # left bound candidate
            if len(egoLeft_j) <= 0 or egoLeft_j[0] < j[0]:
                egoLeft_i = i
                egoLeft_j = j
        elif i[0] >= WIDTH/2: # right bound candidate
            if len(egoRight_j) <= 0 or egoRight_j[0] < j[0]:
                egoRight_i = i
                egoRight_j = j
    if len(egoLeft_i) <= 0 and len(egoRight_i) <= 0:
        print("ego-lane not found for this image! assuming center trapizoidal region.")
        egoLeft_i = [0, WIDTH/2-16]
        egoLeft_j = [HEIGHT-1, HEIGHT/2]
        egoRight_i = [WIDTH-1, WIDTH/2+15]
        egoRight_j = [HEIGHT-1, HEIGHT/2]
    elif len(egoLeft_i) > 0 and len(egoRight_i) <= 0:
        egoRight_i = [WIDTH-egoLeft_i[index] for index in range(len(egoLeft_i))]
        egoRight_j = egoLeft_j
    elif len(egoLeft_i) <= 0 and len(egoRight_i) > 0:
        egoLeft_i = [WIDTH-egoRight_i[index] for index in range(len(egoRight_i))]
        egoLeft_j = egoRight_j

    # Shift the spline points so that it considers lane-changing vehicles
    egoLeft_i = [egoLeft_i[index]-50*(1 - index/len(egoLeft_i)) for index in range(len(egoLeft_i))]
    egoRight_i = [egoRight_i[index]+50*(1 - index/len(egoRight_i)) for index in range(len(egoRight_i))]

    # Having intersector_mode as True means that the lanemap will
    # directly be used to intersect with input image to reduce search region.
    # In this case, since the top part of the ego-lane cars may get cropped-off by the region,
    # we extend the upper half of lane region all the way up to top.
    if intersector_mode:
        egoLeft_j = [egoLeft_j[index] if egoLeft_j[index] > HEIGHT/2 else 0 for index in range(len(egoLeft_i))]
        egoRight_j = [egoRight_j[index] if egoRight_j[index] > HEIGHT/2 else 0 for index in range(len(egoRight_i))]

    #contour_list = [(int(egoLeft_i[0]), HEIGHT-1)] # extend the bottommost contour point towards the bottom-side of image
    contour_list = [(0, HEIGHT-1)] # extend the bottommost contour point towards the bottom-left-side of image
    contour_list += [(int(egoLeft_i[index]), int(egoLeft_j[index])) for index in range(len(egoLeft_i))]
    contour_list += [(int(egoRight_i[index]), int(egoRight_j[index])) for index in reversed(range(len(egoRight_i)))]
    #contour_list += [(int(egoRight_i[0]), HEIGHT-1)] # extend the bottommost contour point towards the bottom-side of image
    contour_list += [(WIDTH-1, HEIGHT-1)] # extend the bottommost contour point towards the bottom-side of image
    cv2.fillConvexPoly(laneImage, points=np.array(contour_list, np.int32), color=p.color[color_index])

    return laneImage

###############################################################
##
## calculate
## 
###############################################################
def convert_to_original_size(x, y, ratio_w, ratio_h):
    # convert results to original size
    out_x = []
    out_y = []

    for i, j in zip(x,y):
        out_x.append((np.array(i)/ratio_w).tolist())
        out_y.append((np.array(j)/ratio_h).tolist())

    return out_x, out_y

def get_closest_point_along_angle(x, y, point, angle):
    index = 0
    for i, j in zip(x, y): 
        a = get_angle_two_points(point, (i,j))
        if abs(a-angle) < 0.1:
            return (i, j), index
        index += 1
    return (-1, -1), -1


def get_num_along_point(x, y, point1, point2, image=None): # point1 : source
    x = np.array(x)
    y = np.array(y)

    x = x[y<point1[1]]
    y = y[y<point1[1]]

    dis = np.sqrt( (x - point1[0])**2 + (y - point1[1])**2 )

    count = 0
    shortest = 1000
    target_angle = get_angle_two_points(point1, point2)
    for i in range(len(dis)):
        angle = get_angle_two_points(point1, (x[i], y[i]))
        diff_angle = abs(angle-target_angle)
        distance = dis[i] * math.sin( diff_angle*math.pi*2 )
        if distance <= 12:
            count += 1
            if distance < shortest:
                shortest = distance

    return count, shortest

def get_closest_upper_point(x, y, point, n):
    x = np.array(x)
    y = np.array(y)

    x = x[y<point[1]]
    y = y[y<point[1]]

    dis = (x - point[0])**2 + (y - point[1])**2

    ind = np.argsort(dis, axis=0)
    x = np.take_along_axis(x, ind, axis=0).tolist()
    y = np.take_along_axis(y, ind, axis=0).tolist()

    points = []
    for i, j in zip(x[:n], y[:n]):
        points.append((i,j))

    return points

def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

def sort_along_x(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(i, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

def sort_batch_along_y(target_lanes, target_h):
    out_x = []
    out_y = []

    for x_batch, y_batch in zip(target_lanes, target_h):
        temp_x = []
        temp_y = []
        for x, y, in zip(x_batch, y_batch):
            ind = np.argsort(y, axis=0)
            sorted_x = np.take_along_axis(x, ind[::-1], axis=0)
            sorted_y = np.take_along_axis(y, ind[::-1], axis=0)
            temp_x.append(sorted_x)
            temp_y.append(sorted_y)
        out_x.append(temp_x)
        out_y.append(temp_y)
    
    return out_x, out_y
