#!/usr/bin/python2

import cv2, os
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from scipy.io import loadmat

go_auto = True

#sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2010', 'train'), ('2010', 'val')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "person2"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

total_num_persons = 0
total_num_red_persons = 0

def convert_annotation(year, image_id):
    global total_num_persons
    global total_num_red_persons

    in_file = open('VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOC%s/labels/%s.txt'%(year, image_id), 'w')

    num_person_masks = 0

    in_objs = None
    if os.path.exists('VOC%s/Annotations_Part/%s.mat'%(year, image_id)):
        frame = cv2.imread('VOC%s/JPEGImages/%s.jpg'%(year, image_id))
        in_objs = loadmat('VOC%s/Annotations_Part/%s.mat'%(year, image_id))
        total_part_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        in_objs = in_objs["anno"][0][0][1][0]
        for in_obj in in_objs:
            name = in_obj[0][0]
            if name == 'person':
                masked = False
                if len(in_obj[3]) > 0:
                    parts = in_obj[3][0]
                else:
                    parts = []
                for part in parts:
                    part_name = part[0][0]
                    #if part_name in ["torso", "ruarm", "luarm"]:
                    if part_name in ["torso"]:
                        part_mask = part[1]
                        total_part_mask = cv2.bitwise_or(part_mask, total_part_mask)
                        masked = True
                if masked:
                    num_person_masks += 1

    tree=ET.parse(in_file)
    root = tree.getroot()

    num_persons = 0

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        if cls == "person":
            num_persons += 1

    if (in_objs is not None) and (num_persons != num_person_masks) or (num_persons == 0):
        return (False, False)

    if in_objs is not None:
        msk3 = cv2.merge((total_part_mask, total_part_mask, total_part_mask * 255))
        frame[total_part_mask == 1] = cv2.addWeighted(frame, 0.4, msk3, 0.6, 0)[total_part_mask == 1]

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        if (in_objs is not None) and (cls == "person"):
            cls = "person2"
            total_num_red_persons += 1

        if cls not in classes or int(difficult)==1:
            continue
        if (cls == "person") and (in_objs is None):
            total_num_persons += 1
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    if (in_objs is not None):
        cv2.imwrite('VOC%s/JPEGImages2/%s.jpg'%(year, image_id), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    if not go_auto:
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
    return (True, in_objs is not None)

if __name__ == "__main__":
    wd = getcwd()

    for year, image_set in sets:
        if not os.path.exists('VOC%s/labels/'%(year)):
            os.makedirs('VOC%s/labels/'%(year))
        image_ids = open('VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        list_file = open('%s_%s.txt'%(year, image_set), 'w')
        for image_id in image_ids:
            good, im2 = convert_annotation(year, image_id)
            if good:
                if im2:
                    list_file.write('%s/VOC%s/JPEGImages2/%s.jpg\n'%(wd, year, image_id))
                else:
                    list_file.write('%s/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        list_file.close()

    print(total_num_persons, total_num_red_persons)

    #os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
    #os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
