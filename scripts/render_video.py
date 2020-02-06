#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
# import keras
import keras
import glob

import sys

# ADD YOUR PATH TO THE REPO BELOW
sys.path.append('/home/ubuntu/keras-retinanet')


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

CONDFIDENCE_THRESHOLD = 0.5


def py_cpu_softnms(dets, scores, sigma=0.5):
    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        weight = np.exp(-(ovr * ovr) / sigma)

        scores[pos:] = weight * scores[pos:]

    return dets[:, :4], scores


def render_video(model, video_path, destination_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    filename = video_path.split('/')[-1]
    if not(os.path.exists(destination_path)):
        os.makedirs(destination_path)
    rec = cv2.VideoWriter(destination_path +
                          filename, fourcc, fps, (width, height))

    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # copy to draw on
        draw = image.copy()
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=480, max_side=720)

        # process image
        boxes, scores, labels = model.predict_on_batch(
            np.expand_dims(image, axis=0))

        # correct for image scale
        boxes /= scale
        label_dicts = {}
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if label == -1:
                continue
            if label not in label_dicts:
                label_dicts[label] = {'boxes': [box], 'scores': [score]}
            else:
                label_dicts[label]['boxes'].append(box)
                label_dicts[label]['scores'].append(score)
        all_boxes = []
        all_scores = []
        all_labels = []
        for label in label_dicts:
            boxes = np.array(label_dicts[label]['boxes'])
            scores = np.array(label_dicts[label]['scores'])
            new_boxes, new_scores = py_cpu_softnms(
                boxes, scores, sigma=0.5)
            for box, score in zip(new_boxes, new_scores):

                all_boxes.append(box)
                all_scores.append(score)
                all_labels.append(label)
        # visualize detections
        for box, score, label in zip(all_boxes, all_scores, all_labels):
            # scores are sorted so we can break
            if score < CONDFIDENCE_THRESHOLD:
                continue
            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)

        rec.write(draw)

    print(
        "processing time: {} for video: {}".format(
            time.time() - start, video_path))

    cap.release()
    rec.release()

if __name__ == '__main__':
    # ENTER THE PATH TO THE MODEL .h5 FILE HERE
    model_path = '/home/vinit/keras_model/resnet50_csv_09.h5'

    model = models.load_model(model_path, backbone_name='resnet50')

    model = models.convert_model(model)

    # DO NOT CHANGE
    labels_to_names = {1: 'person', 0: 'seatbelt'}

    # ENTER THE PATH TO THE INPUT VIDEO HERE
    video_path = '/home/vinit/filename.mov'

    # ENTER THE OUTPUT DIRECTORY HERE
    destination_path = '/home/vinit/Desktop/'

    render_video(model, video_path, destination_path)
