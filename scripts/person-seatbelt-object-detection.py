#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
# import keras
import keras
import glob

import sys
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

# use this to change which GPU to use
# gpu = 0

# set the modified tf session as backend in keras
# setup_gpu(gpu)

# ## Run detection on example

# In[ ]:


def render_videos(model, input_glob_path, destination_path):
    for vid in glob.glob(input_glob_path):
        vid_parts = vid.split('/')
        filename = vid_parts[-1]
        vid_name = filename.split('.')[0]
        row = vid_parts[-4]
        action = vid_parts[-3]
        pass_count = int(vid_parts[-2])
        cap = cv2.VideoCapture(vid)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        destination_folder = '/'.join(destination_path.split('/')
                                      [:-1]).format(row, action, pass_count)
        if not(os.path.exists(destination_folder)):
            os.makedirs(destination_folder)
        rec = cv2.VideoWriter(destination_path.format(
            row, action, pass_count, filename), fourcc, fps, (width, height))

        false_positives = 0
        false_negatives = 0
        true_positives = 0
        # load image
        # image = read_image_bgr('000000008021.jpg')
        start = time.time()

        while cap.isOpened():
            for _ in range(10):
                ret, image = cap.read()
                if not ret:
                    break
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
            predicted_pass_count = 0
            # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.5:
                    break
                if labels_to_names[label] == 'person':
                    predicted_pass_count += 1
                color = label_color(label)

                b = box.astype(int)
                draw_box(draw, b, color=color)

                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)
            if predicted_pass_count == pass_count:
                true_positives += 1
            elif predicted_pass_count > pass_count:
                false_positives += 1
            elif predicted_pass_count < pass_count:
                false_negatives += 1
                # plt.figure(figsize=(15, 15))
                # plt.axis('off')
                # plt.imshow(draw)
                # plt.show()

            rec.write(draw)

            # cv2.imshow('out', draw)
            # cv2.waitKey(1)
        print(
            "processing time: {} for video: {}".format(
                time.time() - start, vid))

        with open(destination_path.format(row, action, pass_count, vid_name + '.csv'), 'w') as f:
            f.write('true_positives,false_positives,false_negatives\n')
            f.write(
                '{},{},{}'.format(
                    true_positives,
                    false_positives,
                    false_negatives))

        cap.release()
        rec.release()

if __name__ == '__main__':
    # ## Load RetinaNet model

    # adjust this to point to your downloaded/trained model
    # models can be downloaded here:
    # https://github.com/fizyr/keras-retinanet/releases
    for model_path in glob.glob(
            '/home/ubuntu/snapshots_to_test_2/resnet50_csv_*.h5'):
        snapshot_number = int(model_path.split(
            '/')[-1].split('.')[0].split('_')[-1])
        # if snapshot_number % 5 != 0:
        #     continue
        # load retinanet model
        model = models.load_model(model_path, backbone_name='resnet50')

        # if the model is not converted to an inference model, use the line below
        # see:
        # https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
        model = models.convert_model(model)

        # print(model.summary())

        # load label to names mapping for visualization purposes
        labels_to_names = {0: 'person', 1: 'seatbelt'}

        input_glob_path = '/home/ubuntu/videos/**/**/**/*'
        destination_path = '/home/ubuntu/render_videos_2/{}/{}/{}/{}/{}'.format(
            snapshot_number, '{}', '{}', '{}', '{}')
        snapshot_render_path = '/'.join(destination_path.split('/')[:5])
        if os.path.exists(snapshot_render_path):
            continue
        render_videos(model, input_glob_path, destination_path)

        model = None
        tf.reset_default_graph()
