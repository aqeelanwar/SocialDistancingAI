# Already trained model available @
# https://github.com/tensorflow/models/tree/master/research/object_detection
# was used as a part of this code.

import backbone
import tensorflow as tf
import cv2
import numpy as np


class model:
    def __init__(self):
        # detection_graph, self.category_index = backbone.set_model('ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
        #                                                           'mscoco_label_map.pbtxt')
        # # detection_graph, self.category_index = backbone.set_model(
        # #     'faster_rcnn_resnet50_coco_2018_01_28',
        # #     'mscoco_label_map.pbtxt')
        self.detection_graph, self.category_index = backbone.set_model(
            "ssd_mobilenet_v1_coco_2018_01_28", "mscoco_label_map.pbtxt"
        )

    def get_category_index(self):
        return self.category_index

    def detect_pedestrians(self, frame):
        # Actual detection.
        # input_frame = cv2.resize(frame, (350, 200))

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(input_frame, axis=0)

        image = np.asarray(frame)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        output_dict = self.detection_graph(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        (boxes, scores, classes, num) = (output_dict['detection_boxes'], output_dict['detection_scores'],
            output_dict['detection_classes'], output_dict['num_detections'])

        pedestrian_score_threshold = 0.35
        pedestrian_boxes = []
        total_pedestrians = 0

        for i in range(int(num)):
            if classes[i] in self.category_index.keys():
                class_name = self.category_index[classes[i]]["name"]
                # print(class_name)
                if class_name == "person" and scores[i] > pedestrian_score_threshold:
                    total_pedestrians += 1
                    score_pedestrian = scores[i]
                    pedestrian_boxes.append(boxes[i])

        return pedestrian_boxes, total_pedestrians
