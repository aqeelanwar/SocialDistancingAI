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
        detection_graph, self.category_index = backbone.set_model(
            "ssd_mobilenet_v1_coco_2018_01_28", "mscoco_label_map.pbtxt"
        )
        self.sess = tf.InteractiveSession(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
        self.detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
        self.detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
        self.detection_classes = detection_graph.get_tensor_by_name(
            "detection_classes:0"
        )
        self.num_detections = detection_graph.get_tensor_by_name("num_detections:0")

    def get_category_index(self):
        return self.category_index

    def detect_pedestrians(self, frame):
        # Actual detection.
        # input_frame = cv2.resize(frame, (350, 200))
        input_frame = frame

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(input_frame, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [
                self.detection_boxes,
                self.detection_scores,
                self.detection_classes,
                self.num_detections,
            ],
            feed_dict={self.image_tensor: image_np_expanded},
        )

        classes = np.squeeze(classes).astype(np.int32)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        pedestrian_score_threshold = 0.35
        pedestrian_boxes = []
        total_pedestrians = 0
        for i in range(int(num[0])):
            if classes[i] in self.category_index.keys():
                class_name = self.category_index[classes[i]]["name"]
                # print(class_name)
                if class_name == "person" and scores[i] > pedestrian_score_threshold:
                    total_pedestrians += 1
                    score_pedestrian = scores[i]
                    pedestrian_boxes.append(boxes[i])

        return pedestrian_boxes, total_pedestrians
