#!/usr/bin/env python3
"""
Yolo
"""
import tensorflow.keras as K


class Yolo():
    """
    Class Yolo
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor of class

        Args:
            model_path (str): the path to where a Darknet Keras model is stored
            classes_path (str): the path to where the list of class names used
                                for the Darknet model, listed in order of
                                index, can be found.
            class_t (float): the box score threshold for the initial filtering
                             step.
            nms_t (float): the IOU threshold for non-max suppression.
            anchors (np.ndarray): matrix of shape (outputs, anchor_boxes, 2)
                                  containing all of the anchor boxes:
                                  outputs: is the number of outputs
                                  (predictions) made by the Darknet model.
                                  anchor_boxes: is the number of anchor boxes
                                  used for each prediction.
                                  2: [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f]
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
