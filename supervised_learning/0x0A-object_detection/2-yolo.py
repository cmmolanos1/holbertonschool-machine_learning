#!/usr/bin/env python3
"""
Yolo
"""
import tensorflow.keras as K
import numpy as np


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

    def _sigmoid(self, x):
        """
        Perform sigmoid function of a vector.
        """
        return 1. / (1. + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Args:
            outputs (list of np.ndarray): the predictions from the Darknet
                                          model for a single image:
            Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
            grid_height & grid_width => the height and width of the grid used
                                        for the output.
            anchor_boxes             => the number of anchor boxes used.
            4                        => (t_x, t_y, t_w, t_h).
            1                        => box_confidence.
            classes                  => class probabilities for all classes.

            image_size (numpy.ndarray):  containing the image’s original size
                                         [image_height, image_width]

        Returns:
            Tuple of (boxes, box_confidences, box_class_probs):

            boxes: a list of numpy.ndarrays of shape
                   (grid_height, grid_width, anchor_boxes, 4) containing the
                   processed boundary boxes for each output, respectively:
                   4 => (x1, y1, x2, y2)
                   (x1, y1, x2, y2) should represent the boundary box relative
                   to original image.

            box_confidences: a list of  of shape (grid_height, grid_width,
                             anchor_boxes, 1) containing the box confidences
                             for each output, respectively.

            box_class_probs: a list of numpy.ndarrays of shape
                             (grid_height, grid_width, anchor_boxes, classes)
                             containing the box’s class probabilities for each
                             output, respectively.
        """
        boxes = [output[..., :4] for output in outputs]
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i in range(len(outputs)):
            grid_height, grid_width, anchor_boxes, _ = outputs[i].shape

            # BOXES
            # Extract from outputs tx, ty, tw, th.
            tx = outputs[i][..., 0]
            ty = outputs[i][..., 1]
            tw = outputs[i][..., 2]
            th = outputs[i][..., 3]

            # From anchors extract pw and ph.
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # Calculate the current cx and cy
            cx = np.tile(np.arange(0, grid_width), grid_height)
            cx = cx.reshape(grid_width, grid_width, 1)

            cy = np.tile(np.arange(0, grid_width), grid_height)
            cy = cy.reshape(grid_height, grid_height).T
            cy = cy.reshape(grid_height, grid_height, 1)

            # Calculate the prediction of boundary box.
            bx = self._sigmoid(tx) + cx
            by = self._sigmoid(ty) + cy
            bw = pw * np.exp(tw)
            bh = ph * np.exp(th)

            # Normalize to grid.
            bx /= grid_width
            by /= grid_height

            # Normalizing to model input size.
            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            bw /= input_width
            bh /= input_height

            # Coordinates of bounding box
            x1 = (bx - bw / 2)
            y1 = (by - bh / 2)
            x2 = (bx + bw / 2)
            y2 = (by + bh / 2)

            # Scaling to the image size, and assigning to the pre-created
            # box.
            boxes[i][..., 0] = x1 * image_width
            boxes[i][..., 1] = y1 * image_height
            boxes[i][..., 2] = x2 * image_width
            boxes[i][..., 3] = y2 * image_height

            """ BOX CONFIDENCE
            Extract box confidence from output, then standarize the
            probability using sigmoid and then add to the list. It is necesary
            to reshape to get the required output"""
            box_confidence = self._sigmoid(outputs[i][:, :, :, 4])
            box_confidences.append(box_confidence.reshape(grid_height,
                                                          grid_width,
                                                          anchor_boxes,
                                                          1))

            """ BOX CLASSES PROBABILITIES
            Extract the classes prob from output, then standarize the
            probability using sigmoid and then add to the list."""
            box_class_prob = self._sigmoid(outputs[i][:, :, :, 5:])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters the boxex.

        Args:
            boxes (list): contains ndarrays of shape
                          (grid_height, grid_width, anchor_boxes, 4)
                          containing the processed boundary boxes for each
                          output, respectively.
            box_confidences (list): contains ndarrays of shape
                                    (grid_height, grid_width, anchor_boxes, 1)
                                    containing the processed box confidences
                                    for each output, respectively.
            box_class_probs (list): contains ndarrays of shape
                                    (grid_height, grid_width,
                                    anchor_boxes, classes)
                                    containing the processed box class
                                    probabilities for each output,
                                    respectively.

        Returns:
            A tuple of (filtered_boxes, box_classes, box_scores):

            * filtered_boxes: a numpy.ndarray of shape (?, 4) containing all
                              of the filtered bounding boxes.
            * box_classes: a numpy.ndarray of shape (?,) containing the class
                           number that each box in filtered_boxes predicts,
                           respectively.
            * box_scores: a numpy.ndarray of shape (?) containing the box
                          scores for each box in filtered_boxes, respectively.
        """
        # Step 1: Compute box scores
        box_scores = []
        for box_confidence, box_class_prob in zip(box_confidences,
                                                  box_class_probs):
            box_scores.append(box_confidence * box_class_prob)

        # Step 2: Find the box_classes using the max box_scores, keep track of
        # the corresponding score
        box_class = [score.argmax(axis=-1) for score in box_scores]
        box_class_list = [box.reshape(-1) for box in box_class]
        box_class_concat = np.concatenate(box_class_list, axis=-1)

        box_class_scores = [score.max(axis=-1) for score in box_scores]
        box_score_list = [box.reshape(-1) for box in box_class_scores]
        box_scores_concat = np.concatenate(box_score_list, axis=-1)

        boxes_list = [box.reshape(-1, 4) for box in boxes]
        boxes_concat = np.concatenate(boxes_list, axis=0)

        # Step 3: Create a filtering mask based on "box_class_scores" by using
        # "threshold". The mask should have the same dimension as
        # box_class_scores, and be True for the boxes you want to keep
        # (with probability >= threshold)
        filtering_mask = np.where(box_scores_concat >= self.class_t)

        # Step 4: Apply the mask to box_class_scores, boxes and box_classes
        boxes_ = boxes_concat[filtering_mask]
        box_classes = box_class_concat[filtering_mask]
        box_scores = box_scores_concat[filtering_mask]

        return (boxes_, box_classes, box_scores)
