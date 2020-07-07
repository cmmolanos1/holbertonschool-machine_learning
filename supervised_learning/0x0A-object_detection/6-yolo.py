#!/usr/bin/env python3
"""
Yolo
"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Perform non max suppression

        Args:
            filtered_boxes (np.ndarray): matrix of shape (?, 4) containing all
                                         of the filtered bounding boxes.
            box_classes (np.ndarray): matrix of shape (?,) containing the
                                      class number for the class that
                                      filtered_boxes predicts, respectively.
            box_scores (np.ndarray): matrix of shape (?) containing the box
                                     scores for each box in filtered_boxes,
                                     respectively.

        Returns:
            Tuple of
            (box_predictions, predicted_box_classes, predicted_box_scores):

            * box_predictions: a numpy.ndarray of shape (?, 4) containing all
                               of the predicted bounding boxes ordered by
                               class and box score.
            * predicted_box_classes: a numpy.ndarray of shape (?,) containing
                                     the class number for box_predictions
                                     ordered by class and box score,
                                     respectively.
            * predicted_box_scores: a numpy.ndarray of shape (?) containing
                                    the box scores for box_predictions ordered
                                    by class and box score, respectively.
        """
        tmp_boxes = []
        tmp_classes = []
        tmp_scores = []

        for clase in set(box_classes):
            # filter the inputs by the current class.
            indexes = np.where(box_classes == clase)
            boxes_ofclas = filtered_boxes[indexes]
            classes_ofclas = box_classes[indexes]
            scores_ofclas = box_scores[indexes]

            # theory:towardsdatascience.com/
            # non-maximum-suppression-nms-93ce178e177c
            x1 = boxes_ofclas[:, 0]
            y1 = boxes_ofclas[:, 1]
            x2 = boxes_ofclas[:, 2]
            y2 = boxes_ofclas[:, 3]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores_ofclas.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= self.nms_t)[0]
                order = order[inds + 1]

            # Append the survival indexes.
            tmp_boxes.append(boxes_ofclas[keep])
            tmp_classes.append(classes_ofclas[keep])
            tmp_scores.append(scores_ofclas[keep])

        boxes_predic = np.concatenate(tmp_boxes, axis=0)
        classes_predic = np.concatenate(tmp_classes, axis=0)
        scores_predic = np.concatenate(tmp_scores, axis=0)

        return boxes_predic, classes_predic, scores_predic

    @staticmethod
    def load_images(folder_path):
        """Loads images from dataset.

        folder_path (str): the path to the folder holding all the images to
                           load.

        Returns:
            (images, image_paths)

            * images: a list of images as numpy.ndarrays
            * image_paths: a list of paths to the individual images in images
        """
        images = []
        image_paths = []
        for file in glob.glob(folder_path + "/*.*"):
            image_paths.append(file)
            image = cv2.imread(file)
            images.append(image)

        return images, image_paths

    def preprocess_images(self, images):
        """Resize pictures.

        Args:
            images(list): images as numpy.ndarrays.

        Returns:
            (pimages, image_shapes):

            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                     containing all of the preprocessed images.
                     * ni: the number of images that were preprocessed
                     * input_h: the input height for the Darknet model
                     * input_w: the input width for the Darknet model
                     * 3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2) containing
                          the original height and width of the images
                          2 => (image_height, image_width)
        """
        input_width = self.model.input.shape[1].value
        input_height = self.model.input.shape[2].value
        dim = (input_width, input_height)

        pimages = []
        image_shapes = []

        for image in images:
            preprocessed = cv2.resize(image,
                                      dim,
                                      interpolation=cv2.INTER_CUBIC)
            pimage = preprocessed / 255
            pimages.append(pimage)

            image_shapes.append(image.shape)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Draws rectangle on each detected object.

        Args:
            image (np.ndarray): unprocessed image.
            boxes (np.ndarray: the boundary boxes for the image.
            box_classes (np.ndarray: the class indices for each box.
            box_scores (np.ndarray: the box scores for each box.
            file_name (str): the file path where the original image is stored.
        """
        for i in range(len(boxes)):
            x1, y2, x2, y1 = boxes[i]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            cv2.rectangle(img=image,
                          pt1=(x1,y1),
                          pt2=(x2,y2),
                          # (B, G, R)
                          color=(255,0,0),
                          thickness=2)

            text = "{} {:.2f}".format(self.class_names[box_classes[i]],
                                      box_scores[i])

            cv2.putText(img=image,
                        text=text,
                        org=(x1, y2 - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        cv2.imshow(file_name, image)

        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.isdir('detections'):
                os.mkdir('detections')
            os.chdir('detections')
            cv2.imwrite(file_name, image)
            os.chdir('../')
        cv2.destroyAllWindows()
