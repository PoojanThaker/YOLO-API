import os
import cv2
import time
import numpy as np

confthres = 0.5
nmsthres = 0.1
yolo_path = "./"
labels_path = "coco.names"
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"


class cached_property(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """

    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)

        return attr


# Lables=get_labels(labelsPath)
# CFG=get_config(cfgpath)
# Weights=get_weights(wpath)
# nets=load_model(CFG,Weights)


class Yolo:
    @cached_property
    def labels(self):
        # load the COCO class labels our YOLO model was trained on
        # labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
        lpath = os.path.sep.join([yolo_path, labels_path])
        labels = open(lpath).read().strip().split("\n")
        return labels

    @cached_property
    def weights(self):
        # derive the paths to the YOLO weights and model configuration
        wpath = os.path.sep.join([yolo_path, weights_path])
        return wpath

    @cached_property
    def config(self):
        cpath = os.path.sep.join([yolo_path, config_path])
        return cpath

    @cached_property
    def net(self):
        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        return net

    def get_predection(self, image):
        (H, W) = image.shape[:2]

        net = self.net
        labels = self.labels

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                # print(scores)
                classID = np.argmax(scores)
                # print(classID)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > confthres:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                                nmsthres)
        final_boxes = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (int(boxes[i][0]), int(boxes[i][1]))
                (w, h) = (int(boxes[i][2]), int(boxes[i][3]))
                final_boxes.append((x, y, w, h, str(labels[classIDs[i]]), confidences[i]))
        return final_boxes
