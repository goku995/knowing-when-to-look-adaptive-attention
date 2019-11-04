import torch
import torchvision
import torchvision.transforms as T

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from PIL import Image
import numpy as np
import os
import cv2
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 9 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',    
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

print(len(COCO_INSTANCE_CATEGORY_NAMES))

def get_prediction(img_path, threshold):
    img = Image.open(img_path)  # Load the image
    transform = T.Compose([T.ToTensor()])  # Defing PyTorch Transform
    img = transform(img)  # Apply the transform to the image
    pred = model([img])  # Pass the image to the model
    # print(pred)
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(
        pred[0]['labels'].numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(
        pred[0]['boxes'].detach().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    print(len(pred_class), len(pred_boxes))
    # Get list of index with score greater than threshold.
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def object_detection_api(img_path, threshold=0.5, rect_th=1, text_size=1, text_th=1):

    boxes, pred_cls = get_prediction(img_path, threshold)  # Get predictions
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(
            0, 255, 0), thickness=rect_th)  # Draw Rectangle with the coordinates
        # cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
        print(pred_cls[i])
    plt.figure(figsize=(20, 30))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


image_path = '../../caption-dataset/flickr30k_images/'

image_folder = os.listdir(image_path)
filename = np.random.choice(image_folder, 1)[0] # 1584315962.jpg, 392467282.jpg # 
file_key = filename.split(".")[0]

# print(filename)

object_detection_api(os.path.join(image_path, filename), threshold=0)

annotation_path = "../../caption-dataset/flickr30k_entities/annotation_data.json"
sentence_path = "../../caption-dataset/flickr30k_entities/sentence_data.json"


def get_annotation_data(annotation_path):
    with open(annotation_path, 'r') as j:
        annotation_data = json.load(j)
    return annotation_data


def get_sentence_data(sentence_path):
    with open(sentence_path, 'r') as j:
        sentence_data = json.load(j)
    return sentence_data


annotation_data = get_annotation_data(annotation_path)
sentence_data = get_sentence_data(sentence_path)

sentences = sentence_data[file_key]

print(annotation_data[file_key])
boxes = annotation_data[file_key]['boxes']

image = plt.imread(os.path.join(image_path, filename))

def union_boxes(bboxes):
    if len(bboxes) == 1:
        return bboxes[0]
    else:
        union_box = bboxes[0]
        for bbox in bboxes:
            union_box[0] = min(union_box[0], bbox[0])
            union_box[1] = min(union_box[1], bbox[1])
            union_box[2] = max(union_box[2], bbox[2])
            union_box[3] = max(union_box[3], bbox[3])
        return union_box

# for sentence in sentences:
#     sentence_phrases = sentence['phrases']
#     print(sentence['sentence'])
#     for phrase in sentence_phrases:
#         phrase_id = phrase['phrase_id']
#         if phrase_id not in boxes:
#             continue
#         phrase_str = phrase['phrase'].lower() # .split()
#         print(phrase_str)
#         bboxes = boxes[phrase_id]
#         bbox = union_boxes(bboxes)
#         image_patch = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

#         plt.imshow(image_patch)
#         plt.show()

#     print("----------------------")


# img_lbl, regions = selectivesearch.selective_search(
#     image, scale=500, sigma=0.9, min_size=10)

# print(len(regions))
# candidates = set()
# for r in regions:
#     # excluding same rectangle (with different segments)
#     if r['rect'] in candidates:
#         continue
#     # excluding regions smaller than 2000 pixels
#     if r['size'] < 2000:
#         continue
#     # distorted rects
#     x, y, w, h = r['rect']
#     if w / h > 1.2 or h / w > 1.2:
#         continue
#     candidates.add(r['rect'])
# print(len(candidates))
# # draw rectangles on the original image
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
# ax.imshow(image)
# for x, y, w, h in candidates:
#     # print(x, y, w, h)
#     rect = mpatches.Rectangle(
#         (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
#     ax.add_patch(rect)

# plt.show()