import os
import cv2
import torch
import numpy as np

def draw_bbox(image, bbox, class_idx, score, color=(0, 255, 0), thickness=2):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    label = f"Class {class_idx}: {score:.2f}"
    
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y2 - label_size[1] - baseline), (x1 + label_size[0], y2), color, -1)
    cv2.putText(image, label, (x1, y2 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return image

def post_processing(args, output : torch.Tensor):
    # output shape : [S, S, 5 * B + C]

    bboxes = []
    scores = []
    classes = []

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i, j, 4] > args.threshold:
                cx = int((output[i, j, 0] + i) * (args.img_size // args.S))
                cy = int((output[i, j, 1] + j) * (args.img_size // args.S))
                w = int(output[i, j, 2] ** 2)
                h = int(output[i, j, 3] ** 2)

                cx = max(0, cx)
                cy = max(0, cy)
                w = max(0, w)
                h = max(0, h)

                score = output[i, j, 4]
                cls_prob = output[i, j, 10:]
                
                cls_prob = torch.mul(score, cls_prob)
                cls = torch.argmax(cls_prob)
                
                bboxes.append([cx - (w // 2), cy - (h // 2), w, h])
                scores.append(score)
                classes.append(cls)
            
            # implement if you modify loss function
            # if output[i, j, 9] > args.threshold:

    return bboxes, scores, classes

def get_all_files_in_folder(folder_path : str):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

def non_max_suppression(bboxes : list, scores : list, iou_thresh : float):
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bboxes_sorted = [bboxes[i] for i in sorted_indices]
    scores_sorted = [scores[i] for i in sorted_indices]
    
    keep_indices = []
    
    while len(bboxes_sorted) > 0:
        i = sorted_indices[0]
        keep_indices.append(i)
        
        bboxes_sorted.pop(0)
        scores_sorted.pop(0)
        sorted_indices.pop(0)
        
        iou_scores = []
        for j, bbox in enumerate(bboxes_sorted):
            iou_score = calculate_iou(bboxes[i], bbox)
            iou_scores.append(iou_score)
        
        indices_to_remove = [j for j, iou in enumerate(iou_scores) if iou >= iou_thresh]
        for j in reversed(indices_to_remove):
            bboxes_sorted.pop(j)
            scores_sorted.pop(j)
            sorted_indices.pop(j)
    
    bboxes_nms = [bboxes[i] for i in keep_indices]
    scores_nms = [scores[i] for i in keep_indices]
    
    return bboxes_nms, scores_nms

def calculate_iou(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]
    
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    
    return iou