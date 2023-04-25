import os
import torch

def get_all_files_in_folder(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def non_maximum_suppression(boxes, scores, threshold):
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = torch.argsort(scores, descending=True)
    
    indices = []
    while order.numel() > 0:
        i = order[0]
        
        indices.append(i.item())
        
        if order.numel() == 1:
            break
        
        j = order[1:]
        iou = compute_iou(boxes[i], boxes[j])
        mask = (iou < threshold)
        order = order[1:][mask]

    indices = torch.tensor(indices)
    
    return indices

def compute_iou(box1, box2):
    xmin = torch.max(box1[0], box2[:, 0])
    ymin = torch.max(box1[1], box2[:, 1])
    xmax = torch.min(box1[2], box2[:, 2])
    ymax = torch.min(box1[3], box2[:, 3])
    
    width = torch.clamp(xmax - xmin, min=0)
    height = torch.clamp(ymax - ymin, min=0)
    area_intersection = width * height
    
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    area_union = area_box1 + area_box2 - area_intersection
    
    iou = area_intersection / area_union
    return iou