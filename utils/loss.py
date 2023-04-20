import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, args):
        super(YOLOLoss, self).__init__()

        self.img_size = args.img_size

        self.S = args.S
        self.B = args.B
        self.C = args.C

        self.l_coord = args.l_coord
        self.l_noobj = args.l_noobj
        
    def forward(self, pred, target):
        batch_size = pred.shape[0]
        
        mse_loss = torch.nn.MSELoss()
        bce_loss = torch.nn.BCELoss()

        # localization loss
        obj_mask = target[:, :, :, 4] > 0
        obj_pred = pred[obj_mask]
        obj_target = target[obj_mask]

        pred_bbox1 = obj_pred[:, 0:4]
        pred_bbox2 = obj_pred[:, 5:9]

        target_bbox1 = obj_target[:, 0:4]
        target_bbox2 = obj_target[:, 5:9]

        ious1 = calc_iou(pred_bbox1, target_bbox1)
        ious2 = calc_iou(pred_bbox2, target_bbox2)

        int_tb1 = torch.floor(target_bbox1[:, :2])
        target_bbox1[:, :2] = target_bbox1[:, :2] - int_tb1
        int_tb2 = torch.floor(target_bbox2[:, :2])
        target_bbox2[:, :2] = target_bbox2[:, :2] - int_tb2

        # objectness loss
        pred_obj1 = obj_pred[:, 4]
        pred_obj2 = obj_pred[:, 9]
        pred_class = obj_pred[:, 10:13]

        target_obj1 = obj_target[:, 4]
        target_obj2 = obj_target[:, 9]
        target_class = obj_target[:, 10:13]
        
        loc_loss = 0
        obj_loss = 0

        for i in range(len(ious1)):
            if ious1[i] > ious2[i]:
                loc_loss += mse_loss(pred_bbox1[i, :2], target_bbox1[i, :2]) + mse_loss(pred_bbox1[i, 2:4], target_bbox1[i, 2:4])
                obj_loss += torch.pow((pred_obj1[i] - ious1[i]), 2) + self.l_noobj * torch.pow((pred_obj2[i] - 0.0), 2)
            else:
                loc_loss += mse_loss(pred_bbox2[i, :2], target_bbox2[i, :2]) + mse_loss(pred_bbox2[i, 2:4], target_bbox2[i, 2:4])
                obj_loss += torch.pow((pred_obj2[i] - ious2[i]), 2) + self.l_noobj * torch.pow((pred_obj1[i] - 0.0), 2)
        
        # classification loss
        cls_loss = mse_loss(pred_class, target_class)

        return self.l_coord * loc_loss, obj_loss, cls_loss
        
def calc_iou(boxes1, boxes2):
    ious = []

    img_size = 448
    divisor = img_size // 7

    for box1, box2 in zip(boxes1, boxes2):

        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x1 = x1 * divisor
        x2 = x2 * divisor
        y1 = y1 * divisor
        y2 = y2 * divisor

        w1 = w1 * img_size
        h1 = h1 * img_size
        w2 = w2 * img_size
        h2 = h2 * img_size

        # Calculate the coordinates of the intersection rectangle
        x_left = max(x1 - w1 / 2, x2 - w2 / 2)
        y_top = max(y1 - h1 / 2, y2 - h2 / 2)
        x_right = min(x1 + w1 / 2, x2 + w2 / 2)
        y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)
        
        # If the intersection is empty, return 0
        if x_right < x_left or y_bottom < y_top:
            iou = torch.tensor(0.0)
            if torch.cuda.is_available():
                iou = iou.cuda()
            ious.append(iou)
            continue
        
        # Calculate the area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate the area of both bounding boxes
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Calculate the IOU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        if torch.cuda.is_available():
            iou = iou.cuda()
        
        ious.append(iou)
    return ious