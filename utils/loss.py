import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class YOLOLoss(nn.Module):
    def __init__(self, args):
        super(YOLOLoss, self).__init__()

        self.img_size = args.img_size

        self.S = args.S
        self.B = args.B
        self.C = args.C

        self.l_coord = args.l_coord
        self.l_noobj = args.l_noobj

        self.mse_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def compute_iou(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2

        i_x1 = max(b1_x1, b2_x1)
        i_y1 = max(b1_y1, b2_y1)
        i_x2 = min(b1_x2, b2_x2)
        i_y2 = min(b1_y2, b2_y2)

        inter_area = max(0, i_x2 - i_x1 + 1) * max(0, i_y2 - i_y1 + 1)

        box1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        box2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / float(box1_area + box2_area - inter_area)

        return iou
        

    def responsible_masking(self, pred, target):
        batch_size, rows, cols, _ = target.shape

        for b in range(batch_size):
            for c in range(cols):
                for r in range(rows):
                    if target[b, c, r, 4] > 0:
                        pred1_tx, pred1_ty, pred1_w, pred1_h = pred[b, c, r, :4]

                        pred1_cx = (pred1_tx + c) * (self.img_size // self.S)
                        pred1_cy = (pred1_ty + r) * (self.img_size // self.S)
                        pred1_w = pred1_w ** 2
                        pred1_h = pred1_h ** 2

                        box1 = [pred1_cx - pred1_w // 2, pred1_cy - pred1_h // 2,\
                                 pred1_cx + pred1_w // 2, pred1_cy + pred1_h // 2]
                        
                        pred2_tx, pred2_ty, pred2_w, pred2_h = pred[b, c, r, :4]

                        pred2_cx = (pred2_tx + c) * (self.img_size // self.S)
                        pred2_cy = (pred2_ty + r) * (self.img_size // self.S)
                        pred2_w = pred2_w ** 2
                        pred2_h = pred2_h ** 2

                        box2 = [pred2_cx - pred2_w // 2, pred2_cy - pred2_h // 2,\
                                 pred2_cx + pred2_w // 2, pred2_cy + pred2_h // 2]

                        g_tx, g_ty, g_w, g_h = target[b, c, r, :4]

                        g_cx = (g_tx + c) * (self.img_size // self.S)
                        g_cy = (g_ty + r) * (self.img_size // self.S)
                        g_w = g_w ** 2
                        g_h = g_h ** 2

                        g_box1 = [g_cx - g_w // 2, g_cy - g_h // 2, g_cx + g_w // 2, g_cy + g_h // 2]

                        iou1 = self.compute_iou(box1, g_box1)
                        iou2 = self.compute_iou(box2, g_box1)
                        
                        if iou1 > iou2:
                            target[b, c, r, 9] = 0
                        elif iou2 > iou1:
                            target[b, c, r, 4] = 0
                        else:
                            if random.random() > 0.5:
                                target[b, c, r, 9] = 0
                            else:
                                target[b, c, r, 4] = 0
                                
        return pred, target

    def forward(self, pred, target):
        batch_size = pred.shape[0]

        pred, target = self.responsible_masking(pred, target)
        
        box1_obj_mask = target[:, :, :, 4] > 0
        box2_obj_mask = target[:, :, :, 9] > 0

        noobj_mask = (target[:, :, :, 4] == 0) & (target[:, :, :, 9] == 0)
    
        pred_res1 = pred[box1_obj_mask]
        pred_res2 = pred[box2_obj_mask]
        pred_noobj = pred[noobj_mask]
        
        target_res1 = target[box1_obj_mask]
        target_res2 = target[box2_obj_mask]
        target_noobj = target[noobj_mask]

        loc_loss = self.mse_loss(pred_res1[:, :4], target_res1[:, :4])\
                    + self.mse_loss(pred_res2[:, 5:9], target_res2[:, 5:9])
        
        obj_loss = self.mse_loss(pred_res1[:, 4], target_res1[:, 4])\
                    + self.mse_loss(pred_res2[:, 9], target_res2[:, 9])

        cls_loss = self.mse_loss(pred_res1[:, 10:], target_res1[:, 10:])\
                    + self.mse_loss(pred_res2[:, 10:], target_res2[:, 10:])

        noobj_loss = self.mse_loss(pred_noobj[:, 4], target_noobj[:, 4])\
                    + self.mse_loss(pred_noobj[:, 9], target_noobj[:, 9])
        
        return self.l_coord * loc_loss / batch_size, cls_loss / batch_size, obj_loss / batch_size, self.l_noobj * noobj_loss / batch_size