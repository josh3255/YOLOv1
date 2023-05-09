import json
import torch

from tqdm import tqdm

from utils.utils import post_processing
from utils.utils import non_max_suppression

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import get_args
from models.yolo import YOLO
from dataset.coco import ValDataset
from torch.utils.data import DataLoader

def val(args):
    # dataloader
    val_dataset = ValDataset(args, args.val_ann)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = YOLO(args)

    device_count = torch.cuda.device_count()
    devices = [torch.device("cuda:"+str(i) if torch.cuda.is_available() else "cpu") for i in range(device_count)]
    
    model = torch.nn.DataParallel(model, device_ids=range(device_count))
    for i in range(device_count):
        model.to(devices[i])
    
    if args.weights != '':
        checkpoint = torch.load(args.weights, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        return 0
    
    # Model Evaluation
    with torch.no_grad():
        model.eval()
        
        prediction = []
        
        for batch_idx, (image_ids, inputs) in enumerate(tqdm(val_dataloader)):
            outputs = model(inputs)
            
            for image_id, output in zip(image_ids, outputs):
                bboxes, scores, classes = post_processing(args, output)
                bboxes, scores = non_max_suppression(bboxes, scores, args.iou_threshold)
                
                for bbox, score, cls in zip(bboxes, scores, classes):
                    prediction.append({'image_id' : int(image_id.item()), 'category_id' : int(cls.item() + 1), \
                                        'bbox' : bbox, 'score' : score.item()})

        with open(args.results, 'w') as wf:
            json.dump(prediction, wf)
        
        cocoGt = COCO(args.val_ann)
        cocoDt = cocoGt.loadRes(args.results)
        imgIds=sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

def main():
    args = get_args()
    val(args)

if __name__ == "__main__":
	main()