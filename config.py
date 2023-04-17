import argparse

def get_args():
    parser = argparse.ArgumentParser(description='YOLO Arguments')

    # dataset args
    parser.add_argument('--train-ann', type=str, default='/data/detection/detection/annotations/train.json', help='COCO train json')
    parser.add_argument('--val-ann', type=str, default='/data/detection/detection/annotations/val.json', help='COCO val json')

    # training args
    parser.add_argument('--batch-size', type=int, default=8, help='')
    parser.add_argument('--max-epoch', type=int, default=300, help='' )
    parser.add_argument('--eval-interval', type=int, default=5, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')
    

    # model args
    parser.add_argument('--S', type=int, default=7, help='num of grid cell')
    parser.add_argument('--B', type=int, default=2, help='num of bounding box per cell')
    parser.add_argument('--C', type=int, default=3, help='num of classes')
    # augmentation args

    args = parser.parse_args()
    return args

