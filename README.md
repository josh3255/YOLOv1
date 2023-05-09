# YOLOv1

YOLOv1 Paper : [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

This code follows the contents of the paper as much as possible, but has the following differences.

1. Use random select instead of respondable select due to bias occurring early in the training.

2. To prevent data from being heavily modulated, color space variations are within the 20% range.

3. Confidence and class probability pass through the sigmoid layer.



## TODO

- [x] data augmentation
- [x] add demo.py
- [x] non-maximum suppression
- [x] add val.py
- [ ] video demo
- [ ] upload pretrained weights



## Installation

Currently, only support docker development environments.

1. Clone the repo

``` shell
git clone https://github.com/josh3255/YOLOv1.git
cd YOLOv1
```

2. Build the docker image

``` shell
docker build -t YOLOv1 docker/
```

3. Docker run

``` shell
nvidia-docker run --name yolov1 -it -v /your/data/path:/data/ -v /your/projects/path:/YOLOv1/ YOLOv1
```

4. Install Requirements

``` shell
pip install -r requirements.txt
```



## Before Training

1. Prepare COCO format json

2. Modify Config.py


## Training

``` shell
python train.py --resume weights/path
```

## Evaluation

```shell
python val.py --weights weights/path --val-ann validation/json/path
```

### Output
```shell
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.590
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.402
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.219
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.409
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.481
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.310
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.495
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.521
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.323
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.560
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667
```

## Demo

1. Make a directory to save the results.

``` shell
mkdir demo
```

2. Run

```shell
python demo.py --weights weights/path --source [img, vid, folder]/path
```


