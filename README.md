**Dynamic DETR: End-to-End Object Detection with Dynamic Attention**
========


This is the CCBDA final project, NYCU, 2022 Fall which is an PyTorch implementation for **Dynamic DETR**.


For paper's details see [Dynamic DETR: End-to-End Object Detection with Dynamic Attention](https://openaccess.thecvf.com/content/ICCV2021/papers/Dai_Dynamic_DETR_End-to-End_Object_Detection_With_Dynamic_Attention_ICCV_2021_paper.pdf) by Xiyang Dai, Yinpeng Chen, Jianwei Yang, Pengchuan Zhang, Lu Yuan, Lei Zhang


## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train Dynamic DETR on a single gpu with 12 epochs:
```
python main.py --output_dir [./DIR] --epochs 12 --lr_drop 11 --coco_path /path/to/coco 
```
We train Dynamic DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation. 
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.


## Evaluation
To evaluate Dynamic DETR:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume /path/to/checkpoint --coco_path /path/to/coco
```

## Distributed training
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```
