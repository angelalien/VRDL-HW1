# VRDL-HW1

Code for Selected Topics in NYCU Visual Recognition using Deep Learning(2021 Autumn) Homework1: Image Classification. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Dataset Preparation

Please download the dataset from [2021VRDL_HW1_datasets](https://competitions.codalab.org/my/datasets/download/83f7141a-641e-4e32-8d0c-42b482457836) and unzip it.
After downloading and unzip, the data directory is structured as:
```
2021VRDL_HW1_datasets
  +- training_images
    +- 0003.jpg
    +- 0008.jpg
    ...
  +- testing_images
    +- 0001.jpg
    +- 0002.jpg
    ...
  classes.txt
  training_labels.txt
  testing_img_order.txt
```

## Training 

To train the model, run this command:

```train
python train.py
```

> The python program file is put in the same level of 2021VRDL_HW1_datasets file.



## Pre-trained Models

You can download pretrained models here:

- [My ResNeXt model](https://drive.google.com/file/d/1f2qEE8EBa4MswRK1tpyeCg0uMaOOv166/view?usp=sharing) trained on given bird dataset.
  

Model's hyperparameter setting:

-  batch size = 32
-  epochs = 8
-  loss function: cross entropy
-  optimizer: SGD, momentum=0.9, weight_decay=1e-6
-  initial learning rate=0.01
-  learning scheduler: StepLR, step_size=5, gamma=0.1


## Make Submission

To make the submission file, run this command:

```inference
python inference.py --model model_resnext.pkl
```
> The model file is put in the same level of python program file.
