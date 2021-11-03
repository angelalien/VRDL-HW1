# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import PIL.Image as Image

NUM_CLASSES = 200
BATCH_SIZE = 32
DATASET_DIR = "./2021VRDL_HW1_datasets/"

# Main function
if __name__ == "__main__":

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    model = models.resnext101_32x8d(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)
    model.load_state_dict(torch.load("./model_resnext.pkl"))

    # Read image classes
    with open(DATASET_DIR + "classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # Read test images name for predicting
    with open(DATASET_DIR + "testing_img_order.txt") as f:
        predicted_names = [line.strip() for line in f.readlines()]

    # Pre-process test images for test dataloader
    test_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if not (os.path.isfile("./predicted_images.pt")):
        predicted_images = []
        for name in predicted_names:
            image = test_tfms(
                Image.open(DATASET_DIR + "testing_images/" + name).convert(
                    "RGB"
                )
            )
            predicted_images.append(image)
        torch.save(predicted_images, "predicted_images.pt")
    else:
        predicted_images = torch.load("./predicted_images.pt")

    test_loader = Data.DataLoader(
        dataset=predicted_images, batch_size=1, shuffle=False, num_workers=0
    )

    # Make submission
    submission = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            outputs = model(data.to(device))
            _, predicted_class = torch.max(outputs.data, 1)
            submission.append([predicted_names[i], classes[predicted_class]])
    np.savetxt("answer.txt", submission, fmt="%s")
