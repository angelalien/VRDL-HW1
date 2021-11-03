# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import time
import os
import PIL.Image as Image

NUM_CLASSES = 200
TRAIN_SIZE = 2400
TEST_SIZE = 600
BATCH_SIZE = 32
DATASET_DIR = "./2021VRDL_HW1_datasets/"


def train_model(model, criterion, optimizer, scheduler, n_epochs=5):
    losses = []
    accuracies = []
    test_accuracies = []
    test_losses = []

    model.train()  # Set the model to train mode initially
    for epoch in range(n_epochs):

        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(train_loader, 0):

            # get the inputs and assign them to cuda
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate the loss/acc later
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_duration = time.time() - since
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 1 / BATCH_SIZE * running_correct / len(train_loader)
        print("【Epoch %s】duration: %d s" % (epoch + 1, epoch_duration))
        print("[train] loss: %.4f, acc: %.4f" % (epoch_loss, epoch_acc))

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        # Sswitch the model to eval mode to evaluate on test data
        model.eval()
        test_acc, test_loss = eval_model(model)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

        # Rre-set the model to train mode after validating
        model.train()
        scheduler.step()
        since = time.time()

    print("Finished Training")
    return model, losses, accuracies, test_accuracies, test_losses


def eval_model(model):
    correct = 0.0
    total = 0.0
    val_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            total += labels.size(0)
            val_loss += loss.item()
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    val_loss = val_loss / len(val_loader)
    print("[val]   loss: %.4f, acc: %.4f" % (val_loss, val_acc))
    return val_acc, val_loss


if __name__ == "__main__":

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read training image names and labels
    with open(DATASET_DIR + "training_labels.txt") as f:
        train_y = [int(x.strip().split()[1][:3]) - 1 for x in f.readlines()]
    with open(DATASET_DIR + "training_labels.txt") as f:
        train_x_name = [line.strip().split()[0] for line in f.readlines()]

    # Pre-process method of images for train and validation dataloader
    train_tfms = transforms.Compose(
        [
            transforms.Resize((256)),
            transforms.CenterCrop(224),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Read training images and split to trainig and validation images
    image_index = 0
    if not (os.path.isfile("./train_x.npy")):
        train_images = []
        val_images = []
        for name in train_x_name:
            if image_index < 2400:
                train_image = train_tfms(
                    Image.open(
                        DATASET_DIR + "training_images/" + name
                    ).convert("RGB")
                )
                train_image = train_image.numpy()
                train_images.append(train_image)
            else:
                val_image = val_tfms(
                    Image.open(
                        DATASET_DIR + "training_images/" + name
                    ).convert("RGB")
                )
                val_image = val_image.numpy()
                val_images.append(val_image)
            image_index += 1

        train_images = np.array(train_images)
        val_images = np.array(val_images)
        np.save("train_x.npy", train_images)
        np.save("val_x.npy", val_images)
    else:
        train_images = np.load("./train_x.npy")
        val_images = np.load("./val_x.npy")
    train_x = torch.from_numpy(train_images)
    val_x = torch.from_numpy(val_images)

    # Split training and validation labels
    val_y = train_y[2400:]
    train_y = train_y[:2400]
    val_y = torch.LongTensor(val_y)
    train_y = torch.LongTensor(train_y)

    # Prepare for training and validation dataloader
    train_dataset = Data.TensorDataset(train_x, train_y)
    val_dataset = Data.TensorDataset(val_x, val_y)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Use pretrained model and modified the last layer
    model = models.resnext101_32x8d(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    # Hyperparameter setting
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6
    )
    lrscheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Start training
    (
        model,
        training_losses,
        training_accs,
        test_accs,
        test_losses,
    ) = train_model(model, criterion, optimizer, lrscheduler, n_epochs=8)

    # Save model
    torch.save(model.state_dict(), "trained_model.pkl")
