import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.datasets import ImageFolder


# 학습을 위한 데이터 증가(Augmentation)와 일반화하기
# 단지 검증을 위한 일반화하기
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4704,0.4534,0.4571),(0.1929,0.2107,0.2006))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4704,0.4534,0.4571),(0.1929,0.2107,0.2006))
])

batch_size = 64

data_dir = '/Users/jusuklee/PycharmProjects/Introduction_AI/face_dataset'
train_set = ImageFolder(root='/Users/jusuklee/PycharmProjects/Introduction_AI/face_dataset/facescrub_train', transform= transform_train)
test_set = ImageFolder(root='/Users/jusuklee/PycharmProjects/Introduction_AI/face_dataset/facescrub_test',transform = transform_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
for data in train_loader:
    inputs = data[0]
    classes = data[1]

class_names = train_set.classes
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
#change conv1 stride into (1,1)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
temp = model.conv1.weight  # store weight
model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
model.conv1.weight = temp  # restore weight

temp = model.layer2[0].conv1.weight
model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.layer2[0].conv1.weight = temp

temp = model.layer2[0].downsample[0].weight
model.layer2[0].downsample[0] = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.layer2[0].downsample[0].weight = temp

model.eval()
