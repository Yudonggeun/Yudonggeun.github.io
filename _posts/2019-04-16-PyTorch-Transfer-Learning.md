---
title: PyTorch Transfer Learning
tags: PyTorch
key: page-PyTorch-Transfer-Learning
---

## Transfer Learning

기존에 만들어진 모델을 사용하여 새로운 모델을 만들시 학습을 빠르게 하며, 예측을 더 높이는 방법이다. 더 자세한 내용은 [cs231n notes](https://cs231n.github.io/transfer-learning/) 참고.

인용:

실제로, 충분한 양의 dataset을 갖는 것은 상대적으로 드물기 때문에 전체 CNN을 처음부터 (랜덤 초기화를 통해) 훈련시키는 사람은 거의 없습니다. 그 대신 매우 큰 데이터 세트 (예 : 1000 개의 카테고리가있는 120 만 개의 이미지가 포함 된 ImageNet)에서 ConvNet을 사전 트레인 한 다음 해당 작업에 대한 초기화 또는 고정 특징 추출기(fixed feature extractor)로 ConvNet을 사용하는 것이 일반적입니다.

중요한 두가지 transfer learning 시나리오:

- **Finetuning the convnet**: 랜덤 초기화 대신 imagenet 1000 데이터 세트에서 교육받은 네트워크와 같은 사전 네트워크를 사용하여 네트워크를 초기화합니다. 학습의 나머지 부분은 평소와 같습니다.
- **ConvNet as fixed feature** **extractor**: 여기에서는 마지막 fully connected layer를 제외한 전체 네트워크의 가중치를 고정합니다. 마지막 fully connected layer만 새로운 랜덤 가중치로 교체되고 이 마지막 layer만 학습됩니다.

```
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode
```

### Load Data

data를 불러올 때 torchvision과 torch.utils.data를 사용합니다.

우리가 해결할 문제는 개미와 벌을 분류하는 것입니다. 우리는 학습을 위한 개미와 벌 이미지 120개를 가지고 있습니다. 테스트를 위한 75개의 이미지도 있습니다. 일반적으로 처음부터 교육을 받으면 generalize할 수 있는 아주 작은 데이터 세트입니다. 우리가 transfer learning을 사용하고 있기 때문에 우리는 쉽게 합리적으로 generalize할 수 있습니다.

[data download](https://download.pytorch.org/tutorial/hymenoptera_data.zip)

```
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### Visualize a few images

training images중 몇가지의 사진을 확인해봅시다.

```
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
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```

![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2019/16/a.png)

### Training the model

모델을 일반적인 함수를 사용하여 작성합니다. 설명:

-   Learning rate 설정
-   가장 좋은 모델 저장

```
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
```

### Visualizing the model predictions

약간의 이미지 정확도를 표시하는 함수.

```
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```

### Finetuning the convnet

미리 학습된 모델을 불러오고 마지막 fully connected layer부분만 초기화 합니다.

```
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

Train and evaluate

CPU에서는 15-25분정도 소요되고 GPU에서는 몇 분이면 충분하다.

```
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
```

Out:

```
Epoch 0/24
----------
train Loss: 0.5169 Acc: 0.7459
val Loss: 0.2326 Acc: 0.9020

.
.
.

Training complete in 1m 8s
Best val Acc: 0.921569
```

```
visualize_model(model_tf)
```

![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2019/16/b.png)

### ConvNet as fixed feature extractor

여기서 마지막 레이머를 제외하고 모든 네트워크를 고정시켰습니다. requires\_grad == False 설정을 하면 파라미터가 고정되고 gradients는 backward()에서 개산되지 않습니다.

```
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

### Train and evaluate

CPU에서는 이전보다 절반의 시간이 소요됩니다. 이는 대부분의 네트워크에서 gradient 연산이 필요하지 않기 때문입니다. 하지만, forward의 연산은 필요합니다.

```
model_conv = train_model(model_conv, criterion, optimizer_conv,
						exp_lr_scheduler, num_epochs=25)
```

Out:

```
Epoch 0/24
----------
train Loss: 0.5947 Acc: 0.7131
val Loss: 0.2794 Acc: 0.8824

.
.
.

Training complete in 0m 35s
Best val Acc: 0.960784
```

```
visualize_model(model_conv)

plt.ioff()
plt.show()
```

![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2019/16/c.png)
