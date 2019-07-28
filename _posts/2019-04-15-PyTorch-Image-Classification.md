---
title: PyTorch Image Classification
tags: PyTorch
key: page-PyTorch-Image-Classification
---

## Traning a classifier

어떻게 nearal network를 만드는지, loss 계산과 network의 weight를 업데이트하는 법을 알게되었다.

### data란?

일반적으로 이미지, 문자, 소리나 동영상 데이터를 다룰 때 표준 python packages를 사용하여 데이터를 numpy array로 불러올 수 있습니다. 그런 다음 배열을 torch.\*Tensor로 변환할 수 있습니다.

-   이미지의 경우 Pillow, OpenCV
-   오디오의 경우 scipy, librosa
-   문자의 경우 NLTK, SpaCy

특히 vision을 위해 Imagenet, CIFAR10, MNIST 등과 같은 공통 데이터셋 및 이미지 용 데이터 변환기, torchvision.datasets 및 torch.utils.data.DataLoader에 대한 데이터 loader가 있는 torchvision이라는 패키지가 존재합니다.

이것은 큰 편의를 제공하고 boilerplate code 작성을 피합니다.

이번 튜토리얼에서는 CIFAR10 dataset을 사용할 것 입니다. 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭 class를 가지고 있다. CIFAR-10의 이미지 크기는 3x32x32입니다. 즉, 크기가 32x32 픽셀이고 3채널 컬러 이미지입니다.

![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2019/15/a.png)

### Traning an image classifier

다음 순서를 따를 것입니다:

1.  torchvision을 사용하여 CIFAR10 datasets  load 및 normalizing
2.  CNN(Convolutional Neural Network)  구축
3.  loss function 구축
4.  traning data로 network 학습
5.  test data로 network 평가

### 1. Loading and normalizing CIFAR 10

torchvision을 사용하여 매우 쉽게 CIFAR10을 불러옵니다.

```
import torch
import torchvision
import torchvision.tranforms as transforms
```

torchvision datasets의 출력은 \[0, 1\]범위의 PILImage 이미지입니다. normalized \[-1, 1\]의 Tensors로 변환합니다.

```
import os
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                         shuffle=True, num_workers=os.cpu_count())

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=os.cpu_count())

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

Out:

```
0it [00:00, ?it/s]Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
100%|█████████▉| 169959424/170498071 [00:10<00:00, 7886204.55it/s]Files already downloaded and verified
```

다운 받은 이미지를 확인해봅시다.

```
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

Out:

[##_Image|kage@bP3mI0/btquxsUvfXM/tWl3K3phI0icjHGCP7PEV0/img.png|alignCenter|data-filename="Screen Shot 2019-04-13 at 8.57.08 PM.png"|_##]

### 2. Define a CNN(Convolutional Neural Network)

Neural Network는 전 섹션에서 복사한 다음 3채널 이미지로 수정합니다.

```
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

### 3. Define a Loss function and optimizer

Classification Cross-Entropy loss와 SGD를 사용합니다.

```
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

### 4. Train the network

data를 network에 넣고 최적화 과정을 반복해야합니다.

```
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))

print('Finished Training')
```

Out:

```
Files already downloaded and verified
Files already downloaded and verified
[1,  2000] loss: 2.192
[1,  4000] loss: 4.016
[1,  6000] loss: 5.663
[1,  8000] loss: 7.209
[1, 10000] loss: 8.714
[1, 12000] loss: 10.175
[2,  2000] loss: 1.396
[2,  4000] loss: 2.758
[2,  6000] loss: 4.098
[2,  8000] loss: 5.423
[2, 10000] loss: 6.721
[2, 12000] loss: 7.993
Finished Training
```

### 5. Test the network on the test data

training dataset을 사용하여 network를 2번 학습시켰습니다. 그 다음 network가 얼만큼 학습을 했는지 확인해야합니다.

neural network의 출력 값과 실제 값을 비교하여 확인할 것입니다. 예측이 정확하면 샘플을 올바른 예측 목록에 추가합니다.

첫 번째로 test set의 이미지를 표시합니다.

```
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

[##_Image|kage@bKNvVV/btquw3Vx176/dvFy6sLK9RBvQj7rxra2Q1/img.png|alignCenter|data-filename="sphx_glr_cifar10_tutorial_002.png"|GroundTruth: cat ship ship plane_##]

이제 neural network가 위 사진을 보고 무엇인지 예측할 것입니다.

```
outputs = net(images)
```

츌력은 10개의  클래스에 대한 에너지입니다. class의 에너지가 가장 높을 수록 network는 이미지가 특정 class에 속한다고 생각합니다.

```
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
```

Out:

```
Predicted:    cat  ship truck plane
```

결과가 매우 좋다.

network의 평가 방법을 자세하게 알아보겠습니다.

```
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

Out:

```
Accuracy of the network on the 10000 test images: 55 %
```

정확도는 55%입니다.

그러면 class별로도 정확도 차이가 있는지를 확인해봅시다.

```
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```

Out:

```
Accuracy of plane : 58 %
Accuracy of   car : 73 %
Accuracy of  bird : 46 %
Accuracy of   cat : 46 %
Accuracy of  deer : 44 %
Accuracy of   dog : 31 %
Accuracy of  frog : 72 %
Accuracy of horse : 51 %
Accuracy of  ship : 61 %
Accuracy of truck : 59 %
```

### Training on GPU

Tensor를 GPU로 옮기는 것과 마찬기지로 neural net도 GPU로 옮겨야합니다.

GUDA를 사용할 수 있는 경우 첫 번째 보이는 cuda 장치로 설정합니다.

```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
```

Out:

```
cuda:0
```

다음 메소드는 모든 모듈을 재귀적으로 거쳐 매개 변수와 버퍼를 CUDA 텐서로 변환합니다.

```
net.to(device)
```

모든 단계의 input과 target을 GPU에 보내야한다.

```
inputs, labels = inputs.to(device), labels.to(device)
```