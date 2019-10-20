---
title: PyTorch Neural Network
tags: PyTorch
key: page-PyTorch-Neural-Network
---

### Neural Network
신경망(Neural Network)는 `torch.nn` 패키기를 사용하는 구조입니다.

nn은 모델을 정의하고 차별화하기 위해 autograd에 의존합니다. `nn.Module`은 layers를 포함하고 method `forward(input)`에 대해 `output`을 반환합니다.

다음 이미지를 구별하는 network를 보자:

![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2019/04/14/a.png)

이것은 simple feed-forward network이다. 입력을 받아 여러 layer를 차례로 통과한 다음 최종적으로 값을 출력해준다.

다음은 신경망의 전형적인 학습 순서이다:

- 신경망에 사용되는 학습 가능한 parameters를 정의한다.(또한 weights)
- 입력 데이터 집합 반복
- network를 통한 입력 프로세스
- loss 계산(결과가 얼마나 정확한지)
- gradient를 뒤 network's parameters에게 전달하여 미분
- network의 weights를 업데이트, 전형적으로 간단한 update rule을 사용한다: `weight = weight - learning_rate * gradient`

### Define the network
network 정의

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(1, 6 ,5)
    self.conv2 = nn.Conv2d(6, 16, 5)

    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def num_flat_features(self, x):
     size = x.size()[1:]
     num_features = 1
     for s in size:
         num_features *= s
     return num_features

net = Net()
print(net)
```

Out:

```
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

forward 함수를 정의하기만 하면 되며, 역함수(gradients 계산)는 autograd를 사용하면 자동으로 정의된다. forward function에서 모든 Tensor 작업을 사용할 수 있다.

```
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```

Out:

```
10
torch.Size([6, 1, 5, 5])
```

32x32 input size로 만들어봅시다.

```
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

Out:

```
tensor([[-0.1237,  0.0473,  0.1735, -0.0477,  0.0033, -0.0246,  0.0988,  0.0485,
          0.0210, -0.0180]], grad_fn=<AddmmBackward>)
```

모든 매개변수의 gradient buffer를 0으로 설정하고, 무작위 값으로 역전파를 합니다:

```
net.zero_grad()
out.backward(torch.randn(1, 10))
```

`torch.nn`은 mini-batches 형식만 지원한다. 예를 들어 `nnConv2d`는 `nSamples x nChannels x Height x Width`의 4차원 Tensor를 입력으로 한다. 만약 하나의 샘플만 있다면 `input.unsqueenze(0)`을 사용해서 가짜 차원을 추가한다.

#### Recap:

-   `torch.Tensor` - `backward()`와 같은 autograd 작업을 지원하는 다차원 배열. 또한 tensor는 gradient w.r.t도 보유합니다.
-   `nn.Module` - neural network 모듈. parapeters를 캡슐화하는 편리한 방법, GPU로 이동, 내보내기, 불러오기 등의 작업을 수행하는 도우미 역할.
-   `nn.Parameter` - Tensor의 일종으로 모듈에 속성으로 지정되면 parameters로 자동 등록됩니다.
-   'autograd.Function' - autograd 조작의 forward 정의를 구현합니다. 모든 Tensor 작업은 Tensor를 작성하고 해당 기록을 인코딩하는 함수에 연결하는 하나 이상의 Function 노드를 만듭니다.

#### 지금까지 해결한 것

-   신경망 정의
-   input과 backward 호출

#### 앞으로 남은 것

-   loss 계산
-   network의 weight 업데이트

### Loss Function

loss function은 (output, target) 입력 쌍을 가져와서 대상에서 얼마나 멀리 떨어져 있는지 추정하는 값을 계산합니다. nn 페키지에는 여러가지 loss 기능이 있습니다. 제곱 평균 계산하여 input과 target 사이의 간단한 loss계산할 수 있습니다: `nn.MSELoss`

예제:

```
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

Out:

```
tensor(0.9736, grad_fn=<MseLossBackward>)
```

이제 `.grad_fn` 속성을 사용하여 역방향으로 손실을 추적하면 다음과 같은 계산 graph가 표시된다. 다음과 같은 계산 graph를 볼 수 있습니다:

```
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

`loss.backward()`를 호출할 때, 전체 Graph가 차등화됩니다. 손실 및 `requires_grad = True`인 Graph의 모든 Tensor는 Gradient로 누적 된 .grad Tensor를 갖습니다.

예를 들어 전 단계를 확인해봅시다:

```
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
```

Out:

```
<MseLossBackward object at 0x7f30a52d0630>
<AddmmBackward object at 0x7f30a52cfd30>
<AccumulateGrad object at 0x7f30a52d0630>
```

### Backprop

오차를 역전파하려면 `loss.backward()`에 대한 작업만 수행하면 됩니다. gradients가 기존 gradients에 누적되면 기존 gradient를 지워야한다.

이제는 `loss.backward()`를 호출하고 conv1의 뒤쪽 전후에 bias gradients를 살펴보겠습니다.

```
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

Out:

```
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([-0.0118,  0.0095, -0.0087, -0.0028,  0.0098,  0.0052])
```

어떻게 loss function을 사용하는지 알게되었다.

### Update the weights

실제로 사용되는 가장 단순한 업데이트 방법은 Stochastic Gradient Descent(SGD)이다.  
`weight = weight - learning_rate * gradient`

간단한 파이썬 코드를 사용하여 구현할 수 있습니다.

```
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

그러나 neural networks를 사용할 때 SGD, Nesterov-SGD, Adam, RMSProp 등과 같은 다양한 다른 업데이트 방법을 사용하려고 합니다. 이 기능을 사용하려면 `torch.optim`를 사용하면됩니다.

```
import torch.optim as optim

# optimizer 제작
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()   # gadient buffers 0으로 초기화
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

Backprop 섹션에서 설명한 것처럼 gradient가 누적되기 때문에 `optimizer.zero_grad()`를 사용하여 gradient buffer를 수동으로 0으로 설정한다.