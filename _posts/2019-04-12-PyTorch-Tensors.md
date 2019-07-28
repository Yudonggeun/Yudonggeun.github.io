---
title: PyTorch Tensors
tags: PyTorch
key: page-PyTorch Tensors
---

#### Tensors

Tensors는 NumPy의 ndarrays와 유사하며 Tensors를 GPU에서도 사용할 수 있다.

```
from __future__ import print_function
import torch
```

  

초기화 되지 않은 5x3 행렬 생성

```
x = torch.empty(5, 3)
print(x)
```

Out:

```
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
```

  

랜덤으로 초기화 된 행렬을 만든다.

```
x = torch.rand(5, 4)
print(x)
```

Out:

```
tensor([[0.8541, 0.3420, 0.8181, 0.5187],
        [0.0324, 0.0796, 0.6898, 0.1944],
        [0.3299, 0.1840, 0.0843, 0.1516],
        [0.4275, 0.6507, 0.5878, 0.1182],
        [0.7116, 0.0855, 0.8491, 0.1673]])
```

  

행렬을 dtype long의 0으로 채운다.

```
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

Out:

```
tensor([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])
```

  

텐서에 데이터를 직접 넣을 수 있습니다.

```
x = torch.tensor([5.5, 3])
print(x)
```

Out:

```
tensor([5.5000, 3.0000])
```

#### Operations

Tensors를 다양한 방법으로 연산을 할 수 있습니다.

  

방법 1:

```
x = torch.zeros(5, 4)
y = torch.ones(5, 4)
print(x + y)
```

Out:

```
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
```

  

방법 2:

```
x = torch.zeros(5, 4)
y = torch.ones(5, 4)
print(torch.add(x, y))
```

Out:

```
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
```

  

출력 텐서를 인수로 제공할 수 있습니다.

```
x = torch.zeros(5, 4)
y = torch.ones(5, 4)
result = torch.empty(5, 4)
torch.add(x, y, out=result)
print(result)
```

Out:

```
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
```

  

행열 연산해서 행열에 추가하기

```
x = torch.zeros(5, 4)
y = torch.ones(5, 4)
y.add_(x)
print(result)
```

Out:

```
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
```

  

행렬에서 색인을 생성할 수 있습니다.

```
x = torch.ones(5, 4) print(x\[:, 1\])
```

Out:

```
tensor(\[-0.4377, 0.1942, 0.2410, 0.5899, 0.2724\])
```

```
x = torch.randn(4, 4) y = x.view(16) z = x.view(-1, 8) # -1은 행렬 사이즈에 맞춰서 변경됩니다.
print(x.size(), y.size(), z.size())
```

Out:

```
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

  

tensor가 하나의 요소만 가지고 있을 때 .item()으로 파이썬 숫자 값을 얻을 수 있습니다.

```
x = torch.randn(1) print(x) print(x.item())
```

Out:

```
tensor([1.3589])
1.3589212894439697
```

#### NumPy Array to Torch Tensor

tensor를 numPy 형식으로 변환 할 수 있습니다.

```
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

Out:

```
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

#### CUDA Tensors

Tensors를 .to 메소드를 이용해서 다른 디바이스로 옮길 수 있습니다.

```
# GPU가 있어야함.
if torch.cuda.is_available():
  device = torch.device("cuda")
  y = torch.ones_like(x, device=device)
  print(y)
  x = x.to(device)
  print(x)
  z = x + y
  print(z)
  print(z.to("cpu", torch.double))
```

Out:

```
tensor([1.], device='cuda:0')
tensor([0.6489], device='cuda:0')
tensor([1.6489], device='cuda:0')
tensor([1.6489], dtype=torch.float64)
```