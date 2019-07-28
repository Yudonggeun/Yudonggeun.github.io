---
title: PyTorch Autograd
tags: 강화학습
key: page-PyTorch-Autograd
---

### AUTOGRAD

PyTorch의 모든 신경 네트워크의 중심은 autograd package입니다. autograd package는 Tensors의 작동에 대하여 자동 미분해줍니다. 이것은 define-by-run framework이며, 즉 코드가 실행되는 방식에 따라 backprop이 정의 되고 매번 바뀐다.

#### Tensor

torch.Tensor는 패키지의 중심 class입니다. `.requires_grad`를 True로 설정한다면 모든 작업을 추적하기 시작합니다. 계산이 끝나고 `.backward()`를 호출하면 모든 gradients를 자동으로 계산해줍니다. gradient는 tensor의 `.grad`에 쌓이게 될 것입니다.

Tensor가 히스토리를 추적하는 것을 막으려면 `.detach()`를 호출하여 계산 히스토리에서 이를 분리하고 미래의 계산을 추적하지 못하게 할 수 있습니다.

추적(메모리 사용 발생)을 방지하기 위해 `torch.no_grad():`로 코드 블록을 래핑 할 수도 있습니다. Model에는 `required_grad=True`를 사용하는 학습 가능한 매개변수가 있을 수 있지만 Gradient는 필요하지 않습니다.

autograd 구현을 위해 하나더 중요한 class가 있습니다. Function

Tensor와 Function은 서로 연결되어 있으며 완전한 계산 내역을 부호화하는 비순환 graph를 만듭니다. 각 tensor에는 Tensor를 작성한 함수를 참조하는 `.grad_fn` 속성이 있습니다. (`grad_fn`이 None이면 Tensors는 예외입니다.)

Tensors가 스칼라이면 `backward()`에 인수를 저장할 필요가 없지만 더 많은 요소가 있는 경우 일치하는 모양의 텐서 gradient 인수를 지정해야합니다.

```
import torch
```

텐서를 만들고 연산을 추적하기 위해 `requires_grad=True`로 설정

```
x = torch.ones(2, 2, requires_grad=True)

print(x)
```

Out:

```
tensor([[1., 1.], [1., 1.]], requires_grad=True)
```

텐서 조작

```
x = [torch.ones(2,](torch.ones(2,) 2, requires_grad=True)  
y = x + 2  
print(y)
```

Out:

```
tensor([[3., 3.], [3., 3.]], grad_fn=<AddBackward0>)
```

y는 작업 결과이고 `grad_fn`을 가집니다.

```
print(y.grad_fn)
```

Out:

```
<AddBackward0 object at 0x7f8033c4dac8>
```

다양한 y 조작 방법

```
x = [torch.ones(2,](torch.ones(2,) 2, requires_grad=True)  
y = x + 2  
z = y * y * 3  
out = z.mean()  

print(z, out)
```

Out:

```
tensor([[27., 27.], [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward1>)
```

`.requires_grad_(...)`는 기존 Tensor의 `requires_grad`를 현재 위치로 변경합니다. 입력된 flag가 False일 경우 값을 주지 않습니다.

```
a = [torch.randn(2,](torch.randn(2,) 2)  
a = ((a * 3) / (a - 1))  
print(a.requires_grad)  
a.requires\_grad_(True)  
print(a.requires_grad)  
b = (a * a).sum()  
print(b.grad_fn)
```

Out:

```
False
True
<SumBackward0 object at 0x7f8033c4dfd0>
```

#### Gradients

backprop 시작. `out`가 single scalar를 포함하고 있다면 `out.backward()`는 `out.backward(torch.tensor(1,))`와 동일합니다.

```
out.backward()
```

print gradients d(out)/dx

```
print(x.grad)
```

Out:

```
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

backprob에 대한 수식 생략...

이제 vector-Jacobian 예제를 살펴봅시다.

```
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000: # data.norm() == torch.sqrt(torch.sum(torch.pow(y, 2)))
    y = y * 2

print(y)
```

Out:

```
tensor([ 778.5364, -672.1349,  723.7890], grad_fn=<MulBackward0>)
```

이제 y는 더이상 scaler가 아닙니다. `torch.autograd`는 전체 Jacobian에 대해 계산할 수 없습니다. 만약 우리가 vector-Jacobian 곱을 원한다면 단순히 벡터를 인수로 역방향으로 전달하면됩니다:

```
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
```

Out:

```
tensor([2.0480e+02, 2.0480e+03, 2.0480e-01])
```

또한 `torch.no_grad ()`를 사용하여 코드 블록을 래핑하여 `.requires_grad = True`로 Tensors의 추적 기록에서 자동 시작을 중지 할 수 있습니다.

```
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
```

Out:

```
True
True
False
```