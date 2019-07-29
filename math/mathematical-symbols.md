---
layout: article
title: 수학 기호
key: page-sidebar-aside
permalink: /math/mathematical-symbols.html

aside:
  toc: true
sidebar:
  nav: layout_math
---

# 수학 기호

## 기초

### 집합

| 기호 |    의미     |               유래               |
| :--: | :---------: | :------------------------------: |
|  ℙ   |  소수 집합  |          Prime numbers           |
|  ℕ   | 자연수 집합 |         Natural numbers          |
|  ℤ   |  정수 집합  | (독일어) Zahlen -> 수들(numbers) |
|  ℚ   | 유리수 집합 |          Quotient -> 몫          |
|  ℝ   |  실수 집합  |           Real numbers           |
|  ℂ   | 복소수 집합 |         Complex numbers          |

* 포함관계: ℕ⊂ℤ⊂ℚ⊂ℝ⊂ℂ



#### Scalars, Vectors, Matrices, Tensors

![a](https://github.com/Yudonggeun/Deep-Learning-of-Deep-Learning/blob/master/Image/2.%20mathematical%20symbols/a.png?raw=true)

- 스칼라(Scalars): 그냥 숫자
  - ex) 실수 스칼라(real-valued scalar): s∈R 선의 기울기
  - ex) 정수 스칼라(natural number scalar): n∈N 단위의 갯수
- 영문 소문자 사용 $a$
- 벡터(Vectors): 숫자의 배열 형태
  - 배열의 순서가 중요하고 각각의 순서는 인덱스(index)로 구분된다.
  - 벡터는 굵은 영문 소문자 사용 $\mathbf{a}$
- 행렬(Matrices): 숫자의 2차원 배열
  - 배열과 마찬가지로 각각의 원소가 인덱스(index)로 구분된다.
- 굵은 영문 대문자 사용 $A$
- 텐서(*Tensors*): 3차원 이상의 배열 (벡터, 매트릭스의 일반화)
  - 굵은 영문 대문자 사용 $\mathbf{A}$



## 기호

#### 등호, 디비전 기호

|                기호                |    의미    |                             설명                             |            예시            |
| :--------------------------------: | :--------: | :----------------------------------------------------------: | :------------------------: |
|                $=$                 |    같다    |   $x=y$는 $x$ 와 $y$가 같은 수학 객체를 나타냄을 의미한다.   |          $2 = 2$           |
|               $\ne$                | 같지 않다  | $x \ne y$는 $x$ 와 $y$가 같은 수학 객체를 나타내지 않음을 의미한다. |       $2 + 2 \ne 5$        |
|             $\approx$              | 근사값이다 |      $x \approx y$는 $x$ 가 $y$의 근사값임을 의미한다.       |       $π ≈ 3.14159$        |
|              $\cong$               |  합동이다  |      △ABC ≅ △DEF는 삼각형 ABC는 삼각형 DEF와 합동이다.       |                            |
| $\Leftrightarrow, \leftrightarrow$ |    동치    | $A \Leftrightarrow B$는 $B$가 참이면 $A$는 참이고, $B$가 거짓이면 $A$도 거짓이다. | $x + 5 = y+ 2 ⇔ x + 3 = y$ |



#### 술어 논리

| 기호 |        의미        |                             설명                             | 예시                        |
| :--: | :----------------: | :----------------------------------------------------------: | --------------------------- |
|  ∀   |  모든 것에 대하여  | ∀ *x*: *P*(*x*)는 *P*(*x*)는 모든 *x*에 대하여 참이다를 의미한다. | ∀ *n* ∈ ℕ:*n*2 ≥ *n*.       |
|  ∃   | 존재한다; …이 있다 | ∃ *x*: *P*(*x*)는 *P*(*x*)가 참이기 위해서는 적어도 하나의*x* 가 존재하여야 한다는 의미이다. | ∃ *n* ∈ ℕ: *n*은 짝수이다.  |
|  ∃!  |      유일하다      | ∃! *x*: *P*(*x*)는 *P*(*x*)가 참이기 위해서는 오로지 하나의 *x*만 존재해야 한다는 의미이다. | ∃! *n* ∈ ℕ: *n* + 5 = 2*n*. |
