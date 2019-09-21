---
title: How to measure performance of object detection
tags: object-detection
key: how-to-measure-performance-of-object-detection
---

## Object detection의 성능 평가 방법

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/05/a.jpg?raw=true)

어떠한 논문이든 결과를 도출해 낸다. 과거에 비해 얼만큼 성능이 향상되었는지 말이다. 간단한 classification문제는 정답을 맞춘 수를 전체 수로 나누면 정확도가 계산된다. 그러나 Object detection은 bounding box, classification 두가지를 동시에 평가해야한다.

## IoU(Intersection Over Union)
IoU는 Bouning Box를 얼마나 잘 예측했는지를 측정한다. ground-truth bounding-box와 Predicted bounding-box의 교집합을 합집합으로 나눈다. 결국에는 얼마나 겹치는 부분이 있는지를 통해 측정한다.

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/05/b.png?raw=true)


보통 IoU가 0.5 이상이면 정답이라고 판단한다.

## Recall & Precision
TP(True Positive): True를 True라고 판별.
FP(False Positive): False를 True라고 판별
TN(True Negative): True를 False라고 판별
FN(False Negative): False를 False라고 판별

Recall은 검출률을 뜻한다. 얼마나 잘 잡아내는지를 나타낸다.


recall = TP / (TP + FN)

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/05/c.png?raw=true)

precision은 정확도를 말한다. 검출된 결과가 얼마나 정확한지를 나타낸다.

precision = TP / (TP + FP)

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/05/d.png?raw=true)

인식 알고리즘을 고정된 값으로 알고리즘의 성능을 평가하는 것은 잘못되었다. 왜냐하면 recall과 precision은 알고리즘의 파라미터 조절에 따라 유동적으로 변화는 값이기 때문이다. 일반적으로 알고리즘의 recall과 precision은 서로 반비례 관계를 가진다. 따라서 인식 알고리즘들의 성능을 제대로 비교하기 위해서는 precision과 recall의 전체 성능 변화를 살펴봐야 한다.

## PR(Precision Recal) 곡선
PR 곡선은 confidence 레벨에 대한 threshold값의 변화에 의한 물체 검출기의 성능을 평가하는 방법이다. confidence 레벨은 검출한 것에 대해 알고리즘이 얼마나 정확한지를 알려주는 값이다.

confidence 레벨이 높다고 해서 무조건 정확하다는 것은 아니다. confidence 레벨이 낮으면 그만큼 검출 결과에 대해 자신이 없는 것이다. 따라서 confidence 레벨에 대해 threshold 값을 부여해서 특정값 이상이 되어야 검출된 것으로 인정한다. threshold보다 cofidence 레벨이 작은 검출은 무시된다.

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/05/e.png?raw=true)

10개중 7개가 제대로 검출되었다.
precision = 옳게 검출된 얼굴 갯수 / 검출된 얼굴 갯수 = 7/10
recall = 옭게 검출된 얼굴 갯수 / 실제 얼굴 갯수 = 7/15

다음은 검출된 결과를 confidence 레벨에 따라 재정렬한다.

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/05/f.png?raw=true)

confidence 레벨에 대한 threshold 값을 95%로 하면 하나만 검출한 것으로 판단할 것이고 이 때 precision = 1/1 = 1, recall = 1/15 = 0.067이 된다. threshold 값을 검출들의 confidence 레벨에 맞춰 낮춰가면 다음과 같이 precision과 recall이 계산된다.

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/05/g.png?raw=true)

이 precision 값들과 recall값들을 아래와 같이 그래프로 나타내면 이것이 PR 곡선이다.

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/05/h.jpeg?raw=true)

## AP(Average Precision)
PR 곡선은 성능을 평가하는데 매우 좋은 방법이지만 단 하나의 숫자로 성능을 평가할 수 있다면 훨씬 더 좋을 것이다. 그래서 AP를 사용한다 AP는 precision recall 그래프에서 그래프 선 아래쪽의 면적으로 계산된다.

넓이를 계산하기 쉽게 곡선을 직선으로 변환시킨다.

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/05/i.png?raw=true)

이제 선 아래의 넓이를 계산하여 AP를 구한다. AP = 왼쪽 큰 사각형의 넓이 + 오른쪽 작은 사각형의 넓이 = 1 * 0.33 + 0.88 * (0.47 - 0.33) = 0.4532

## mAP(mean Average Precision)
AP에서는 class가 하나였지만 여러개 있을 경우에는 mAP를 사용한다. 각 class별로 AP를 구한 후 전부 합한다. 그리고 평균을 구하면 끝이다.

## 참고한 자료
https://bskyvision.com/465