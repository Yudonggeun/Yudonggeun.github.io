---
title: Observation 살펴보기
tags: 강화학습
key: page-Observation-Review
---


## Observation 살펴보기
다음 GitHub에서 test.py 파일을 디버깅하며 observation에 어떤 정보가 들어가 있는지를 알아볼 것이다. PyCharm 환경에서 진행하였다.

Test.py 파일을 열고 38 줄에서 숫자 옆 빈공간을 누르게 되면 빨간색 원이 생기며 이 라인에서 멈추게 된다:
![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2018/18/a.png)

디버깅:
![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2018/18/b.png)

위와 같이 디버깅이 되며 obs->0 까지 펼치면 observation이 보인다:
![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2018/18/c.png)

observation을 펼쳐보면:

![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2018/18/d.png)

이와 같이 17개 종류가 나온다. 예를 들어 single_select를 펼져보면:
![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2018/18/e.png)

여기서 single_select에 어떤 내용이 들어가는지를 살펴 보기 위해서 오른쪽으로 밀어 보면 
~~~
[‘unit_type’, ‘player_relative’, ‘health’, ’shields’, ‘energy’, ’transport_slots_taken’, ‘build_progress']
~~~

이런 배열을 볼 수 있을 것이다. Single_select가 7개의 배열로 이루어져 있고 각 위치가 어떤 정보를 가지고 있는지를 알려준다.

[0:1]를 들어가보면 1차원 배열로 이루어저 있음을 알 수 있고 현재 어떤 값이 들어가 있는지를 알 수 있다. 지금은 선택된게 없기 때문에 [0 0 0 0 0 0 0] 배열을 나타낸다:
![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2018/18/f.png)

feature_screen -> [0:17] -> 05 -> [0:84] 를 들어가보면:
![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2018/18/g.png)

위와 같이 0, 1, 4로 이루어져 있다. 현재 게임 상태와 비교해보면:
![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2018/18/h.png)

4가 있는 위치는 적 유닛이 있고, 1이 있는 위치에는 아군 유닛이 있음을 확인 할 수 있다.

이렇게 디버깅을 하며 우리가 상황에 따라 필요한 observation을 찾을 수 있을 것이다.