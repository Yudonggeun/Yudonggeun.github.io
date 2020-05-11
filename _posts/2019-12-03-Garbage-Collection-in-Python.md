---
title: Garbage Collection in Python
tags: book-report
key: garbage-collection-in-python
---

![출처: https://stackify.com/python-garbage-collection/](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/12/a.jpg?raw=true)

## 서론
### GC(Garbage Collector)
GC는 현대적인 언어에는 거의 필수로 존재하며 개발자의 생산성을 향상해준다. C#, JS, Python 등의 언어는 GC를 기본적으로 제공하며, C, C++과 같은 언어에서는 malloc(), free()와 같은 저수준의 메모리 관리 함수를 제공한다. (물론 C, C++에서도 GC를 library 형태로 사용할 수 있다. libgc) 실제로 현대적인 언어로 넘어오면서 개발자는 메모리를 직접 관리하는 코드를 거의 사용하지 않게 되었다.

### GC를 공부하는 이유
GC는 메모리를 자동으로 관리해주는 ‘과정’이다. 당연히 자동으로 메모리를 관리해 주니 사람이 직접 하는 것보다는 최적화가 덜 되어있다. 그래서 공부를 하고 업무에 적용해야 한다는 것이다. 실제로 인스타그램은 Python GC를 사용하지 않는다. ([Instagram이 Python garbage collection 없앤 이유 참고](https://b.luavis.kr/python/dismissing-python-garbage-collection-at-instagram))


굳이 기준을 나누자면, 동기적인 코드(Computer Science, Machine Learning 등) 는 메모리 관리를 크게 신경 쓰지 않아도 되지만 비동기적인 코드(Cloud, DB, Backend, Frontend)는 메모리 관리에 노력해야 한다.
성능이 좋아야 하는 장기(long-running) 프로그램의 경우, 일부 언어에는 여전히 수동 메모리 관리 기능을 사용한다. C++은 물론이고 macOS, iOS에 사용되는 언어인 Objective-C에서는 수동으로 메모리 관리를 한다.

### 기존 메모리 관리의 문제점
현대적인 언어가 아닌 과거 언어인 경우, 메모리 관리를 직접 해줘야 하는 언어들은 크게 두 가지 문제점을 가지고 있다.

1. 필요 없는 메모리를 비우지 않았을 때: 메모리 사용을 마쳤을 때 비우지 않을 경우 메모리 누수가 발생할 수 있고 장기적인 관점에서 심각한 문제가 발생할 수 있다.
2. 사용중인 메모리 비우기: 존재하지 않는 메모리에 접근하려고 하면 프로그램이 중단되거나 메모리 데이터 값이 손상될 수 있다.

이러한 문제를 해결하기 위해 현대적인 언어는 자동 메모리 관리(Automatic Memory Management)를 갖추게 되었다.

## 본론
### Python의 Garbage Collection 구현
Python에서 Garbage Collection이 어떻게 작동하는지를 알아본다.

CPython에서의 메모리 관리와 Garbage Collection은 두 가지 측면이 있다.

1. 레퍼런스 카운팅(Reference counting)
2. 세대별 가비지 컬렉션(Generational garbage collection)

### CPython의 Reference Counting
CPython에서의 주요 garbage collection mechanism은 reference counts 방식이다. Python에서 객체를 만들 때마다 기본 C 객체에서는 Python 유형(list, dict 또는 function)과 reference count가 생성된다.

매우 기본적으로 Python 객체의 reference count는 객체가 참조될 때마다 증가하고 객체의 참조가 해제될 때 감소한다. 객체의 reference count가 0이 되면 객체의 메모리 할당이 해제된다.

### Python의 Reference Counting
Python standard library의 sys 모듈을 사용하여 특정 객체의 reference counts(참조 횟수)를 확인할 수 있다. 참조 횟수를 증가시키는 방법은
변수에 객체 할당.

list에 추가하거나 class instance에서 속성으로 추가하는 등의 data structure에 객체 추가.
객체를 함수의 인수로 전달.

Python REPL과 sys module로 참조 횟수를 확인할 수 있다. 변수를 만들고 참조 횟수를 확인해보자. (IDE를 사용하거나 환경에 따라 값이 다르게 출력될 수 있다.)

~~~
>>> import sys
>>> a = 'hello'
>>> sys.getrefcount(a)
2
~~~

왜 참조 횟수가 2인가? 하나는 variable을 생성하는 것이고 두 번째는 변수 a를 sys.getrefcount() 함수에 전달할 때이다.
variable을 data structure 각 list 또는 dictionary에 추가하면 참조 횟수가 증가한다.

~~~
>>> import sys
>>> a = 'hello'
>>> sys.getrefcount(a)
2
>>> b = [a]
>>> sys.getrefcount(a)
3
>>> c = {'first': a}
>>> sys.getrefcount(a)
4
~~~

위 코드처럼 list나 dictionary에 추가될 때 a의 참조 횟수가 증가한다.

### Generational Garbage Collection
Python은 메모리 관리를 위한 reference counting 외에도 generational garbage collection(세대별 가비지 컬렉션)이라는 방법을 사용한다. reference counting이 주로 사용되는 방법이고 보조로 가비지 컬렉션을 사용한다는 것이다.
왜 가비지 컬렉션이 필요한가? 전에 객체를 배열이나 객체에 추가하면 참조 횟수가 증가하는 것을 확인할 수 있었다. 그러나 객체가 순환 참조하면 어떻게 작동 될까? (순환 참조는 객체가 자기 자신을 가르키는 것을 말한다.)

~~~
>>> a = []
>>> a.append(a)
>>> del a
~~~

a의 참조 횟수는 1이지만 이 객체는 더 이상 접근할 수 없으며 레퍼런스 카운팅 방식으로는 메모리에서 해제될 수 없다.
또 다른 예로는 서로를 참조하는 객체이다.

~~~
>>> a = Func_pr() # 0x01
>>> b = Func_pr() # 0x02
>>> a.x = b # 0x01의 x는 0x02를 가리킨다.
>>> b.x = a # 0x02의 x는 0x01를 가리킨다.
# 0x01의 레퍼런스 카운트는 a와 b.x로 2다.
# 0x02의 레퍼런스 카운트는 b와 a.x로 2다.
>>> del a # 0x01은 1로 감소한다. 0x02는 b와 0x01.x로 2다.
>>> del b # 0x02는 1로 감소한다.
~~~

마지막 상태에서 0x01.x와 0x02.x가 서로를 참조하고 있기 때문에 레퍼런스 카운트는 둘 다 1이지만 0에 도달할 수 없는 garbage(쓰레기)가 된다.
이러한 유형의 문제를 reference cycle(참조 주기)이라고 하며 reference counting으로 해결할 수 없다.

### Generational Garbage Collector
generational garbage collector(세대별 가비지 컬렉터)와 이해해야 할 두 가지 주요 개념이 있다. 이 장은 “아~ 이런 게 있구나~”라고 대충 보고 넘기고 다음 장에서 실습해보며 증명해보자.

첫 번째는 generation(세대) 개념이다.
가비지 컬렉터는 메모리의 모든 객체를 추적한다. 새로운 객체는 1세대 가비지 수집기에서 life(수명)를 시작한다. Python이 세대에서 가비지 수집 프로세스를 실행하고 객체가 살아남으면, 두 번째 이전 세대로 올라간다. Python 가비지 수집기는 총 3세대이며, 객체는 현재 세대의 가비지 수집 프로세스에서 살아남을 대마다 이전 세대로 이동한다.
두 번째 핵심 개념은 threshold(임계) 값이다. 각 세대마다 가비지 컬렉터 모듈에는 임계값 개수의 개체가 있습니다. 객체 수가 해당 임계값을 초과하면 가비지 콜렉션이 콜렉션 프로세스를 trigger(추적) 합니다. 해상 프로세스에서 살아남은 객체는 이전 세대로 옮겨진다.
가비지 컬렉터는 내부적으로 generation(세대)과 threshold(임계값)로 가비지 컬렉션 주기와 객체를 관리한다. 세대는 0~2세대로 구분되고 최근 생성된 객체는 0세대(young)에 들어가고 오래된 객체일수록 2세대(old)로 이동한다. 당연히 한 객체는 단 하나의 세대에만 속한다. 가비지 컬렉터는 0세대일수록 더 자주 가비지 컬렉션을 하도록 설계되어있는데 generational hypothesis에 근거한다.

![https://plumbr.io/handbook/garbage-collection-in-java/generational-hypothesis](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2019/12/b.png?raw=true)

이 가설은 대부분은 어린 객체가 오래된 객체보다 해제될 가능성이 훨씬 높다는 가설이다.
reference counting mechanism과 달리 Python 프로그램에서 세대 가비지 컬렉터의 동작을 변경할 수 있다. 여기에는 코드에서 garbage collection process를 trigger 하기 위한 changing the thresholds(임계값 변경), garbage collection process(가비지 컬렉션 프로세스)를 수동으로 trigger 하거나 garbage collection process(가비지 컬렉션 프로세스)를 모두 비활성화하는 것이 포함된다.

### 왜 Garbage Collection은 성능에 영향을 주나
가비지 컬렉션이 어떤 역할을 하는지를 알게 되었다. 그러면 가비지 컬렉션이 성능에 어떤 영향을 주길래 중요한 것일까? 가비지 컬렉션을 수행하려면 응용 프로그램을 완전히 중지해야 한다. 그러므로 객체가 많을수록 모든 가비지를 수집하는 데 시간이 오래 걸린다는 것도 분명하다.
가비지 컬렉션 주기가 짧다면 응용 프로그램이 중지되는 상항이 증가하고 반대로 주기가 길어진다면 메모리 공간에 가비지가 많이 쌓일 것이다. 시행착오를 거치며 응용 프로그램의 성능을 끌어 올려야 한다.

### GC module 사용
gc 모듈을 사용하여 가비지 컬렉션 통계를 확인하거나 가비지 컬렉터의 동작을 변경하는 방법을 살펴본다.
gc 모듈을 사용하며 get_threshold() method를 사용하여 가비지 컬렉터의 구성된 임계값을 확인할 수 있다.

~~~
>>> import gc
>>> gc.get_threshold()
(700, 10, 10)
각각 threshold 0, threshold 1, threshold 2를 의미하는데 n세대에 객체를 할당한 횟수가 threshold n을 초과하면 가비지 컬렉션이 수행된다.
get_count() method를 사용하여 각 세대의 객체 수를 확인할 수 있다.
>>> import gc
>>> gc.get_count()
(121, 9, 2)
~~~

(위 값은 method를 호출할 때마다 변경된다) 위 예에서는 youngest generation(가장 어린 세대)에 121개의 객체, 다음 세대에 9개의 객체 oldest generation(가장 오래된 세대)에 2개의 객체가 있다.
Python은 프로그램을 시작하기 전에 기본적으로 많은 객체를 새성한다. gc.collect() 메소드를 사용하여 수동 가비지 콜렉션 프로세스를 추적할 수 있다.

~~~
>>> gc.get_count()
(121, 9, 2)
>>> gc.collect()
54
>>> gc.get_count()
(54, 0, 0)
~~~

가비지 컬렉션 프로세스를 실행하면 0세대에서 67개, 다음 세대에서는 11개 정도의 객체가 정리된다.
gc module에서 set_threshold() method를 사용하여 가비지 컬렉션 트리거 임계값을 변경할 수 있다.

~~~
>>> import gc
>>> gc.get_threshold()
(700, 10, 10)
>>> gc.set_threshold
(1000, 15, 15)
>>> gc.get_threshold()
(1000, 15, 15)
~~~

임계 값을 증가시키면 가비지 컬렉션이 실행되는 빈도가 줄어든다. 죽은 객체를 오래 유지하는 cost(비용)로 프로그램에서 계산 비용이 줄어든다.

### GC 예제
순환 참조 함수에서 발생할 수 있는 문제이다. 이 경우 reference counting으로는 메모리가 해제되지 않으며 garbage collector로만 가능하다.

~~~
import sys, gc, time

def create_cycle():
    a = [8, 9, 10]
    a.append(a)

def main():
    start = time.time()
    for i in range(70):
        create_cycle()

    return time.time() - start

if __name__ == "__main__":
    t = 0
    iter = 10
    gc.collect()
    for i in range(iter):
        t += main()
        print(gc.get_count())
    print(t % iter)
    sys.exit()
~~~

~~~
Output:
(74, 0, 0)
(146, 0, 0)
(216, 0, 0)
(286, 0, 0)
(356, 0, 0)
(426, 0, 0)
(496, 0, 0)
(566, 0, 0)
(636, 0, 0)
(0, 1, 0)
0.00018739700317382812
~~~

a 객체의 참조 카운트는 프로그램에서 삭제되거나 범위를 벗어난 경우에도 항상 0보다 크다. 따라서 순환 참조로 인해 a 객체가 가비지 수집되지 않는다.
threshold 0의 임계값인 700에 도달하기 전까지 가비지가 발생하는 모습을 볼 수 있다. 메모리가 낭비되는 것이다.
기본적으로 임계값이 (700, 10, 10)로 설정되어 있다. 객체 수가 threshold n을 초과하면 가비지 컬렉션이 실행된다. 다만 그 이후 세대부터는 조금 다른데 0세대 가비지 컬렉션이 일어난 후 0세대 객체를 1세대로 이동시킨 후 카운터를 1 증가시킨다. 1세대 또한 초과하면 2세대로 이동한다. 0세대 가비지 컬렉션이 객체 생성 700번 만에 일어난다면 1세대는 7000번 만에, 2세대는 70000번 만에 일어난다는 뜻이다.

##### gc.collect() 추가:
~~~
import sys, gc, time

def create_cycle():
    a = [8, 9, 10]
    a.append(a)

def main():
    start = time.time()
    for i in range(70):
        create_cycle()
    gc.collect()
    return time.time() - start

if __name__ == "__main__":
    t = 0
    iter = 10
    gc.collect()
    for i in range(iter):
        t += main()
        print(gc.get_count())
    print(t % iter)
    sys.exit()
~~~

~~~
Output:
(0, 0, 0)
(0, 0, 0)
(0, 0, 0)
(0, 0, 0)
(0, 0, 0)
(0, 0, 0)
(0, 0, 0)
(0, 0, 0)
(0, 0, 0)
(0, 0, 0)
0.042791128158569336
~~~

main() 함수가 끝날 때마다 GC를 실행시킨다. 그래서 함수 종료 시 임계값은 항상 0이다. 그러나 main() 함수마다 GC를 실행시키기 때문에 프로그램 속도가 느려진다. 위 프로그램에서는 약 200배 이상 느려진 것을 확인할 수 있다.

### Manual Garbage Collection
Manual Garbage Collection(수동 가비지 컬렉션)을 수행하는 방법은 두 가지가 있다.
1. Time-based(시간 기반): 가비지 컬렉터를 고정된 시간 간격마다 호출하는 것이다.
2. Event-based(이벤트 기반): 이벤트 발생 시 가비지 컬렉터를 호출한다. 예를 들어, 사용자가 응용 프로그램을 종료하거나 응용 프로그램이 중단 상태일 때 호출하는 것이다.

## 결론
reference counting과 garbage collector module이 어떻게 작동하는지 알았으니 이제 Python 응용 프로그램에서 어떻게 사용해야하는지 알아보자.

### Garbage Collector 동작을 수정하지 말자!
Python의 주요 장점 중 하나는 개발자의 생산성이 향상하는 것이다. 이유 중 하나는 개발자를 위한 여러 하위 수준의 세부 사항을 처리하는 고급 언어이기 때문이다.
수동 메모리 관리는 컴퓨터 자원이 제한된 환경에 더 적합하다. 그러나 garbage collector를 수정하는 것보다 컴퓨터 자원을 증가시키는 편이 훨씬 좋다.
Python이 일반적으로 운영 체제 메모리를 다시 release 하지 않는다는 사실을 고려하면 더욱더 그렇다. 메모리를 확보하기 위해 수행하는 수동 가비지 컬렉션 프로세스는 원하지 않는 결과가 나올 수 있다. (Memory management in Python)

### Garbage Collector 비활성화
위 경고를 무시하고 가비지 컬렉션 프로세스를 관리하려는 경우가 종종 있다. Python의 주요 garbage collection mechanism의 reference count는 비활성화할 수 없다. 변경할 수 있는 유일한 가비지 컬렉션 동작은 gc module의 generational garbage collector이다.
세대별 가비지 컬렉션을 변경하는 흥미로운 예 중 하나는 Instagram에서 가비지 컬렉션을 모두 비활성화 한 것이다. (Dismissing Python Garbage Collection at Instagram)
좀 더 파고들면 Instagram은 Python web framwork인 Django를 사용한다. single compute instance에서 여러 웹 어플리케이션 instance를 실행한다. 이러한 instance는 하위 프로세스가 마스터와 메모리를 공유하는 master-child mechanism을 사용하여 실행된다. Instagram 개발팀은 자식 프로세스가 생성된 직후 공유 메모리가 급격히 떨어지는 것을 발견했다. 더 분석해보자 가비지 컬렉터가 문제라는 것을 알게 됬다. Instagram 개발팀은 모든 세대의 임계값을 0으로 설정하여 가비지 컬렉터 모듈에서 비활성화했다. 이 변경으로 인해 웹 응용 프로그램이 10% 더 효율적으로 실행되었다.
이 예는 흥미롭지만, 이 방법을 따르기 전에 비슷한 상황에 처했는지를 확인해야 한다. Instagram의 웹 어플리케이션 규모는 수백만 명이 사용한다. 이들에게는 non-standard(비표준적)으로 행동하여 웹 응용 프로그램의 모든 성능을 한계치로 끌어 올리는 것이 좋다. 대부분의 개발자에게는 가비지 컬렉션과 관련된 Python의 표준 동작이 충분하다.
Python에서 가비지 컬렉션을 수동으로 관리하려는 경우 먼저 Stackify’s Retrace와 같은 툴로 응용 프로그램 성능 및 문제를 정확하게 파악하는 것이 중요하다. 문제를 파악했다면 다양한 튜닝을 통해 해결하면 된다.

### 마무리
이 글에서 Python의 메모리 관리 방법에 대해 배웠다. 가장 먼저 과거의 언어와 현대적인 언어의 메모리 관리 방법에 대해 알아보고 자동 메모리 관리의 장단점을 알게되었다. 그런 다음 automatic reference couting과 generational garbage collector를 통해 Python에서의 garbage collection의 구현에 대해 살펴보았다. 마지막으로 Python 개발자로서 이것이 얼마나 중요하고 어떻게 실무에 적용할 수 있는지를 알게 되었다.

## 참고 자료
[Python GC가 작동하는 원리](https://winterj.me/python-gc/)
[Python Garbage Collection: What It Is and How It Works](https://stackify.com/python-garbage-collection/)
[Garbage Collection in Python](https://www.geeksforgeeks.org/garbage-collection-python/)
[Python Garbage Collection(Python GC)](https://www.journaldev.com/17927/python-garbage-collection-gc)
[Python garbage collection](https://stackoverflow.com/questions/1035489/python-garbage-collection)
[Python Garbage Collection](https://weicomes.tistory.com/277)
[Basics of Memory Management in Python](https://stackabuse.com/basics-of-memory-management-in-python/)
