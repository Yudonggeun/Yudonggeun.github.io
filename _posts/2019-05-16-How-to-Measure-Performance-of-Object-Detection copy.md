---
title: How to measure performance of object detection
tags: object-detection
key: how-to-measure-performance-of-object-detection
---

# 전문가를 위한 파이썬(내가 배운 것들)

## Part 1
### Collections.namedtuple()
튜플의 성질을 가지면서 이름으로 인덱스 접근 가능.

## Part 2
### __str__과 __repr__ 차이
__str__은 비공식적(informal)인 문자열을 출력하고 __repr__은 공식적(official)인 문자열을 출력한다. 공통점은 문자열을 출력하는데 사용된다는 것이고, 차이점은 사용처가 다른 것이다.

[Python __str__와 __repr__의 차이 살펴보기](https://shoark7.github.io/programming/python/difference-between-__repr__-vs-__str__)

### 내장 시퀀스 개요
#### 시퀀스형
컨테이너 시퀀스(container sequence)는 객체에 대한 참조를 담고 있으며 객체는 어떠한 자료형도 될 수 있지만, 균일 시퀀스(flat sequence)는 객체에 대한 참조 대신 자신의 메모리 공간에 각 항목의 값을 직접 담는다. 따라서 균일 시퀀스가 메모리를 더 적게 사용하지만, 문자, 바이트, 숫자 등 기본적인 자료형만 저장할 수 있다.
- 컨테이너 시퀀스: 서로 다른 자료형의 항목들을 담을 수 있는 list, tuple, collections,deque 등
- 균일 시퀀스: 단 하나의 자료형만 담을 수 있는 str, bytes, bytearray, memoryview, array, array 형

#### 시퀸스형을 가변성에 따라 분류
- 가변 시퀀스: list,bytearray, array.array, collections.deque, memoryview 형
- 불변 시퀀스: tuple,str, bytes 형

### 지능형 리스트와 제너레이터 표현식
지능형 리스트(리스트형의 경우)나 제너레이터 표현식(그 외 시퀀스의 경우)을 사용하면 시퀀스를 간단히 생성할 수 있다.

다음은 for문을 사용:
~~~
for symbol in symbols:
    codes.append(ord(symbol))
~~~

지능형 리스트:
~~~
codes = [ord(symbol) for symbol in symbols]
~~~

#### 지능형 리스트와 map()/filter() 비교
지능형 리스트는 map()/filter()를 조합한 방법보다 빠르다.
~~~
beyond_ascii = [ord(s) for s in symbols if ord(s) > 127]
beyond_ascii = list(filter(lambda c: c > 127, map(ord, symbols)))
~~~

### list.sort()와 sorted() 내장 함수
list.sort() 메소드는 사본을 만들지 않고 리스트 내부를 변경해서 정렬한다. sort() 메소드는 타깃 객체를 변경하고 새로운 리스트를 생성하지 않았음을 알려주기 위해 None을 반환한다.
이와 반대로 sorted() 내장 함수는 새로운 리스트를 생성해서 반환한다. sorted() 함수는 불변 시퀀스 및 재너레이터를 포함해서 반복 가능한 모든 객체를 인수르 받을 수 있다.

### 리스트와 배열
리스트형은 융통성이 있고 사용하기 편하지만, 세부 요구사항에 따라 더 나은 자료형도 있다. 예를 들어 실수를 천만 개 저장해야할 때는 배열이 훨씬 더 효율적이다. 배열은 모든 기능을 갖춘 float 객체 대신 C언어의 배열과 마찬가지로 기계가 사용하는 형태로 표현된 바이트 값만 저장하기 때문이다.

리스트 안에 숫자만 들어 있다면 배열이 리스트보다 훨씬 더 효율적이다. 배열은 pop(), insert(), extend() 등을 포함해서 가변 시퀀스가 제공하는 모든 연산을 지원하며, 빠르게 파일에 저장하고 읽어올 수 있는 frombytes()와 tofile() 메소드도 추가로 제공한다. 파이썬 배열은 C 배열만큼 가볍다. 또한 메모리가 절약된다.

array.fromfile() 메소드가 array.tofile() 메소드로 생성한 이진 파일에서 배밀도 실수 천만 개를 로드하는 데 0.1초 정도의 시간이 걸렸다. 이 속도는 float() 내장 함수를 이용해서 파싱하면서 텍스트 파일에서 숫자를 읽어오는 것보다 거의 60배 빠르다. array.tofile() 메소드로 저장하는 것은 각 행마다 실수 하나씩 텍스트 파일에 저장하는 것보다 약 7배 빠르다. 게다가 배밀도 실수 천만 개를 저장한 이진 파일의 크기는 80,000,000바이트(배밀도 실수 하나에 8바이트씩이며, 오버헤드가 전혀 없다)인 반면, 동일한 데이터를 저장한 텍스트 파일의 크기는 181,515,739바이트이다.

### 매핑
이 책에서는 평범한 dict 대신 collections의 dict를 주로 설명한다.

#### collections.defaultdict
defualtdict는 존재하지 않는 키로 검색할 때 요청에 따라 항목을 생성하도록 설정되어 있다. 작동하는 방식은, default 객체를 생성할 때 존재하지 않는 키 인수로 __getitem__() 메소드를 호출할 때마다 기본값을 생성하기 위해 사용되는 콜러블을 제공하는 것이다.

~~~
import collections
d = collections.defaultdict(list)
print(d['a'])
print(d)
~~~

Output:
~~~
defaultdict(<class 'list'>, {'a': []})
~~~

#### collections.OrderedDict
키를 삽입한 순서대로 유지함으로써 항목을 반복하는 순서를 예측할 수 있다. OrderedDict의 popitem() 메소드는 기본적으로 최근에 삽입한 항목을 꺼내지만, my_odict.popitem(last=True) 형태로 처음 삽입한 항목을 커낸다.

#### collections.ChainMap
매핑들의 목록을 담고 있으며 한꺼번에 모두 검색할 수 있다. 각 매핑을 차례대로 검색하고, 그중 하나에서라도 키가 검색되면 성공한다.

#### collections.Counter
모든 키에 정수형 카운터를 갖고 있는 매핑, 기존 키를 갱신하면 카운터가 늘어난다. 이 카운터는 해시 가능한 객체(키)나 한 항목이 여러 번 들어갈 수 있는 다중 집합에서 객체의 수를 세기 위해 사용할 수 있다.

~~~
import collections
ct = collections.Counter('asdfbadsf')
print(ct)

ct.update('aabbcc')
print(ct)
print(ct.most_common(3))
~~~

Output:
~~~
Counter({'a': 2, 's': 2, 'd': 2, 'f': 2, 'b': 1})
Counter({'a': 4, 'b': 3, 's': 2, 'd': 2, 'f': 2, 'c': 2})
[('a', 4), ('b', 3), ('s', 2)]
~~~

#### 불변 매핑
표준 라이브러리에서 제공하는 매핑형은 모두 가변형이지만, 사용자가 실수로 매핑을 변경하지 못하도록 보장하고 싶은 경우가 있을 것이다.
~~~
from types import MappingProxyType
d = {1: 'A'}
d_proxy = MappingProxyType(d)
print(d_proxy)
print(d_proxy[1])
d_proxy[2] = 'B'
~~~
Output:
~~~
{1: 'A'}
A
...
...
<module>
    d_proxy[2] = 'B'
TypeError: 'mappingproxy' object does not support item assignment
~~~

### dict와 set
dict와 set은 list보다 훨씬 빠른 속도를 가지고 있다. set은 데이터를 추가할 때 O(1)의 시간을 가진다.

#### 딕셔너리 안의 해시 테이블
해시 테이블은 희소 배열(중간에 빈 항목을 가진 배열)이다. 데이터 구조 교과서를 보면 해시 테이블 안에 있는 항목을 종종 버킷이라고 한다. dict 해시 테이블에는 각 항목 별로 버킷이 있고, 버킷에는 키에 대한 참조와 항목의 값에 대한 참조가 들어간다. 모든 버킷의 크기가 동일하므로 오프셋을 계산해서 각 버킷에 바로 접근할 수 있다.

파이썬은 버킷의 1/3 이상을 비워두려고 노력한다. 해시 테이블 항목이 많아지면 더 넓은 공간에 복사해서 버킷 공간을 확보한다.

##### 해시와 동치성
hash() 내장 함수는 내장 자료형은 직접 처리하고 사용자 정의 자료형의 경우 \_\_hash\_\_() 메서드를 호출한다. 두 객체가 동일하면 이 값들의 해시값도 동일해야 한다. 그렇지 않으면 해시 테이블 알고리즘이 제대로 작동하지 않는다. 예를 들어 정수 1과 실수 1의 내부 표현 형태는 다르지만, 1 == 1.0이 참이므로 hash(1) == hash(1.0)도 참이 되어야 한다.

#### 해시 테이블 알고리즘
my_dict[search_key]에서 값을 가져오기 위해 파이썬은 \_\_hash\_\_(search_key)를 호출해서 search_key의 해시값을 가져오고, 해시값의 최하위 비트를 해시 테이블 안의 버킷에 대한 오프셋으로 사용한다(사용하는 비트 수는 현재 테이블 크기에 따라 달라진다). 찾아낸 버킷이 비어있으면 KeyError를 발생시키고, 그렇지 않으면 버킷에 들어있는 항목인 (found_key : found_value) 쌍을 검사해서 search_key == found_key인지 검사한다. 이 값이 일치하면 항목을 찾은 것이므로 found_value를 반환한다.

#### 장단점
- dict 메모리 오버헤드가 크다: 해시가 제대로 작동하려면 빈 공간이 충분해야 하므로, dict의 메모리 공간 효율성은 높지 않다.
- 키 검색이 아주 빠르다: dict는 속도를 위해 공간을 포기하는 예다. 딕셔너리는 메모리 오버헤드가 상당히 크지만, 메모리에 로딩되는 한 딕셔너리 크기와 무관하게 빠른 접근 속도를 제공한다.
- 키 순서는 삽인 순서에 따라 달라진다: 해시 충돌이 발생하면 두 번째 키는 충돌이 발생하지 않았을 때의 정상적인 위치와 다른 곳에 놓이게 된다.

## Part 3
### 일급 함수
파이썬 함수는 일급 객체다. 다음과 같은 작업을 수행할 수 있는 프로그램 개체를 '일급 객체'로 정의한다.
- 런타임에 생성할 수 있다.
- 데이터 구조체의 변수나 요소에 할당할 수 있다.
- 함수 인수로 전달할 수 있다.
- 함수 결과로 반환할 수 있다.

#### 고위 함수
함수를 인수로 받거나. 함수 결과로 반환하는 함수를 고위 함수라고 한다. 대표적으로 map, sorted가 있다.

#### 익명 함수
lambda 키워드는 파이썬 표현식 내에 익명 함수를 생성한다.
~~~
sorted(fruits, key=reverse)

list(map(fact, range(6)))
~~~

그렇지만 파이썬의 단순한 구문이 람다 함수의 본체가 순수한 표현식으로만 구성되도록 제한한다. 즉 람다 본체에는 할당문이나 while, try 등의 파이썬 문장을 사용할 수 없다.

익명 함수는 인수 목록 안에서 아주 유용하게 사용된다.

~~~
sorted(fruits, key=lambda word:word[::-1])
~~~

#### 콜러블 객체
호출 연산자인 ()는 사용자 정의 함수 이외의 다른 객체에도 적용할 수 있는 객체인지 알아보려면 callable() 내장 함수를 사용한다. 파이썬 데이터 모델 문서는 다음 일곱 가지 콜러블을 나열하고 있다.

<br/>

- 사용자 정의 함수: def 문이나 람다 표현식으로 생성한다.
- 내장 함수: len()이나 time.strftime()처럼 C언어로 구현된 함수(CPython의 경우)
- 내장 메서드: dict.get()처럼 C언어로 구현된 메서드
- 메서드: 클래스 분체에 정의된 함수
- 클래스: 호출될 때 클래스는 자신의 \_\_new__() 메서드를 실행해서 객체를 생성하고, \_\_init__()으로 초기화한 후, 최종적으로 호출자에 객체를 반환한다. 파이썬에는 new 연산자가 없으므로 클래스를 호출하는 것은 함수를 호출하는 것과 동일하다.
- 클래스 객체: 클래스가 \_\_call__() 메서드를 구현하면 이 클래스의 객체는 함수로 호출될 수 있다.
- 제너레이터 함수: yield 키위드를 사용하는 함수나 메서드, 이 함수가 호출되면 제너레이터 객체를 반환한다.

파이썬에는 다양한 콜러블형이 존재하므로, callable() 내장 함수를 사용해서 호출할 수 있는 객체인지 판단하는 방법이 가장 안전하다.
~~~
[callable(obj) for obj in (abs, str, 13)]
~~~
Output:
~~~
[True, True, False]
~~~

#### 함수 애너테이션
파이썬3는 함수의 매개변수와 반환값에 메타데이터를 추가할 수 있는 구문을 제공한다.

~~~
def product(*numbers:'list') -> int:
...

print(product.__annotations__)
~~~

Output
~~~
{'numbers': 'list', 'return': <class 'int'>}
~~~

#### 오퍼레이터 모듈
함수형 프로그래밍을 할 때 산술 연산자를 함수로 사용하는 것이 편리할 때가 종종 있다. 예를 들어 팩리얼을 계산하기 위해 재귀적으로 함수를 호출하는 대신 숫자 시퀀스를 곱하는 경우를 생각해보자. 합계를 구할 때는 sum()이라는 함수가 있지만, 곱셈에 대해서는 이에 해당하는 함수가 없다. 5.2.1절 'map(), filter(), reduce()'의 대안에서 설명한 것처럼 reduce()함수를 사용할 수 있지만, reduce()는 시퀀스의 두 항목을 곱하는 함수를 필요로 한다. 람다를 이용해서 이 문제를 해결할 수 있다.

~~~
from functools import reduce

def fact(n):
    return reduce(lambda a, b: a*b, range(1, n + 1))
~~~

operator사용:
~~~
from functools import reduce
from operator import mul

def fact(n):
    return reduce(mul, range(1, n + 1))
~~~

### 함수 데커레이터와 클로저
함수 데커레이터는 소스 코드에 있는 함수를 '표시'해서 함수의 작동을 개선할 수 있게 해준다. 강력한 기능이지만, 데커레이터를 자유자재로 사용하려면 먼저 클로저를 알아야 한다.

#### 데커레이터 기본 지식
데커레이터는 다른 함수를 인수로 받는 콜러블(데커레이트된 함수)이다. 데커레이터는 데커레이트된 함수에 어떤 처리를 수행하고, 함수를 반환하거나 함수를 다른 함수나 콜러블 객체로 대체한다.

~~~
@decorate
def target():
    print('running target()')
~~~
위 코드는 다음 코드와 동일하게 작동한다.
~~~
def target():
    print('running target()')

target = decorate(target)
~~~

다음 처럼 실행 시간을 측정할 수 있다.
~~~
import time

def make_time_checker(func):
    def new_func(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print("실행 시간: ", end_time - start_time)
        return result
    return new_func

@make_time_checker
def big_number(n):
    return n ** n

a = big_number(4)
~~~

#### 표준 라이브러리에서 제공하는 데커레이터
##### functools.lru_cache()를 이용한 메모이제이션
functools.lru_cache()는 실제로 쓸모가 많은 데커레이터로서, 메모이제이션을 구현한다. 메모이제이션은 이전에 실행한 값비싼 함수의 결과를 저장함으로써 이전에 사용된 인수에 대해 다시 계산할 필요가 없게 해준다. 이름 앞에 붙은 LRU는 'Least Reccently Used'(사용한지 가장 오래된)의 약자로서, 오랫동안 사용하지 않은 항목을 버림으러써 캐시가 무한정 커지지 않음을 의미한다.

예시:

~~~
import functools
from clockdeco import clock

@clock
@functools.lru_cache()
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 2) + fibonacci(n - 1)

if __name__ == '__main__':
    print(fibonacci(6))
~~~

<br/>

lru_cach()는 두 개의 선택적 인수를 이용해서 설정할 수 있다.
~~~
functools.lru_cache(maxsize=128, typed=False)
~~~
maxsize 인수는 얼마나 많은 호출을 저장할지를 결정한다. 캐시가 가득차면 가장 오래된 결과를 버리고 공간을 확보한다. 최적의 성능을 내기 위해 maxsize는 2의 제곱이 되어야 한다. typed 인수는 True로 설정되는 경우 인수의 자료형이 다르면 결과를 따로 저장한다.

##### 매개변수화된 데커레이터
소스 코드에서 데커레이터를 파싱할 때 파이썬은 데커레이트된 함수를 가져와서 데커레이터 함수의 첫 번째 인수로 넘겨준다. 인수를 받아 데커레이터를 반환하는 데커레이터 팩토리를 만들고 나서, 데커레이트될 함수에 데커레이터 팩토리를 적용하면 된다.

~~~
registry = set()

def register(active=True):
    def decorate(func):
        print('running register(active=%s)->decorate(%s)' % (active, func))
        if active:
            registry.add(func)
        else:
            registry.discard(func)

        return func
    return decorate

@register(active=False)
def f1():
    print('running f1()')

@register()
def f2():
    print('running f2()')
~~~

f2() 함수만 registry에 남아 있다.

#### 클로저
클로저는 함수 본체에서 정의하지 않고 참조하는 비전역 변수를 포함한 확장 범위를 가진 함수다. 함수가 익명 함수인지 여부는 중요하지 않다. 함수 본체 외부에 정의된 비전역 변수에 접근할 수 있다는 것이 중요하다.


2번째~7번째 줄: 클로저
~~~
def make_averager():
    series = []

    def averager(new value):
        series.append(new_value)
        total = sum(series)
        return total/len(series)
    return averager
~~~

## Part 4
### 객체 참조, 기변성, 재활용
모든 객체는 정체성, 자료형, 값을 가지고 있다. 객체의 정체성은 일단 생성한 후에는 결코 변경되지 않는다. 정체성은 메모리 내의 객체 주소라고 생각할 수 있다. is 연산자는 두 객체의 정체성을 비교한다. id() 함수는 정체성을 나타내는 정수를 반환한다.

#### == 연산자와 is 연산자 간의 선택
== 연산자(동치 연산자)가 객체의 값을 비교하는 반면, is 연산자는 객체의 정체성을 비교한다.

#### 튜플의 상대적 불변성
