---
title: Predict Stock Price Using Deep Learning
tags: Project, Deep Learning
key: predict-stock-price-using-deep-learning
---

## 서론
### 왜 주가 예측인가?
RNN, LSTM 등의 아키텍처는 시계열 데이터에 최적화된 모델이다. 과거의 패턴을 분석하여 미래를 예측할 수 있다는 것이다. 대표적인 시계열 데이터는 NLP, 진동, 주식 등이 있다. 이 중 주식에 대한 데이터는 오로지 숫자로 이루어져 있고 뭔가 예측이 될 것 같으면서도 이상하게 안 되기 때문에 오로지 과거 주가 데이터로만 내일의 주가를 예측 가능한지가 궁금해서 실험해 보기로 했다.

$$h_t = f(h_t, x_t)$$

### 자료가 별로 없어요ㅠ
모델의 성능이 바로 돈으로 이어지다 보니 인터넷 자료가 거의 없고 비슷한 내용들 뿐이라 실험하며 개발하는데 오랜 시간이 걸렸다. 또한 모델을 수십번 테스트 해보며 기존 모델들을 제대로 보관하지 않았고 성능 테스트 자료도 남은게 별로 없어 성능에 대한 증거가 불충분하다... 그러나 전부 직접 테스트 해본 것이다.

## 본론
### 데이터 준비
학습은 Colab에서 진행하기 때문에 데이터를 파일 형태로 보관하고 있으면 여러 불편함이 있을 것 같았다. 그래서 어느 환경에서라도 데이터에 접근할 수 있도록 DataBase를 만들기로 했다. AWS를 사용하기로 했다. AWS는 RDS라는 제품을 지원하여 편하게 OS부분 부터가 아닌 Application부터 작업하면 된다. 그러나 편한 RDS(PaaS)보다 불편하더라도 운영 환경의 제약을 받지 않고, 원하는대로 커스터마이징이 쉽고 간편한 EC2(CaaS)를 사용하기로 했다.

EC2는 Ubuntu에 MySQL를 설치하였다.

Data는 [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader) Library를 사용하여 MySQL에 저장하였다. 데이터는 Date, Open(시가), High(고가), Low(저가), Close(종가), Volume(거래량)을 사용하기로 했다.

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2020/05/a.PNG?raw=true)

### 데이터 전처리
주가는 종목별로 다르기 때문에 모델이 학습하는데 어려움이 있다. 그래서 대부분 minmaxscaler를 사용하는데 사실 주가의 범위가 정해지지 않는 주식 데이터에서는 적합한 방법이 아니다. 이유는 간단하다. A종목의 과거 train data에서는 2000원~6000원 사이에 분포해 있었다. 그러나 test데이터에서 6000원의 범위를 넘어서게 되면 과거 train data에서 보기 힘든 상황이기 대문에 모델이 힘들어할 수 있다. (Outlier라고 표현한다.)

그러면 다른 데이터 전처리 방법은 없는가? 주식의 특징이라면 수익률을 매우 중요시하기 때문에 대부분의 사람들 "오늘은 A주식이 2%올랐네", "코스피가 0.4% 하락했어" 등 %(percent)로 표현한다. 그래서 percent로 전처리하기로 했다. 한국 거래소는 가격제한폭(주식시장 및 파생상품 시장에서 개별 주식이나 종목이 일정범위 이상 거래될 수 없도록 한 제한폭이다.)이 있는데 코스피와 코스닥의 상한선은 ±30%이다. 하루 주가가 30%이상 변동할 수 없다는 것이다. percent로 주가를 변환했을 때 데이터는 Outlier가 없다.

전일 대비 percent로 변환 후 MinMax를 취해 최종적으로 0~1 사이값이 되도록 했다.

$$x_t = \In {(x_{t} - x_{t-1}) * 100}$$

### 간단하게 LSTM을 적용해보자
LSTM이 시계열에 최적화된 아키텍처니까 최소 정확도가 70%는 나오지 않을까 생각했다. 두 번째 LSTM은 Many-to-One이었고 마지막 Cell Output 값만 Linear에 들어갔다. 모델의 최종 Output은 t의 시가, 고가, 저가, 종가 대비 t+1의 시가, 고가, 저가, 종가 상승률을 예측한 값이다. 학습 데이터는 3종목의 2000-01-01 ~ 2020-04-01 데이터로 학습시켰으며 성능 테스트에서는 이 종목이 포함되지 않았다.

LSTM(hidden=512) -> LSTM(hidden=256) -> Linear(64) -> Linear(32) -> Lienar(4)

#### 성능
인터넷에 있는 주가 예측 결과를 어떻게 보면 잘 예측하는 것처럼 보일 수 있으나 직접 확대해서 보면 내일 값을 잘 추정하는게 아니라 오늘 값을 추정하는 듯 보였으며 선형적으로 optimize되기 때문에 loss를 낮추기 위해 단순히 전체적으로 추정하는 듯 보였다.

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2020/05/b.PNG?raw=true)
출처: http://investingdeeply.com/blog/predicting-stock-market-keras-tensorflow/

남은게 없어 인터넷에서 찾아온 자료지만 이처럼 결과가 나온다. 마치 내일 주가를 예측하는 것처럼 보이지만 수익률 테스트를 해보면 처참하다.

#### 수익률 테스트
수익률 테스트는 50종목의 내일 주가를 예측하여 가장 높은 상승률로 예측되는 주식의 실제 내일 주가를 곱한다. 아래 그래프는 직접 테스트하고 저장한 자료이며 파란색 선은 누적 수익률, 초록색 선은 50종목의 일별 평균 변화율을 나타낸다. 수익의 변화율이 매우 크며 30%까지 수익을 보고 바로 30%가 사라지는 성능을 보였다. 실제로 사용하기에는 무리가 있어보인다.

![](https://github.com/Yudonggeun/yudonggeun.github.io/blob/master/images/2020/05/b.png?raw=true)
