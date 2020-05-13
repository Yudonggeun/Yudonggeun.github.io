---
title: Predict Stock Price Using Deep Learning
tags: Project, Deep Learning
key: predict-stock-price-using-deep-learning
---

## 서론
### 왜 주가 예측인가?
RNN, LSTM 등의 아키텍처는 시계열 데이터에 최적화된 모델이다. 과거의 패턴을 분석하여 미래를 예측할 수 있다는 것이다. 대표적인 시계열 데이터는 NLP, 진동, 주식 등이 있다. 이 중 주식에 대한 데이터는 오로지 숫자로 이루어져 있고 뭔가 예측이 될 것 같으면서도 이상하게 안 되기 때문에 오로지 과거 주가 데이터로만 내일의 주가를 예측 가능한지가 궁금해서 실험해 보기로 했다.

$$h_t = f(h_t, x_t)$$

## 본론
### 데이터 준비
학습은 Colab에서 진행하기 때문에 데이터를 파일 형태로 보관하고 있으면 여러 불편함이 있을 것 같았다. 그래서 어느 환경에서라도 데이터에 접근할 수 있도록 DataBase를 만들기로 했다. AWS를 사용하기로 했다. AWS는 RDS라는 제품을 지원하여 편하게 OS부분 부터가 아닌 Application부터 작업하면 된다. 그러나 편한 RDS(PaaS)보다 불편하더라도 운영 환경의 제약을 받지 않고, 원하는대로 커스터마이징이 쉽고 간편한 EC2(CaaS)를 사용하기로 했다.

EC2는 Ubuntu에 MySQL를 설치하였다.

