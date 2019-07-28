---
title: PySC2 Agent에 reward 사용하기
tags: 강화학습
key: page-Use-Reward-in-PySC2-Agent
---

#### PyTorch란?

사실 우리가 흔하게(?) 접하는 프로그래밍 언어는 Dynamic이다. 그냥 변수 선언 하고 출력하면 바로 값이 출력된다. 하지만 TensorFlow를 포함한 몇몇의 Deep Learning Library들은 Graph라는 것을 설계한 후에 Graph 내에서 작동한다. 구조를 바꾸려면 Graph부터 다시 설계해야 한다는 것이다. 그래서 TensorFlow를 사용해봤던 유저라면 tf.Session()을 알 것이다. 그러나 PyTorch는 다르다. PyTorch에게 가장 잘 어울리는 단어는 'Dynamic'이다. 일반 프로그래밍 처럼 동적이라는 것이다. Graph 구조를 자주 바꿔야 되는 연구나 실험을 할때 주로 PyTorch를 사용한다.

#### PyTorch와 TensorFlow 비교

| 구분 | TensorFlow | PyTorch |
| :-: | :-: | :-: |
| 패러다임 | Define and Run | Define by Run |
| 그래프 형태 | Static graph | Dynamic graph |

#### PyTorch 단점

1.  문서화가 잘 안되었다.
2.  협소한 사용자 커뮤니티.
3.  상용이 아니라 연구용으로 적합.

#### PyTorch 패키지

| 패키지 | 기술 |
| :-: | :-: |
| torch | 강력한 GPU 지원 기능을 갖춘 Numpy와 같은 Tensor 라이브러리 |
| torch.autograd | Torch에서 모든 차별화된 Tensor 작업을 지원하는 테이프 기반 자동 차별화 라이브러리 |
| torch.nn | 최고의 유연성을 위해 설계된 자동 그래프와 깊이 통합된 신경 네트워크 라이브러리 |
| torch.optim | SGD, RMSProp, LBFGS, Adam 등과 같은 표준 최적화 방법으로 torch.nn과 함께 사용되는 최적화 패키지 |
| torch.multiprocessing | 파이썬 멀티 프로세싱을 지원하지만, 프로세스 전반에 걸쳐 Torch Tensors의 마법같은 메모리 공유 기능을 제공. 데이터 로딩 및 호그 워트 훈련에 유용. |
| torch.utils | 편의를 위해 DataLoader, Trainer 및 기타 유틸리티 기능 |
| torch.legacy(.nn/optim) | 이전 버전과의 호환성을 위해 Torch에서 이식된 레거시 코드 |