---
title: 스타크래프트 2 강화학습 개발환경 설정
tags: 강화학습
key: page-StarCraft2-Reinforcment-Learning-Development-Environment
---

## PySC2 설치
DeepMind에서 개발한 Python 라이브러리 PySC2를 사용합니다.

## 스타크래프트 II 설치
##### Linux
블리자드 [문서](https://github.com/Blizzard/s2client-proto#downloads)에서 Linux 버전의 스타크래프트 2를 설치합니다. PySC2에서는 '~/StarCraftII/'경로에 있기를 원합니다. 'SC2PATH' 환경 설정을 하거나 고유한 run_config을 작성하여 이를 대체할 수 있습니다.



##### Windows/MacOS
일반적인 [Battle.net](https://www.blizzard.com)에서 설치하면 됩니다. 만약 설치 경로를 변경하고 싶다면 SC2PATH 환경 설정이 필요할 수도 있습니다.


##맵 설정
PySC2에서는 학습을 위한 [mini game](https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip) 맵을 제공하고 있습니다. 또한 1대1 경기를 위한 [ladder maps](ladder maps)을 제공하고 있습니다. 맵을 다운 받은 후 'StarcraftII/Maps/'경로에 이동시키면 됩니다.

랜덤으로 기본 Agent를 실행해줍니다:
~~~
$ python -m pysc2.bin.agent --map Simple64
~~~

특정 Agent를 실행하고 싶다면 다음을 사용합니다:
~~~
$ python -m pysc2.bin.agent --map CollectMineralShards --agent pysc2.agents.scripted_agent.CollectMineralShards
~~~

다른 2개의 Agent를 한 맵에서 대결을 할 수도 있습니다:
~~~
$ python -m pysc2.bin.agent --map Simple64 --agent2 pysc2.agents.random_agent.RandomAgent
~~~

Agent의 race, 상대방의 난이도를 지정하려면 추가 flags를 전달하면 됩니다. 실행 할 때 --help를 사용하면 변경할 수 있는 내용을 확인 할 수 있습니다.

## 테스트 실행
다음을 사용하여 python 2또는 3에서 PySC2가 잘 작동하는지 확인할 수 있습니다:
~~~
$ python -m pysc2.bin.run_test
~~~

## 사람과 플레이
~~~
$ python -m pysc2.bin.play --map Simple64
~~~

UI에서 '?'를 입력하면 hotkeys 리스트를 확인 할 수 있습니다.

단축키
- F4: 종료
- F5: 재시작
- F9: 리플레이 저장
- Pgup/Pgdn: 게임 속도 조절

그렇지 않으면 마우스를 사용하여 왼쪽에 나열된 명령에 대한 선택 및 키보드를 사용하세요.

UI에서 왼쪽은 기본적인 rendering 화면을 보여줍니다. 오른쪽에서는 Agent가 받는 feature lyaers를 우리가 활용하기 쉽게 색으로 나타내 줍니다. Command-line flags를 이용하면 RGB 혹은 feature layer rendering을 활성화/비활성화 할 수 있습니다.

## 리플레이 재생
Agent와 사람과의 플레이를 리플레이로 저장할 수 있습니다. 다음을 사용하여 리플레이를 재생할 수 있습니다:
~~~
$ python -m pysc2.bin.play --replay <path-to-replay>
~~~
리플레이의 지도가 꼭 로컬에 존재해야합니다.

단축키
- F4: 종료
- pgup/pgdn: 속도 조절

'— video flag'를 사용하면 리플레이를 동영상으로 저장할 수 있습니다.


## 자세한 환경설정
전체적으로 환경을 설정하는 특별한 기술들은 [환경 문서](https://github.com/deepmind/pysc2/blob/master/docs/environment.md)에서 확인하세요.