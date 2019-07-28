---
title: PySC2 Agent에 reward 사용하기
tags: 강화학습
key: page-Use-Reward-in-PySC2-Agent
---

## 시작
전 튜토리얼 까지는 Agent가 흥미로운 결과를 가졌음을 알았을 것입니다. 해병이 밖으로 나오기를 기다리며, 유닛이 등장 할 때까지 기다려 Agent는 보상을 얻으려고 할 것입니다.
이번 튜토리얼에서는 reward를 사용할 것입니다. 쉽게 말해 Agent는 게임에서 승리하였을 때 reward를 1를 주고 지게되면 -1를 줍니다. 더 많은 학습을 필요로 하지만 최종 결과는 승리를 많이 하게 될 것입니다.

Library와 변수를 사용한다:
~~~
import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
~~~

위에 마지막 부분은 새로운 것입니다. 일꾼이 광물을 채취하도록 명령을 내릴 것입니다.

유닛의 타입과 ID를 쉽게 코드를 사용할 수 있게 만들어줍니다:
~~~
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341
~~~

새로운 ID가 있는데, 이 ID는 광물의 위치를 나타낸다.
~~~
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]
~~~

화면에서 유닛을 선택할 때 한 종류의 유닛만들을 선택할 수 있습니다.

Q Learning table를 저장 할 것입니다. 우리는 학습이 끝날 때 적용할 것이며, 수백개의 에피소드를 실행할 필요가 있을 때 편리하게 사용할 수 있습니다.
~~~
ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'
~~~

여기서 누락된 것을 확인 할 수 있을 것이다. 이 튜토리얼에서는 일련의 작업(병영 선택, 해병 훈련)을 단일 작업(마린 생산)으로 압축 할 예정입니다. 앞으로 계속 설명할 것입니다.
~~~
smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE]


for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + ‘_’ + str(mm_x - 16) + ‘_’ + str(mm_y - 16))
~~~

더 쉽게 학습할 수 있도록 액션을 할 수 있는 공간을 줄였습니다. 그래서 4분면으로 나누었습니다. 즉, smart_actions에 공격하는 경우를 지역으로 나누어 4가지로 만들었습니다.


## Q Learning Table 추가
이전의 튜토리얼에서도 Q Learning Table을 사용했었다. 이번에는 pandas 업데이트를 지원할 수 있도록 업데이트 하였습니다.
~~~
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    
    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        
        return action
    
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r # next state is terminal
~~~

마지막 4개의 줄은 이전의 튜토리얼과 다릅니다, 각 지점에서 전체 보상을 적용하는 대신, 상태가 터미널이 될 경우(승리 또는 패배) 완전한 보상을 주는 것이 아닙니다. 다른 Learning Step은 보상을 줄일것입니다.
~~~
# update
    self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

def check_state_exist(self, state):
    if state not in self.q_table.index:
        # append new state to q table
        self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
~~~


## Agent 만들기
Agent의 시작은 전 튜토리얼과 같은  QLearningTable 설정을 기본으로 사용합니다.
~~~
class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()
        
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        
        self.previous_action = None
        self.previous_state = None
        
        self.cc_y = None
        self.cc_x = None
        
        self.move_number = 0
~~~

사령부의 위치를 나타내는 두가지의 cc_x와 cc_y를 추가하였다. 또 다른 속성의 move_number는 다중 단계 액션 내에서 시퀀스 위치를 추적합니다.
~~~
if os.path.isfile(DATA_FILE + '.gz'):
    self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
~~~

이 코드는 심플하다, `sparse_agent_data.gz` 파일이 있다면, 파일에서 Q Learning Table 데이터를 불러온다. 이는 이전 학습에서 학습을 재개할 수 있습니다. 만약 학습을 멈추거나 Agent가 충돌하는 버그가 있는 경우, 간단하게 학습을 재시작할 수 있고 과거의 학습 내역을 잃어 버리지 않습니다.

~~~
def transformDistance(self, x, x_distance, y, y_distance):
    if not self.base_top_left:
        return [x - x_distance, y - y_distance]

def transformLocation(self, x, y):
    if not self.base_top_left:
        return [64 - x, 64 - y]

    return [x, y]

def splitAction(self, action_id):
    smart_action = smart_actions[action_id]
    
    x = 0
    y = 0
    
    if '_' in smart_action:
        smart_action, x, y = smart_action.split('_')
    
    return (smart_action, x, y)
~~~

처음 두 메소드는 이전 튜토리얼과 같고, 본질적으로 바닥이 오른쪽 하단에 위치하면 스크린과 미니 맵 위치를 뒤 바꿀 수 있습니다. 즉, 모든 작업을 왼쪽 상단부터 수행하는 것으로 처리 할 수 있으므로 에이전트가 더 빨리 학습 할 수 있습니다.
마지막 메소드는 선택한 작업에서 필요한 정보를 추출하는 유틸리티입니다.
~~~
def step(self, obs):
    super(SparseAgent, self).step(obs)

    unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

    if obs.first():
        player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
~~~

아마도 `obs.first()`를 처음 볼 것입니다. 그리고 이 메소드는 본질적으로 게임의 첫 번째 step때만 활성화 됩니다. 그리고 여기에서 게임의 나머지 부분에 필요한 것을 설정할 수 있습니다.
~~~
cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
cc_count = 1 if cc_y.any() else 0

depot_y, depot_x = (unit_type) == _TERRAN_SUPPLY_DEPOT.nonzero()
supply_depot_count = int(round(len(depot_y) / 137))

barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
barracks_count = int(round(len(barracks_y) / 137))

return actions.FunctionCall(_NO_OP, [])
~~~

여기에 우리가 가지고있는 것을 알 필요가있는 상태 및 다른 코드에 사용할 수있는 몇 가지 카운트를 설정하는 중입니다. 여기에서 자세한 설명을 읽을 수 있습니다.

이제 기본 Agent를 실행해 볼 수 있습니다. 그러나 얻는 것은 없을 겁니다…


## Multi-Step Agent의 첫 단계 추가
이전에 언급 하였듯이, 여러 액션을 하나의 액션으로 합칠 것이다. 이로 인해 우리의 액션이 더욱 단순화해 지고 우리의 Agent가 빠르게 학습하는 것을 도와줄 것이다. 모든 multi-step actions은 모두 3 단계를 소비합니다(적은 양이라 할지라도). 그래서 3 단계마다 학습 호출이 항상 일관되게 유지됩니다.

우리가 취할 주요 행동은 다음과 같습니다.
- 아무 것도 하지 않기 - 3 단계를 수행하지 않는다.
- 보급고 건설 - 일꾼 선택, 보급고 건설, SCV를 광물로 보낸다.
- 병영 건설 - 일꾼 선택, 보급고 건설, SCV를 광물로 보낸다.
- 해병 생산 - 모든 병영 선택, 해병 생산, 아무것도 안함.
- Attack(x, y) - 군대 선택, 좌표 선택, 아무것도 안함.

불행하게도, 일꾼을 미네랄로 다시 돌려보내는 것은 완벽하지 않습니다. 그러나 합리적으로 잘 작동하는 것 같습니다. 목적은 일꾼이 막사 옆에 위치 할 때 병영 대신에 일꾼을 선택하는 것입니다.

이전의 단계의 `barracks_count` 라인 이후에 코드를 삽입하세요:
~~~
if self.move_number == 0:
    self.move_number += 1
~~~

시작하려면 이 단계가 여러 단계 작업의 첫 번째 단계인지 확인해야합니다. 값 0.을 포함하는 self.move_number로 표시됩니다. 숫자를 증가시켜 다음 게임 단계에서 multi-step action의 두 번째 단계를 진행하도록합니다.
~~~
current_state = np.zeros(8)
current_state[0] = cc_count
current_state[1] = supply_depot_count
current_state[2] = barracks_count
current_state[3] = obs.observation['plyaer'][_ARMY_SUPPLY]
~~~

다음으로 각 건물 유형과 해병 수를 포함하도록 상태를 설정하였습니다.
~~~
hot_squares = np.zeros(4)
enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()

for i in range(0, len(enemy_y)):
    y = int(math.ceil((enemy_y[i] + 1) / 32))
    x = int(math.ceil((enemy_x[i] + 1) / 32))

    hot_squares[((y - 1) * 2) + (x - 1)] = 1

if not self.base_top_left:
    hot_squares = hot_squares[::-1]

if i in range(0, 4):
    current_state[i + 4] = hot_squares[i]
~~~

이제 미니맵을 사분면으로 나눌 것입니다. 만약 적 유닛이 포함되어있다면 “hot”으로 표시 할 것입니다. 베이스가 오른쪽 하단에 있다면 모든 게임이 실제로 어느 위치에 있던지 관계 없이 맨 위 왼쪽베이스의 관점에서 보일 수 있도록 사분면을 뒤집습니다. 이는 우리의 Agent가 더 빨리 배우는 데 도움이됩니다.
~~~
if self.previous_action is not None:
    self.qlearn.learn(str(self.previous_state), self.previous_state, 0, str(current_state))
~~~

우리가 첫 번째 game step에 있지 않다면, 우리는 Q Learning Table에서 learn() 메소드를 호출합니다. 이 작업은 각 multi-step action의 첫 번째 단계에서만 수행되기 때문에 상태가 주어지고 매 3 번째 게임 단계마다 학습이 수행됩니다.
~~~
rl_action = self.qlearn.choose_action(str(current_state))

self.previous_state = current_state
self.previous_action = rl_action

smart_actions, x, y = self.splitAction(self.previous_action)
~~~

다음으로 우리는 하나의 액션을 선택하고 x와 y좌표가 존재한다면 이를 깨뜨립니다
~~~
if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
    unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

    if unit_y.any():
        i = random.randint(0, len(unit_y) - 1)
        target = [unit_x[i], unit_y[i]]

        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
~~~

첫 번째 step은 보급고나 병영을 건설하기 위해 일꾼을 선택하는 것입니다. 우리는 스크린상의 모든 SCV 좌표를 확인하고 무작위로 클릭함으로써 이것을 수행합니다.
~~~
elif smart_action == ACTION_BUILD_MARINE:
    if barracks_y.any():
        i = random.randint(0, len(barracks_y) - 1)
        target = [barracks_x[i], barracks_y[i]]

        return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
~~~

해병을 생산하는 첫 번째 단계는 병영을 선택하는 것입니다. 사실 `_SELECT_ALL` 값을 전송함으로써 우리는 모든 병영을 동시에 선택할 수 있습니다. 중요한 이점은 게임에서 바쁜 병영에서 다음 해병을 자동으로 대기열에 넣고, 옳은 병영을 선택하는 것을 처리한다는 것입니다.
~~~
elif smart_action == ACTION_ATTACK:
    if _SELECT_ARMY in obs.observation['available_actions']:
        return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
~~~

이제 군대를 선택하고 특정한 좌표로 공격하는 단계를 배울 것입니다.

## Multi-Step Actions에 두 번째 단계 추가
전 단계에 따라서 코드를 추가한다.
~~~
elif self.move_number == 1:
    self.move_number += 1
    
    smart_action, x, y = self.splitAction(self.previous_action)
~~~

이동 수를 증가시키고 액션 세부 정보를 추출하는 것으로 시작합니다. 
~~~
if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
    if supply_depot_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
        if self.cc_y.any():
            if supply_depot_count == 0:
                target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
            elif supply_depot_count == 1:
                target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)

            return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
~~~

보급고를 짓는 두 번째 단계는 보급고를 지정된 위치에 건설하기 위해서 일꾼에 명령을 내리를 것입니다. 저장된 사령부 위치를 사용함으로써 사령부가 파괴 되더라도 보급고를 건설 할 수 있습니다. Agent가 쉽게 작업을 수행 할 수 있도록 각 공급 저장소 위치는 하드 코딩되어있습니다. 세부 사항을 계산할 필요가 없어서 단순화 시켰습니다.

여기서의 문제는 두 번째 보급고가 건설된 후 첫 번째 보급고가 파괴 될 수 있다는 것입니다.  이로 인해 두 번째 보급고를 첫 번째 보급고로 생각하여 이미 건설 되어있는 두 번째 보급고의 위치에 보급고 건설을 시도할 수 있다는 것입니다. 이 문제를 해결할 수도 있지만, 그것을 필요로하지 않았습니다. 보통 적군이 병영 중 하나를 파괴하면 거의 패배한 것 입니다.
~~~
elif smart_action == ACTION_BUILD_BARRACKS:
    if barracks_count < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:
        if self.cc_y.any():
            if barracks_count == 0:
                target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
            elif barracks_count == 1:
                target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)

            return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
~~~

 병영을 건설하는 두 번째 단계는 보급고와 동일하고 오직 좌표만 다릅니다.
~~~
elif smart_action == ACTION_BUILD_MARINE:
    if _TRAIN_MARINE in obs.observation['available_actions']:
        return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
~~~

해병을 생산하는 두 번째 단계는 상당히 간단합니다. 병영에게 해병을 훈련시키게 하면됩니다. 병영이 여러 해병을 연속으로 대기열에 올릴 수 있도록 명령을 대기 행렬에 넣습니다. 이것은 군대가 공격을 당하고 증원을 필요로 할 때 편리합니다.
~~~
elif smart_action == ACTION_ATTACK:
    do_it = True

    if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
        do_it = False

    if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
        do_it = False

    if do_it and _ATTACK_MINIMAP in obs.observation['available_actions']:
        x_offset = random.randint(-1, 1)
        y_offset = random.randint(-1, 1)

        return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])
~~~

공격을 위한 두 번째 단계는 단순히 군대가 미니맵의 위치로 공격하는 명령을 내리는 것입니다.

실수로 일꾼을 선택하고 함께 공격을 하려고 할 때  `single_select`와 `multi_select`에서 우리가 일꾼을 선택하지 않았는지 확인해야합니다.

우리가 군대를 선택했다면, 무작위로 사분면의 중심 주위의 위치를 선택합니다. 이렇게 하면 우리의 행동 공간을 단지 4개의 좌표로 설정할 수 있습니다. 사분면 주위를 공격하고 적 유닛을 손대지 않게합니다.


## Multi-Step Actions의 마지막 단계 추가
현재 step에 코드를 따라 입력합니다:
~~~
elif self.move_number == 2:
    self.move_number = 0

    smart_action, x, y = self.splitAction(self.previous_action)

    if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
        if _HARVEST_GATHER in obs.observation['available_actions']:
            unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)

                m_x = unit_x[i]
                m_y = unit_y[i]

                target = [int(m_x), int(m_y)]

                return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
~~~

이제 남은 단계는 일꾼을 광물로 돌려보내는 것입니다. 이 작업은 일꾼이 보급고 또는 병영을 완성한 후에 수행되도록 예약된 상태입니다.

모든 무효한 행동은 `_NO_OP`를 호출하게 됩니다.

## 게임 종료 탐지
마지막 단계는 게임이 끝난 후에 보상을 적용시키고 다음 episode의 속성들을 초기화 시키는 것이다.

`super(SparseAgent, self).step(obs)` 호출 후에, Super Agent의 step의 시작부분에 다음 메소드를 추가합니다.
~~~
if obs.last():
    reward = obs.reward
~~~

단순히 `obs.first()`를 호출은 이전에 만든 `obs.first()` 호출과 마찬가지로 episode에서 마지막 게임 step을 감지할 수 있습니다.

다행이도, DeepMind는 우리가 필요한 observation의 일부로 필요한 보상을 obs.reward 형식으로 제공해 줍니다. 이 값은 승리를 하면 1, 패배하면 -1, 교착 상태가 발생하거나 episode가 28,800 step에 도달하면 0을 줍니다.

Game_steps_per_episode 명령 줄 매개 변수를 사용하여 episode step 제한을 늘릴 수 있습니다.(필수는 아닙니다.)

~~~
self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
~~~

다음으로 reward를 Q Learning Table에 적용할 것입니다. 대신에 현재 상태 대신 전체 보상을 적용하는 특수 상태를 나타내는 터미널 문자열을 전달합니다.
~~~
self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
~~~

우리는 Q Learning Table 데이터를 gzipped pickle 형식으로 출력하므로 어떤 이유에서든 Agent가 중지된 경우 다시 불러올 수 있습니다.
~~~
self.previous_action = None
self.previous_state = None

self.move_number = 0

return actions.FunctionCall(_NO_OP, [])
~~~

그런 다음 Agent를 재설정 해서 신선한 상태로 시작할 수 있습니다. 이것들은 첫 번째 단계에서 재설정 될 수 있지만 이렇게 하는 것이 더 깔끔해 보입니다.

코드에서 더 이상 진행할 가치가 없으므로 즉시 _NO_OP 호출을 반환합니다.


## Agent 실행
Agnet를 실행시키기 위해 다음과 같은 명령어를 사용합니다:
~~~
python -m pysc2.bin.agent —-map Simple64 --agent reward.SparseAgent —-agent_race Terran —-max_agent_steps 0 -—norender
~~~

1035 게임을 한 후의 결과이다:
![](https://raw.githubusercontent.com/Yudonggeun/yudonggeun.github.io/master/images/2018/19/a.png)

기본 설정을 사용하는 Agent가 50% 미만으로 패배하는 것에 감명을 받았습니다. 다음 튜토리얼에서는 승률 70%로 증가시킬 것입니다.

reward history와 최종 Q Learning Table의 결과를 [이곳](https://docs.google.com/spreadsheets/d/10I4F4ONFo23DLp7kX5gHk62cOJ7JEzeFNj6d_GxDsqM/edit#gid=376804423)에서 확인 하실 수 있습니다.

여기까지의 소스코드는 [GitHub](https://github.com/Yudonggeun/PySC2-Tutorial/blob/master/7.%20Using%20Reward%20for%20Agent/reward_agent.py)에서 확인하세요.