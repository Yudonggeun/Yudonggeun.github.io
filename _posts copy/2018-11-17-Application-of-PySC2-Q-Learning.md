---
title: PySC2 Q-Learning 적용하기
tags: 강화학습
key: page-Application-of-PySC2-Q-Learning
---

## 강화학습의 시작
이 글에서는 기계학습 기술중 하나인 강화학습을 사용하여 Agent에게 유닛을 만들고 적을 자동으로 공격하는 방법을 알려줍니다.
Agent의 행동에 따라 보상이 다르게 주어집니다. 이를 통해 Agent는 어떤 행동을 해야 보상을 많이 받는지를 판단하고 보상을 많이 받을 수 있는 행동을 하게됩니다. 크게 Action, Reward, State그리고 과거를 통한 전체적인 보상이 조합되어 학습을 진행하게 됩니다.


## Agent 만들기
기본적인 상수를 만들고 class를 만드는 작업이 필요합니다:
~~~
import random
import math

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

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]

class SmartAgent(base_agent.BaseAgent):
    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]
    
    def step(self, obs):
        super(SmartAgent, self).step(obs)
        
        player_y, player_x = (obs.observation[‘feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
        return actions.FunctionCall(_NO_OP, [])
~~~

이제 Q-Learning table class를 추가해봅시다. 이것은 본질적으로 모든 State와 Action를 추적하는 “두뇌”이다:
~~~
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
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
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
~~~

그리고 Agent를 실행할 수 있습니다:
~~~
python -m pysc2.bin.agent --map Simple64 --agent q-learning_agent.SmartAgent --agent_race terran --max_agent_steps 0 --norender
~~~

여기까지의 소스코드는 [GitHub](https://github.com/Yudonggeun/PySC2-Tutorial/blob/master/5.%20Building%20a%20RL%20PySC2%20Agnet/q-learning_agent.py)에서 확인하세요.

Agent 단계 제한을 0으로 설정하여 실행을 중지 할 때 에이전트 기본값을 2500으로 설정하지 않는 것이 좋습니다.

## Action 정의
Agent가 action을 할 수 있는 선택지를 만들어 줘야합니다. 해병을 생산하고 공격하기를 원합니다. 상수를 사용하여 정의 하고 리스트를 만들어 선택할 수 있게 만들어 줍니다:
~~~
ACTION_DO_NOTHING = 'donothing'

ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
]
~~~

아무 것도하지 않는 행동을 정의하는 것이 이상해 보일 수도 있지만, 이것은 다른 행동들을 수행 할 수 있을 때 시스템이 잠기지 않게하기 위해 편리합니다. 또한 시스템 대기가 부정적인 보상을 초래하는 조치를 수행하는 것보다 낫다는 것을 알 수 있습니다.

Step 마다 어떤 Action을 선택을 하면 그 Action을 할 수 있도록 각각 메소드를 만들어 줍니다:
~~~
def step(self, obs):
    super(SmartAgent, self).step(obs)

    player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
    self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

    smart_action = smart_actions[random.randrange(0, len(smart_actions) - 1)]

    if smart_action == ACTION_DO_NOTHING:
        return actions.FunctionCall(_NO_OP, [])

    elif smart_action == ACTION_SELECT_SCV:
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

        if unit_y.any():
            i = random.randint(0, len(unit_y) - 1)
            target = [unit_x[i], unit_y[i]]

            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
        if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

                return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

    elif smart_action == ACTION_BUILD_BARRACKS:
        if _BUILD_BARRACKS in obs.observation['available_actions']:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

    elif smart_action == ACTION_SELECT_BARRACKS:
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

        if unit_y.any():
            target = [int(unit_x.mean()), int(unit_y.mean())]

            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    elif smart_action == ACTION_BUILD_MARINE:
        if _TRAIN_MARINE in obs.observation['available_actions']:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

    elif smart_action == ACTION_SELECT_ARMY:
        if _SELECT_ARMY in obs.observation['available_actions']:
            return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

    elif smart_action == ACTION_ATTACK:
        if _ATTACK_MINIMAP in obs.observation["available_actions"]:
            if self.base_top_left:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])

            return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])

    return actions.FunctionCall(_NO_OP, [])
~~~

마지막 튜토리얼 이후 수행한 최적화 중 하나를 확인할 수 있습니다. SCV를 선택할 때 x 및 y 좌표를 임의로 선택할 수 있습니다. 때로는 첫 번째 x 및 y 좌표를 선택하면 SCV 옆에 있는 것이 선택됩니다. 이 방법은 특히 반복되는 통화에서 이러한 일이 발생할 가능성을 줄여줍니다.

Q-Learning table 추가:
~~~
    def __init__(self):
        super(SmartAgent, self).__init__()
        
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
~~~

## State 정의
Q-Learning 시스템이 게임에서 어떤 일이 일어나는지 알아야 합니다:
~~~
def step(self, obs):
    super(SmartAgent, self).step(obs)

    player_y, player_x = (obs.observation['featre_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
    self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

    unit_type = obs.observation['featre_screen'][_UNIT_TYPE]

    depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
    supply_depot_count = 1 if depot_y.any() else 0

    barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
    barracks_count = 1 if barracks_y.any() else 0

    supply_limit = obs.observation['player'][4]
    army_supply = obs.observation['player'][5]

    current_state = [
        supply_depot_count,
        barracks_count,
        supply_limit,
        army_supply,
    ]
~~~

현재 상태를 알아야 하기 때문에 보급고와 병영이 건설 되었는지, 보급 한도 및 공격 유닛 여부를 측정할 것입니다.

무작위로  action을 선택하는 대신 Q-Learning 시스템이 현재 상태에 따라 action을 행동할 수 있습니다.
~~~
        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]
~~~

지금까지는 행동에 대한 보상이 없기 때문에 특정한 상태일 때 어떤 행동을 해야 좋은지를 모릅니다.


## Reward 정의
이제 보상을 도입하여 Q-Learning 시스템의 행동에 대하여 평가를 합니다. 예를 들어 상대 유닛이나 건물들을 파괴하였을 때 보상을 주는 것입니다:
~~~
KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5
~~~

우리가 언제 유닛이나 건물을 죽였는지를 알려주기 위해 누적 점수 시스템을 사용할 수 있습니다. 이 시스템은 점증적입니다. 따라서 현재 값과 비교할 마지막 단계의 값을 추적해야합니다. 값이 증가했는지 알 수 잇는 방식으로, 우리는 무언가를 파괴하였다는 것을 알 수 있을 것입니다. 몇 가지 속성을 추가하여 이전 값을 추적합니다:
~~~
    def __init__(self):
        super(SmartAgent, self).__init__()
        
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
~~~

다음으로 reward를 계산하고 값을 바꿔줍니다:
~~~
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        
        current_state = [
            supply_depot_count,
            barracks_count,
            supply_limit,
            army_supply,
        ]
        
        reward = 0
            
        if killed_unit_score > self.previous_killed_unit_score:
            reward += KILL_UNIT_REWARD
                
        if killed_building_score > self.previous_killed_building_score:
            reward += KILL_BUILDING_REWARD
                
        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]
~~~

다음으로 다음 새로운 값을 저장합니다:
~~~
        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]
        
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
~~~

## 마무리
시스템이 행동의 결과를 알기 위해서는 이전 상태와 행동을 추적해야합니다.
~~~
    def __init__(self):
        super(SmartAgent, self).__init__()
        
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        
        self.previous_action = None
        self.previous_state = None
~~~

상대의 유닛이나 건물을 파괴하였을 때 reward를 주고 reward와 파괴한 점수를 업데이트 해줍니다.
~~~
        current_state = [
            supply_depot_count,
            barracks_count,
            supply_limit,
            army_supply,
        ]
        
        if self.previous_action is not None:
            reward = 0
                
            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD
                    
            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD
                
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
        
        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]
~~~

마지막으로 다음 단계의 상태 값과 동작 값을 업데이트합니다:
~~~
        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]
        
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action
~~~

테스트를 하게되면 처음에는 랜덤한 행동을 하다가 점점 실력이 늘어나는 모습을 볼 수 있을 것이다.

여기까지의 소스코드는 [GitHub](https://github.com/Yudonggeun/PySC2-Tutorial/blob/master/4.%20Control%20Barrack%20and%20Army/build_barrack.py)에서 확인하세요.