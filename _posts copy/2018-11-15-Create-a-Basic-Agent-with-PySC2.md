---
title: PySC2로 기본적인 Agent 만들기
tags: 강화학습
key: page-Create-a-Basic-Agent-with-PySC2
---

## Agent 만들기
make_general_agent.py를 만든다:
~~~
from pysc2.agents import base_agent
from pysc2.lib import actions
    class Agent(base_agent.BaseAgent):
        def step(self, obs):
            super(Agent, self).step(obs)
            
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
~~~

Agent의 기본적인 형태이다. 그리고 실행을 하기 위해 다음을 터미널에 입력하면 된다:
~~~
$ python -m pysc2.bin.agent --map Simple64 --agent make_general_agent.Agent --agent_race terran
~~~

게임을 천천히 진행하기 위해서 time 라이브러리를 추가한다 그리고 한 step당 0.5초를 멈추게 설정하였다:
~~~
from pysc2.agents import base_agent
from pysc2.lib import actions

import time

class Agent(base_agent.BaseAgent):
    def step(self, obs):
        super(Agent, self).step(obs)
        time.sleep(0.5)
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
~~~


## SCV 컨트롤
step 메소드에 대한 두 번째 매개 변수가 obs임을 알 수 있습니다. 이 변수에는 많은 "관찰"이 포함되어있습니다. obs는 많은 중첩 배열로 이루어저 있다.

다음과 같이 상수를 정의 해준다:
~~~
# Functions
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id

_NOOP = actions.FUNCTIONS.no_op.id

_SELECT_POINT = actions.FUNCTIONS.select_point.id



# Features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]
~~~

Simple64 맵을 사용할 것이다. 이 맵에는 2개의 시작위치가 있습니다. Agent는 인간과 마찬가지로 미니 맵을 보고 유닛이 위치한 곳을 찾습니다. 유닛을 구축하기 위해서 우리가 시작한 위치를 알아야 합니다. 다음으로 우리의 시작 위치를 저장할 변수를 추가합니다:
~~~
class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None
~~~

미니지도는 가로 세로 64 단위이고 높이는 64 단위이며, 왼쪽 상단 좌표는 (0, 0)입니다. 우리가받는 값은 0으로 인덱싱되어 있으므로 0 또는 31 사이의 x 또는 y 좌표를 가진 항목이 왼쪽 상단에 있다고 가정 할 수 있습니다. 32와 63 사이의 x 또는 y 좌표를 가진 것은 오른쪽 하단에 있습니다.

Observation은 우리에게 플레이어의 상대적인 정보를 줍니다. 배열 형태로 주어지고 유닛의 정보를 포함하고 있습니다. 이 배열에서 우리가 필요한 값을 가져올 것 입니다:
~~~
def step(self, obs):
    super(SimpleAgent, self).step(obs)

    time.sleep(0.5)

    if self.base_top_left is None:
        player_y, player_x = (obs.observation["feature_minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

        self.base_top_left = player_y.mean() <= 31
~~~

다음으로 우리의 위치를 알 수 있게 도와주는 함수를 만듭니다:
~~~
def transformLocation(self, x, x_distance, y, y_distance):
    if not self.base_top_left:
        return [x - x_distance, y - y_distance]

    return [x + x_distance, y + y_distance]
~~~

이 방법을 사용하기 위해 우리는 초기 x와 y 좌표를 보낸 다음 선택한 점이 되기를 원하는 이러한 좌표로부터의 거리를 보냅니다. 우리의 기준점이 왼쪽 상단에 있다면, 거리가 추가됩니다. 즉, 양의 거리가 있으면 선택한 점이 오른쪽 하단에 더 가깝게 이동합니다. 우리의 밑면이 오른쪽 하단에 있다면, 거리가 뺄 것입니다. 이는 양의 거리가 선택된 점을 왼쪽 상단에 더 가깝게 옮길 것임을 의미합니다.

## 보급고 짓기
보급고를 짓기 위해서는 2가지의 과정이 필요합니다. 먼저 일꾼을 선택해야하고 그 다음에 보급고를 지어야 합니다:
~~~
    supply_depot_built = False
    scv_selected = False

    if not self.supply_depot_built:
        if not self.scv_selected:
            unit_type = obs.observation["feature_screen"][_UNIT_TYPE]

            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

            target = [unit_x[0], unit_y[0]]

            self.scv_selected = True

            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
~~~

여기서 “화면”의 좌표는 맵 전체의 좌표가 아닌 화면 자체의 좌표를 나타냅니다. Observation은 (y, x)형태로 넘겨줍니다. 이를 우리는 (x, y)형태로 바꿔 처리합니다. 

위의 코드에서 화면의 모든 SCV 단위에 대한 x 및 y 좌표를 얻은 다음 unite_x [0] 및 unit_y [0]으로 첫 번째 x 및 y 좌표를 선택합니다.

마지막으로 우리는 이전처럼 FunctionCall을 반환하지만, 이번에는 인간이 SCV를 선택하기 위해하는 것처럼 SCV의 위치에서 마우스를 “클릭”하도록 게임을 지시하는 다른 매개 변수를 전달합니다:
~~~
    elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
        unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20

        self.supply_depot_built = True

        return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])
~~~

보급고 짓기를 할 때  사령부와 가까운 위치에 짓기 위해서 사령부의 좌표를 구합니다. 사령부의 좌표를 얻어 여러개의 값을 구할 수 있고 이를 평균 내어 중앙 값을 구합니다.

screen은 84x84 크기의 “units”이며 좌표 (0, 0)은 왼쪽 위 코너입니다. 위 코드에서 보급고를 건설할 위치를 사령부의 아래로 설정하였습니다.

마지막으로 step메소드는 실행할 action이 없어도 빈 값으로 넘겨주어야 합니다:
~~~
    return actions.FunctionCall(_NOOP, [])
~~~

실행 해보면 일꾼이 보급고를 건설하는 모습을 확인할 수 있을 것입니다:
~~~
python -m pysc2.bin.agent --map Simple64 --agent make_general_agent.SimpleAgent --agent_race terran
~~~

최종 소스 코드는 다음 [GitHub](https://github.com/Yudonggeun/PySC2-Tutorial/blob/master/3.%20Make%20general%20Agent/make_general_agent.py)에서 확인하세요.