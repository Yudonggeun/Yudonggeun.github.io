---
title: 병영 건설과 해병 생산하고 공격하기
tags: 강화학습
key: page-Controlling-the-Barracks-Controlling-your-Army
---

## 병영 건설
병영을 건설하기 위해서 상수를 사용하여 병영을 정의합니다:
~~~
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
~~~

또한 병영이 지어져 있는지를 확인합니다:
~~~
Barracks_built = False
~~~

병영을 건설하는 과정은 보급고와 매우 비슷합니다:
~~~
        elif not self.barracks_built:
            if _BUILD_BARRACKS in obs.observation["available_actions"]:
                unit_type = obs.observation[“feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                
                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                

                self.barracks_built = True
                
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
~~~

병영은 사령부의 오른쪽이나 사령부가 아래 오른쪽에 있을 경우에는 왼쪽에 건설하기를 원합니다. 보급고를 건설한 일꾼을 선택하여 병영을 건설하는 것으로 가정합니다.

여기까지의 소스코드는 [GitHub](https://github.com/Yudonggeun/PySC2-Tutorial/blob/master/4.%20Control%20Barrack%20and%20Army/build_barrack.py)에서 확인하세요.

## 병영 컨트롤
병영을 건설할 수 있으니 이제는 병영에서 유닛을 생산 할 것이다:
~~~
# Functions
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id

# Unit IDs
_TERRAN_BARRACKS = 21

# Parameters
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
~~~

몇 가지의 상수를 추가해줍니다:
~~~
barracks_selected = False
barracks_rallied = False
~~~

해병을 생산하기 전, 병영의 유닛 집결 지점을 상대 공격을 막기 쉬운 입구 언덕 위쪽으로 설정합니다:
~~~
elif not self.barracks_rallied:
    if not self.barracks_selected:
        unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                
        if unit_y.any():
            target = [int(unit_x.mean()), int(unit_y.mean())]
                
            self.barracks_selected = True
                
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        else:
            self.barracks_rallied = True
                
            if self.base_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])
                
            return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])
~~~

병영이 선택하기 전까지 우리는 기다려야 합니다. `if unit_y():` 로 확인을 합니다. 언덕에 근접한 위치를 알냅니다.

유닛 집결 지점이 설정되었다면 해병 생산을 시작합니다:
~~~
elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_MARINE in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
~~~

테스트를 하기 위해서 충분한 시간을 기다려야 하며 2500 step 정도가 적당합니다.

여기까지의 소스코드는 [GitHub](https://github.com/Yudonggeun/PySC2-Tutorial/blob/master/4.%20Control%20Barrack%20and%20Army/building_marine.py)에서 확인하세요.

## 해병 컨트롤
새로운 상수를 추가합니다:
~~~
# Functions

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
~~~

해병 컨트롤을 위해 속성을 편성합니다:
~~~
    army_selected = False
    army_rallied = False
~~~

마지막으로 해병을 사용하여 공격합니다:
~~~
        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False
                
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                self.army_rallied = True
                self.army_selected = False
            
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])
            
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])
~~~

테스트를 하면 때때로 승리를 할 수도 못할 수도 있을 것입니다.
~~~
python -m pysc2.bin.agent --map Simple64 --agent attack_marine.SimpleAgent --agent_race T
~~~

여기 까지의 소스코드는 [GitHub](https://github.com/Yudonggeun/PySC2-Tutorial/blob/master/4.%20Control%20Barrack%20and%20Army/attack_marine.py)에서 확인하세요