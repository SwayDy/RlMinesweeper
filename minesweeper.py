import random

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MinesweeperEnv(gym.Env):
    GRID_SIZE = (9, 9)  # 网格大小
    NUM_MINES = 10  # 地雷个数
    HIDDEN_STATE_VALUE = -1  # 隐藏状态值
    REMARK_STATE_VALUE = -2  # 标记状态值

    REWARDICT = {
        "mines_click": -2,
        "num_click": 1,
        "mines_remark": 1,
        "num_remark": -1,
        "remark_click": -1,
        "invalid_action": -1,
        "win_game": 3
    }

    def __init__(self, grid_size=GRID_SIZE, num_mines=NUM_MINES, hidden_state_value=HIDDEN_STATE_VALUE,
                 remark_state_value=REMARK_STATE_VALUE, reward_dict=REWARDICT, is_train=True, is_render=False, cell_size=50):
        super(MinesweeperEnv, self).__init__()

        self._first_click = (0, 0)  # 第一次点击的格子

        self.grid_size = grid_size  # 网格大小
        self.num_mines = num_mines  # 地雷个数
        self.hidden_state_value = hidden_state_value
        self.remark_state_value = remark_state_value
        self.reward_dict = reward_dict
        self.is_train = is_train
        self.is_render = is_render  # 是否可视化游戏界面

        # 定义动作空间
        self.action_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])

        # 定义观察空间
        self.observation_space = spaces.Box(
            low=np.exp(-8),
            high=np.exp(2),
            shape=(2,) + self.grid_size,
            dtype=np.float32,
        )

        # 初始化
        self._board = np.zeros(self.grid_size, dtype=np.int32)
        self._num_state = np.full(self.grid_size, self.hidden_state_value, dtype=np.int32)  # 所有格子初始隐藏
        self._count_state = np.zeros(self.grid_size, dtype=np.int32)  # 格子点击次数
        self._mines = set()  # 地雷位置
        self.terminations = False  # 游戏是否结束
        self.truncations = False
        self.info = {"episode": {"r": 0, "l": 0, "h": self.grid_size[0]*self.grid_size[1]}}
        self.state = np.stack((self._num_state, self._count_state), axis=0)

        self._directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        if self.is_render:
            self.cell_size = cell_size
            self.screen_size = (self.grid_size[0] * cell_size, self.grid_size[1] * cell_size)
            self._initialize_pygame()

    def _initialize_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Minesweeper")
        self.font = pygame.font.SysFont("Arial", self.cell_size // 2)
        self.colors = {
            "hidden": (192, 192, 192),
            "revealed": (255, 255, 255),
            "mine": (255, 0, 0),
            "flag": (0, 255, 0),
            "border": (128, 128, 128),
            "text": (0, 0, 0)
        }

    def _reveal(self, x, y):
        if self._num_state[x, y] != -1:
            return
        self._num_state[x, y] = self._board[x, y]
        self._count_state[x, y] += 1
        if self._board[x, y] == 0:
            # directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dx, dy in self._directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self._reveal(nx, ny)

    def reset(self, seed=None, options=None):
        self._board = np.zeros(self.grid_size, dtype=np.int32)
        self._num_state = np.full(self.grid_size, self.hidden_state_value, dtype=np.int32)  # 所有格子初始隐藏
        self._count_state = np.zeros(self.grid_size, dtype=np.int32)  # 格子点击次数
        self._mines = set()
        # 生成地雷
        while len(self._mines) < self.num_mines:
            x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
            # 第一次点击不会是地雷
            if self._first_click == (x, y):
                continue
            if (x, y) not in self._mines:
                self._mines.add((x, y))

        # 生成成数字
        # directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for x, y in self._mines:
            self._board[x, y] = -1
            for dx, dy in self._directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and (nx, ny) not in self._mines:
                    self._board[nx, ny] += 1

        self.terminations = False
        self.truncations = False
        self.info = {"episode": {"r": 0, "l": 0, "h": self.grid_size[0]*self.grid_size[1]}}

        self.state = np.stack((np.exp(-self._num_state), self._count_state), axis=0)

        if self.is_render:
            self._initialize_pygame()

        return self.state.copy(), self.info

    def step(self, action):
        reward = 0
        if isinstance(action, tuple):
            action_type, x, y = action[0], action[1], action[2]
        else:
            action_type, x, y = 0, action // self.grid_size[0], action % self.grid_size[1]

        if self.terminations or self.truncations:
            raise RuntimeError("Game is already over. Please reset the environment.")

        if action_type == 0:  # 点击动作
            # self._count_state[x, y] += 1
            if (x, y) in self._mines:  # 点击到地雷
                # 评估时点击到地雷结束游戏
                if not self.is_train:
                    self.terminations = True
                    self.info["game_state"] = "lose"
                # 训练时点击到地雷不结束游戏
                if self._num_state[x, y] == self.remark_state_value:  # 点击已标记的格子
                    reward = self.reward_dict.get("mines_click", 0) + self.reward_dict.get("remark_click", 0)
                else:
                    reward = self.reward_dict.get("mines_click", 0)

                hidden_state_count = np.sum((self._num_state == -1) + (self._num_state == -2))  # 当前未展开格子数
                reward_weight = (self.grid_size[0] * self.grid_size[1] - hidden_state_count) / (
                            self.grid_size[0] * self.grid_size[1])

                reward *= reward_weight

                self._num_state[x, y] = self._board[x, y]
                self._count_state[x, y] += 1
            # 点不到地雷的情况
            elif self._num_state[x, y] == self.hidden_state_value:  # 点击隐藏的格子
                if self._board[x, y] == 0:  # 点击到数字为0的格子
                    reward = self.reward_dict.get("num_click", 0)

                    hidden_state_count = np.sum((self._num_state == -1) + (self._num_state == -2))  # 当前未展开格子数
                    reward_weight = self.num_mines / hidden_state_count

                    reward *= reward_weight

                    self._reveal(x, y)  # 递归展开
                elif self._board[x, y] > 0:  # 点击到数字的格子
                    reward = self.reward_dict.get("num_click", 0)

                    hidden_state_count = np.sum((self._num_state == -1) + (self._num_state == -2))  # 当前未展开格子数
                    reward_weight = self.num_mines / hidden_state_count

                    reward *= reward_weight

                    self._num_state[x, y] = self._board[x, y]
                    self._count_state[x, y] += 1
                else:
                    print("Warning!")
            elif self._num_state[x, y] == self.remark_state_value:  # 点击已标记的格子
                if self._board[x, y] == 0:  # 点击到数字为0的格子
                    self._reveal(x, y)  # 递归展开
                self._num_state[x, y] = self._board[x, y]
                self._count_state[x, y] += 1
                reward = self.reward_dict.get("remark_click", 0) + self.reward_dict.get("num_click", 0)
            elif self._num_state[x, y] >= 0:  # 点击已展开的格子
                self._count_state[x, y] += 1
                reward = self.reward_dict.get("invalid_action", 0)
            else:
                print("Warning!")
        elif action_type == 1:  # 标记动作
            if self._num_state[x, y] == self.hidden_state_value:  # 标记隐藏的格子
                self._num_state[x, y] = self.remark_state_value  # 将状态更换为已标记
                reward = self.reward_dict.get("mines_remark", 0) if (x, y) in self._mines else self.reward_dict.get(
                    "num_remark", 0)
            else:  # 标记展开的格子
                reward = self.reward_dict.get("invalid_action")
        else:
            print("Warning!")
            reward = 0

        if np.sum(self._num_state == self.hidden_state_value) + np.sum(
                self._num_state == self.remark_state_value) == len(self._mines):  # 隐藏状态下的格子数等于地雷总数
            self.terminations = True
            self.info["game_state"] = "win"
            # 游戏胜利
            reward += self.reward_dict.get("win_game", 0)

        num_state_copy = self._num_state.copy()
        num_state_copy[num_state_copy == -2] = -1
        self.state = np.stack((np.exp(-num_state_copy), self._count_state), axis=0)

        self.info["episode"]["r"] += reward
        self.info["episode"]["l"] += 1
        self.info["episode"]["h"] = np.sum((self._num_state == -1) + (self._num_state == -2))

        # 当动作长度大于数字网格数时进行截断
        if self.info["episode"]["l"] >= self.grid_size[0] * self.grid_size[1] - self.num_mines:
            self.truncations = True

        # next_obs, reward, terminations, truncations, infos

        return self.state.copy(), reward, self.terminations, self.truncations, self.info

    def render(self):
        if not self.is_render:
            return

        self.screen.fill(self.colors["border"])

        if self.terminations:
            self._num_state = self._board

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                if self._num_state[x, y] == -1:  # 隐藏
                    pygame.draw.rect(self.screen, self.colors["hidden"], rect)
                elif self._num_state[x, y] == -2:  # 标记
                    pygame.draw.rect(self.screen, self.colors["hidden"], rect)
                    pygame.draw.circle(self.screen, self.colors["flag"], rect.center, self.cell_size // 4)
                else:  # 已点击显示数字
                    pygame.draw.rect(self.screen, self.colors["revealed"], rect)
                    if self._num_state[x, y] > 0:
                        text = self.font.render(str(self._num_state[x, y]), True, self.colors["text"])
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                if self._board[x, y] == -1 and self.terminations:  # 游戏结束显示地雷
                    pygame.draw.rect(self.screen, self.colors["hidden"], rect)
                    pygame.draw.circle(self.screen, self.colors["mine"], rect.center, self.cell_size // 4)

                pygame.draw.rect(self.screen, self.colors["border"], rect, 2)

        # 游戏结束时显示游戏结果
        if self.terminations:
            font = pygame.font.SysFont(None, self.cell_size)
            text = font.render("you " + self.info["game_state"] + "!", True, self.colors["text"])
            text_rect = text.get_rect(center=((self.grid_size[1] * self.cell_size) // 2,
                                              (self.grid_size[0] * self.cell_size) // 2))
            self.screen.blit(text, text_rect)

        pygame.display.flip()


class MinesweeperEnv_v1(gym.Env):
    GRID_SIZE = (9, 9)  # 网格大小
    NUM_MINES = 10  # 地雷个数
    HIDDEN_STATE_VALUE = -1  # 隐藏状态值

    REWARDICT = {
        "mines_click": -2,
        "num_click": 1,
        "invalid_action": -1,
        "win_game": 3
    }
    def __init__(self, grid_size=GRID_SIZE, num_mines=NUM_MINES, hidden_state_value=HIDDEN_STATE_VALUE,
                 reward_dict=REWARDICT, is_train=True, *args, **kwargs):
        super(MinesweeperEnv_v1, self).__init__()

        self.grid_size = grid_size  # 网格大小
        self.num_mines = num_mines  # 地雷个数
        self.hidden_state_value = hidden_state_value
        self.reward_dict = reward_dict
        self.is_train = is_train

        # 定义动作空间
        self.action_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])

        # 定义观察空间
        self.observation_space = spaces.Box(
            low=np.exp(-8),
            high=np.exp(2),
            shape=(2,) + self.grid_size,
            dtype=np.float32,
        )

        # 初始化
        self._board = np.zeros(self.grid_size, dtype=np.int32)
        self._num_state = np.full(self.grid_size, self.hidden_state_value, dtype=np.int32)  # 所有格子初始隐藏
        self._count_state = np.zeros(self.grid_size, dtype=np.int32)  # 格子点击次数
        self._mines = set()  # 地雷位置
        self.terminations = False  # 游戏是否结束
        self.truncations = False
        self.info = {"episode": {"r": 0, "l": 0, "h": self.grid_size[0]*self.grid_size[1]}}
        self.state = np.stack((self._num_state, self._count_state), axis=0)

        self._directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def _reveal(self, x, y):
        if self._num_state[x, y] != -1:
            return
        self._num_state[x, y] = self._board[x, y]
        self._count_state[x, y] += 1
        if self._board[x, y] == 0:
            # directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dx, dy in self._directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self._reveal(nx, ny)

    def reset(self, seed=None, options=None):
        self._board = np.zeros(self.grid_size, dtype=np.int32)
        self._num_state = np.full(self.grid_size, self.hidden_state_value, dtype=np.int32)  # 所有格子初始隐藏
        self._count_state = np.zeros(self.grid_size, dtype=np.int32)  # 格子点击次数
        self._mines = set()
        # 生成地雷
        while len(self._mines) < self.num_mines:
            x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
            if (x, y) not in self._mines:
                self._mines.add((x, y))

        # 生成成数字
        # directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for x, y in self._mines:
            self._board[x, y] = -1
            for dx, dy in self._directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and (nx, ny) not in self._mines:
                    self._board[nx, ny] += 1

        self.terminations = False
        self.truncations = False
        self.info = {"episode": {"r": 0, "l": 0, "h": self.grid_size[0]*self.grid_size[1]}}

        self.state = np.stack((np.exp(-self._num_state), self._count_state), axis=0)

        return self.state.copy(), self.info

    def step(self, action):
        reward = 0
        if isinstance(action, tuple):
            x, y = action[1], action[2]
        else:
            x, y = action // self.grid_size[0], action % self.grid_size[1]

        if self.terminations or self.truncations:
            raise RuntimeError("Game is already over. Please reset the environment.")

        # self._count_state[x, y] += 1
        if (x, y) in self._mines:  # 点击到地雷
            # 评估时点击到地雷结束游戏
            if not self.is_train:
                self.terminations = True
                self.info["game_state"] = "lose"
            # 训练时点击到地雷不结束游戏
            reward = self.reward_dict.get("mines_click", 0)

            hidden_state_count = np.sum((self._num_state == -1) + (self._num_state == -2))  # 当前未展开格子数
            reward_weight = (self.grid_size[0] * self.grid_size[1] - hidden_state_count) / (
                        self.grid_size[0] * self.grid_size[1])

            reward *= reward_weight

            self._num_state[x, y] = self._board[x, y]
            self._count_state[x, y] += 1
        # 点不到地雷的情况
        elif self._num_state[x, y] == self.hidden_state_value:  # 点击隐藏的格子
            if self._board[x, y] == 0:  # 点击到数字为0的格子
                reward = self.reward_dict.get("num_click", 0)

                hidden_state_count = np.sum((self._num_state == -1) + (self._num_state == -2))  # 当前未展开格子数
                reward_weight = self.num_mines / hidden_state_count

                reward *= reward_weight

                self._reveal(x, y)  # 递归展开
            elif self._board[x, y] > 0:  # 点击到数字的格子
                reward = self.reward_dict.get("num_click", 0)

                hidden_state_count = np.sum((self._num_state == -1) + (self._num_state == -2))  # 当前未展开格子数
                reward_weight = self.num_mines / hidden_state_count

                reward *= reward_weight

                self._num_state[x, y] = self._board[x, y]
                self._count_state[x, y] += 1
            else:
                raise RuntimeWarning('invalid action!')
        elif self._num_state[x, y] >= 0:  # 点击已展开的格子
            self._count_state[x, y] += 1
            reward = self.reward_dict.get("invalid_action", 0)
        else:
            raise RuntimeWarning('invalid action!')

        if np.sum(self._num_state == self.hidden_state_value) == len(self._mines):  # 隐藏状态下的格子数等于地雷总数
            self.terminations = True
            self.info["game_state"] = "win"
            # 游戏胜利
            reward += self.reward_dict.get("win_game", 0)

        num_state_copy = self._num_state.copy()
        num_state_copy[num_state_copy == -2] = -1
        self.state = np.stack((np.exp(-num_state_copy), self._count_state), axis=0)

        self.info["episode"]["r"] += reward
        self.info["episode"]["l"] += 1
        self.info["episode"]["h"] = np.sum((self._num_state == -1) + (self._num_state == -2))

        # 当动作长度大于数字网格数时进行截断
        if self.info["episode"]["l"] >= self.grid_size[0] * self.grid_size[1] - self.num_mines:
            self.truncations = True

        # next_obs, reward, terminations, truncations, infos

        return self.state.copy(), reward, self.terminations, self.truncations, self.info


class MinesweeperEnv_v2(gym.Env):
    GRID_SIZE = (9, 9)  # 网格大小
    NUM_MINES = 10  # 地雷个数
    HIDDEN_STATE_VALUE = -1  # 隐藏状态值

    REWARDICT = {
        "mines_click": -2,
        "num_click": 1,
        "invalid_action": -1,
        "win_game": 3
    }
    def __init__(self, grid_size=GRID_SIZE, num_mines=NUM_MINES, hidden_state_value=HIDDEN_STATE_VALUE,
                 reward_dict=REWARDICT, is_train=True, *args, **kwargs):
        super(MinesweeperEnv_v2, self).__init__()

        self.grid_size = grid_size  # 网格大小
        self.num_mines = num_mines  # 地雷个数
        self.hidden_state_value = hidden_state_value
        self.reward_dict = reward_dict

        # 定义动作空间
        self.action_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])

        # 定义观察空间
        self.observation_space = spaces.Box(
            low=np.exp(-8),
            high=np.exp(2),
            shape=(2,) + self.grid_size,
            dtype=np.float32,
        )

        # 初始化
        self._board = np.zeros(self.grid_size, dtype=np.int32)
        self._num_state = np.full(self.grid_size, self.hidden_state_value, dtype=np.int32)  # 所有格子初始隐藏
        self._count_state = np.zeros(self.grid_size, dtype=np.int32)  # 格子点击次数
        self._mines = set()  # 地雷位置
        self.terminations = False  # 游戏是否结束
        self.truncations = False
        self.info = {"episode": {"r": 0, "l": 0, "h": self.grid_size[0]*self.grid_size[1]}}
        self.state = np.stack((np.exp(-self._num_state), self._count_state), axis=0)

        self._directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def _reveal(self, x, y):
        if self._num_state[x, y] != -1:
            return
        self._num_state[x, y] = self._board[x, y]
        self._count_state[x, y] += 1
        if self._board[x, y] == 0:
            # directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dx, dy in self._directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self._reveal(nx, ny)

    def reset(self, seed=None, options=None):
        self._board = np.zeros(self.grid_size, dtype=np.int32)
        self._num_state = np.full(self.grid_size, self.hidden_state_value, dtype=np.int32)  # 所有格子初始隐藏
        self._count_state = np.zeros(self.grid_size, dtype=np.int32)  # 格子点击次数
        self._mines = set()
        # 生成地雷
        while len(self._mines) < self.num_mines:
            x, y = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
            if (x, y) not in self._mines:
                self._mines.add((x, y))

        # 生成成数字
        # directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for x, y in self._mines:
            self._board[x, y] = -1
            for dx, dy in self._directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and (nx, ny) not in self._mines:
                    self._board[nx, ny] += 1

        self.terminations = False
        self.truncations = False
        self.info = {"episode": {"r": 0, "l": 0, "h": self.grid_size[0]*self.grid_size[1]}}

        self.state = np.stack((np.exp(-self._num_state), self._count_state), axis=0)

        return self.state.copy(), self.info

    def step(self, action):
        reward = 0
        x, y = action // self.grid_size[0], action % self.grid_size[1]

        hidden_state_count = np.sum(self._num_state == -1)  # 当前未展开格子数

        if self.terminations or self.truncations:
            raise RuntimeError("Game is already over. Please reset the environment.")

        # self._count_state[x, y] += 1
        if (x, y) in self._mines:  # 点击到地雷
            self.terminations = True
            self.info["game_state"] = "lose"
            reward_weight = (self.grid_size[0] * self.grid_size[1] - hidden_state_count) / (
                        self.grid_size[0] * self.grid_size[1])

            # reward = self.reward_dict.get("mines_click", 0) * reward_weight
            reward = -self.info["episode"]['r'] - reward_weight  # 点到地雷则将这个episode的return赋值为负值

            self._num_state[x, y] = self._board[x, y]
            self._count_state[x, y] += 1
        # 点不到地雷的情况
        elif self._num_state[x, y] == self.hidden_state_value:  # 点击隐藏的格子
            if self._board[x, y] == 0:  # 点击到数字为0的格子
                self._reveal(x, y)  # 递归展开
            elif self._board[x, y] > 0:  # 点击到数字的格子
                self._num_state[x, y] = self._board[x, y]
                self._count_state[x, y] += 1
            else:
                raise RuntimeWarning("")
            reward_weight = self.num_mines / hidden_state_count

            AVERAGE_LENGTH = 6
            AVERAGE_HSC = 44

            if self.info['episode']['l'] > AVERAGE_LENGTH:  # 大于有效长度时开始给予奖励
                reward = self.reward_dict.get("num_click", 0) * reward_weight
            else:
                reward = 0
        elif self._num_state[x, y] >= 0:  # 点击已展开的格子
            self._count_state[x, y] += 1
            reward = self.reward_dict.get("invalid_action", 0)
        else:
            raise RuntimeWarning("")

        if np.sum(self._num_state == self.hidden_state_value) == len(self._mines):  # 判断胜利条件：隐藏状态下的格子数等于地雷总数
            self.terminations = True
            self.info["game_state"] = "win"
            # 游戏胜利
            reward += self.reward_dict.get("win_game", 0)

        num_state_copy = self._num_state.copy()
        self.state = np.stack((np.exp(-num_state_copy), self._count_state), axis=0)

        self.info["episode"]["r"] += reward
        self.info["episode"]["l"] += 1
        self.info["episode"]["h"] = np.sum(self._num_state == -1)

        # 当动作长度大于数字网格数时进行截断
        if self.info["episode"]["l"] >= self.grid_size[0] * self.grid_size[1] - self.num_mines:
            self.truncations = True

        return self.state.copy(), reward, self.terminations, self.truncations, self.info
