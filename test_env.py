import json

import numpy as np
import torch
import pygame
click = pygame.time.Clock()
import gymnasium as gym

from tqdm import tqdm

from agent import Agent_ppo_minesweeper
from minesweeper import MinesweeperEnv


def human_click(env, is_repeat=True):
    if not env.is_render:
        raise ValueError("env.is_render must be True.")
    # human click
    state, _ = env.reset()
    env.render()
    done = False
    action = None

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                is_repeat = False
                done = True

            if event.type == pygame.MOUSEBUTTONDOWN and not done:
                y, x = event.pos[0] // env.cell_size, event.pos[1] // env.cell_size
                if event.button == 1:  # 左键点击
                    action = (0, x, y)
                elif event.button == 3:  # 右键点击
                    action = (1, x, y)
            if action:
                next_state, reward, done, _, info = env.step(action)
                env.render()
                state = next_state
                action = None
        if done and is_repeat:
            pygame.time.wait(2000)

            env.reset()
            env.render()
            done = False
            action = None


def agent_click(env, agent, is_repeat=True):
    # agent click
    is_render = env.is_render

    if not is_render:
        print("Warning!:env.is_render is False so that video system not initialized.")

    total_reward = 0
    win_count = 0
    len_count = 0
    hsc = 0
    max_epochs = 100
    t = 0

    with tqdm(total=max_epochs, desc="test!...") as pbar:
        # 初始化
        env.reset()
        # 人为选择第一次点击(0, 0)格子
        next_state, reward, done, _, info = env.step(0)
        env.render()
        state = next_state

        while not done and t < max_epochs:
            # action = (0, random.randint(0, env.grid_size[0] - 1), random.randint(0, env.grid_size[1] - 1))
            # while state[1][action[1], action[2]]:
            #     action = (0, random.randint(0, env.grid_size[0] - 1), random.randint(0, env.grid_size[1] - 1))
            # action = action[1] * env.grid_size[0] + action[2]
            with torch.no_grad():
                action = agent.get_action(torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0))[
                    0].cpu().item()

            next_state, reward, done, _, info = env.step(action)
            env.render()
            # click.tick(1)

            # print(f"action: {action}, reward: {reward}")

            state = next_state
            if is_render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        done = True
                        is_repeat = False
            if done and is_repeat:
                # print(f"epoch: {t}, return: {info.get('episode').get('r')}, result: {info.get('game_state')}")

                # 汇总结果
                total_reward += info.get('episode').get('r')
                len_count += info.get('episode').get('l')
                hsc += info.get('episode').get('h')
                win_count += 1 if info.get("game_state", None) == "win" else 0

                pygame.time.wait(2000)

                # 进入下一次epochs
                epoch_reward = 0
                t += 1
                pbar.set_postfix({'return': '%.4f' % info.get('episode').get('r'), 'result': info.get('game_state')})
                pbar.update(1)

                # 再次初始化
                env.reset()
                # 人为选择第一次点击(0, 0)格子
                next_state, reward, done, _, info = env.step(0)
                env.render()
                state = next_state

    print(f"average_return: {total_reward / max_epochs} "
          f"average_len: {len_count / max_epochs} "
          f"average_hsc: {hsc / max_epochs} "
          f"win_rate: {win_count / max_epochs}")


if __name__ == "__main__":
    # is_repeat = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MinesweeperEnv(is_render=False, is_train=False)

    agent = Agent_ppo_minesweeper(
        gym.vector.SyncVectorEnv([lambda: MinesweeperEnv() for i in range(1)])
    ).to(device)
    agent.load_state_dict(torch.load("models/agent_9.2711_0.7148.pt"))

    # human_click(env)
    agent_click(env, agent)
