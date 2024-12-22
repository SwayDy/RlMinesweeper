import json

import gymnasium as gym
import numpy as np
import torch


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def eval_model(
        agent: torch.nn.Module,
        envs: gym.vector.SyncVectorEnv,
        times: int=1000,
        device: torch.device=torch.device("cpu"),
        is_ellipsis: bool=True,
        env_id: str="Minesweeper-v1"
):

    # env = MinesweeperEnv(is_render=False, is_train=False)

    agent.eval()
    # is_train=False

    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(np.array(next_obs)).to(device)

    total_reward = 0
    win_count = 0
    count = 0

    from tqdm import tqdm

    with tqdm(total=times, desc="test...") as pbar:
        for i in range(times):
            with torch.no_grad():
                action = agent.get_action(next_obs)
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            next_obs = torch.Tensor(np.array(next_obs)).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        total_reward += info["episode"]["r"]
                        win_count += 1 if info.get("game_state", None) == "win" else 0
                        count += 1  # 每完整的进行一次游戏加一

            if not is_ellipsis:  # 打印输出
                count = 1 if count == 0 else count
                pbar.set_postfix({'game count': '%d' % count, 'average_return': '%.4f' % (total_reward / count),
                                    'win_rate': '%.4f' % (win_count / count)})
                pbar.update(1)

        count = 1 if count == 0 else count
        pbar.set_postfix({'game count': '%d' % count, 'average_return': '%.4f' % (total_reward / count),
                            'win_rate': '%.4f' % (win_count / count)})
        pbar.update(times)

    return total_reward / count,  win_count / count


def make_env(fnEnv, grid_size=(9, 9), num_mines=10, is_train=True):
    def _thunk():
        return fnEnv(grid_size=grid_size, num_mines=num_mines, is_train=is_train)

    return _thunk


if __name__ == "__main__":
    from agent import Agent_ppo_minesweeper
    from minesweeper import MinesweeperEnv_v1
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [lambda: MinesweeperEnv_v1(is_train=False) for _ in range(128)]
    )

    agent = Agent_ppo_minesweeper(envs).to(device)
    agent.load_state_dict(torch.load("models/last.pt"))

    start_time = time.time()
    eval_model(agent, envs, device=device, is_ellipsis=False)
    print(f"Total time: {time.time() - start_time:.4f} seconds")
