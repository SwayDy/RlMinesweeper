import time

import numpy as np
import gymnasium as gym
import torch
import torch.multiprocessing as mp

from tqdm import tqdm

from utils import make_env
from agent import Agent_ppo_minesweeper
from minesweeper import MinesweeperEnv_v1


def eval_model(
        agent: torch.nn.Module,
        envs: gym.vector.SyncVectorEnv,
        times: int = 1000,
        device: torch.device = torch.device("cpu"),
        is_ellipsis: bool = True
):
    agent.eval()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(np.array(next_obs)).to(device)

    total_reward = 0
    win_count = 0
    count = 0

    with tqdm(total=times, desc="test...") as pbar:
        for _ in range(times):
            with torch.no_grad():
                action = agent.get_action(next_obs)
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(np.array(next_obs)).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        total_reward += info["episode"]["r"]
                        win_count += 1 if info.get("game_state", None) == "win" else 0
                        count += 1

            if not is_ellipsis:
                count = 1 if count == 0 else count
                pbar.set_postfix({'game count': f'{count}', 'average_return': f'{total_reward / count:.4f}',
                                  'win_rate': f'{win_count / count:.4f}'})
                pbar.update(1)

        count = 1 if count == 0 else count
        pbar.set_postfix({'game count': f'{count}', 'average_return': f'{total_reward / count:.4f}',
                          'win_rate': f'{win_count / count:.4f}'})
        pbar.update(times)

    # return total_reward / count, win_count / count
    return total_reward, win_count, count

def worker(rank, model_path, device, num_envs, num_episodes, result_queue):
    envs = gym.vector.SyncVectorEnv(
        [make_env(MinesweeperEnv_v1, is_train=False) for _ in range(num_envs)]
    )

    agent = Agent_ppo_minesweeper(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))

    # avg_return, win_rate = eval_model(agent, envs, times=num_episodes, device=device, is_ellipsis=False)
    total_reward, win_count, count = eval_model(agent, envs, times=num_episodes, device=device, is_ellipsis=False)

    # result_queue.put((avg_return, win_rate))
    result_queue.put((total_reward, win_count, count))

def aggregate_results(result_queue, num_processes):
    # total_avg_return = 0
    # total_win_rate = 0
    total_return = 0
    total_win_count = 0
    total_count = 0

    for _ in range(num_processes):
        # avg_return, win_rate = result_queue.get()
        # total_avg_return += avg_return
        # total_win_rate += win_rate
        total_reward, win_count, count = result_queue.get()
        total_return += total_reward
        total_win_count += win_count
        total_count += count

    # total_avg_return /= num_processes
    # total_win_rate /= num_processes
    total_avg_return = total_return / total_count
    total_win_rate = total_win_count / total_count

    # print(f"Aggregated Results: Average Return = {total_avg_return:.4f}, Win Rate = {total_win_rate:.4f}")
    print(f"Aggregated Results: Average Return = {total_avg_return:.4f}, Win Rate = {total_win_rate:.4f}, Total Count = {total_count}")

    return total_avg_return, total_win_rate

if __name__ == "__main__":
    mp.set_start_method("spawn")

    num_processes = 8
    model_path = "models/agent_9.2711_0.7148.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 128
    num_episodes = 8000 // num_processes  # Number of episodes per process

    result_queue = mp.Queue()

    start_time = time.time()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=worker, args=(rank, model_path, device, num_envs, num_episodes, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    aggregate_results(result_queue, num_processes)
    print(f"Total time: {time.time() - start_time:.4f} seconds")
