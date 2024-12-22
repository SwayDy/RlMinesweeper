import os
import argparse
import random
import time
import shutil

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from gymnasium import spaces
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from minesweeper import MinesweeperEnv_v1, MinesweeperEnv_v2
from agent import Agent_ppo_minesweeper, Agent_ppo_minesweeper_mobilenet_v3_large, \
    Agent_ppo_minesweeper_mobilenet_v3_small
from utils import eval_model, make_env

# 获取当前目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="ppo training")

    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__)[: -len(".py")],
                        help='the name of this experiment')
    parser.add_argument('--grid_size', type=lambda x: tuple(map(int, x.strip("()").split(","))), default=(9, 9),
                        help='the size of the env grid')
    parser.add_argument('--num_mines', type=int, default=10,
                        help='the number of mines in the env grid when testing and so on')
    parser.add_argument('--train_num_mines_range', type=lambda x: tuple(map(int, x.strip("()").split(","))),
                        help='the range of the number of mines in the env grid when training')
    parser.add_argument('--seed', type=int, default=278,
                        help='seed of the experiment')
    parser.add_argument('--torch_deterministic', type=lambda x: str(x).lower() == 'true', default=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: str(x).lower() == 'true', default=True,
                        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x: str(x).lower() == 'true', default=False,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb_project_name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--capture_video', action='store_true', default=False,
                        help='whether to capture videos of the agent performances (check out `videos` folder)')

    # Algorithm specific arguments
    parser.add_argument('--env_id', type=str, default="Minesweeper-v2",
                        help='the id of the environment chose from v1, v2')
    parser.add_argument('--model_id', type=str, default='Agent_ppo_minesweeper_mobilenet_v3_small',
                        help='the id of the model chose from general, mobilenet-small or large')
    parser.add_argument('--total_timesteps', type=int, default=10000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--num_envs', type=int, default=8,
                        help='the number of parallel game environments')
    parser.add_argument('--num_levels', type=int, default=8,
                        help='the number of parallel game environments while eval model')
    parser.add_argument('--num_steps', type=int, default=128,
                        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--pretrained', type=str,
                        help='the pretrained weight path of the agent')
    parser.add_argument('--freeze_weight', type=lambda x: str(x).lower() == 'true', default=False,
                        help='whether to freeze some weight of the agent')
    parser.add_argument('--eval_frequence', type=int, default=500,
                        help='time steps to save the model')
    parser.add_argument('--log_frequence', type=int, default=50000,
                        help='time steps to log the training info')
    parser.add_argument('--anneal_lr', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Toggle learning rate annealing for policy and value networks')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--num_minibatches', type=int, default=4,
                        help='the number of mini-batches')
    parser.add_argument('--update_epochs', type=int, default=4,
                        help='the K epochs to update the policy')
    parser.add_argument('--start_iter', type=int, default=1,
                        help='the iteration to start training')
    parser.add_argument('--norm_adv', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Toggles advantages normalization')
    parser.add_argument('--clip_coef', type=float, default=0.1,
                        help='the surrogate clipping coefficient')
    parser.add_argument('--clip_vloss', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='coefficient of the entropy')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='coefficient of the value function')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target_kl', type=float, default=None,
                        help='the target KL divergence threshold')

    # to be filled in runtime
    parser.add_argument('--batch_size', type=int, default=0,
                        help='the batch size (computed in runtime)')
    parser.add_argument('--minibatch_size', type=int, default=0,
                        help='the mini-batch size (computed in runtime)')
    parser.add_argument('--num_iterations', type=int, default=0,
                        help='the number of iterations (computed in runtime)')

    # args = parser.parse_known_args()[0]
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = (f"{args.model_id}__{args.env_id}__{args.exp_name}__{args.seed}__"
                f"{time.asctime().replace(' ', '-').replace(':', '-')}")
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if args.env_id == "Minesweeper-v1":
        if args.train_num_mines_range:
            a, b = args.train_num_mines_range
            envs = gym.vector.SyncVectorEnv(
                [make_env(
                    MinesweeperEnv_v1, grid_size=args.grid_size, 
                    num_mines=random.randint(a, b)
                ) for i in range(args.num_envs)]
            )  # 同时训练不同num_mines增强模型鲁棒性
        else:
            envs = gym.vector.SyncVectorEnv(
                [make_env(
                    MinesweeperEnv_v1, grid_size=args.grid_size, 
                    num_mines=args.num_mines
                ) for i in range(args.num_envs)],
            )

    elif args.env_id == "Minesweeper-v2":
        if args.train_num_mines_range:
            a, b = args.train_num_mines_range
            envs = gym.vector.SyncVectorEnv(
                [make_env(
                    MinesweeperEnv_v2, grid_size=args.grid_size,
                    num_mines=random.randint(a, b)
                ) for i in range(args.num_envs)]
            )
        else:
            envs = gym.vector.SyncVectorEnv(
                [make_env(
                    MinesweeperEnv_v2, grid_size=args.grid_size,
                    num_mines=args.num_mines
                ) for i in range(args.num_envs)]
            )
    else:
        raise NotImplementedError
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    if args.model_id == 'Agent_ppo_minesweeper':
        agent = Agent_ppo_minesweeper(envs).to(device)
        print(f"begin to train general agent.")
    elif args.model_id == 'Agent_ppo_minesweeper_mobilenet_v3_large':
        agent = Agent_ppo_minesweeper_mobilenet_v3_large(envs).to(device)
        print(f"begin to train mobilenet_v3_large agent")
    elif args.model_id == 'Agent_ppo_minesweeper_mobilenet_v3_small':
        agent = Agent_ppo_minesweeper_mobilenet_v3_small(envs).to(device)
        print(f"begin to train mobilenet_v3_samll agent")
    else:
        raise NotImplementedError

    if args.pretrained and os.path.exists(args.pretrained):
        checkpoint = torch.load(args.pretrained)
        agent.load_state_dict(checkpoint)
        print(f"load checkpoint: {args.pretrained}")

    # freeze network weight
    if args.freeze_weight:
        for para in agent.network.parameters():
            para.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(np.array(next_obs)).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # for model evaling and saving
    best_mr = -10e8
    best_wr = -1
    model_save_path = f"models/{run_name}"
    eval_index = 1
    tb_index = 1

    for iteration in range(args.start_iter, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(np.array(reward)).to(device).view(-1)
            next_obs, next_done = torch.Tensor(np.array(next_obs)).to(device), torch.Tensor(np.array(next_done)).to(
                device)

            if "final_info" in infos and global_step // args.log_frequence > tb_index:  # 减少log采样频率
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                tb_index += 1

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        with tqdm(total=args.update_epochs,
                  desc=f"Iteration {iteration}/{args.num_iterations} global steps {global_step}") as pbar:
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],
                                                                                  b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                pbar.set_postfix(
                    {'value_loss': '%.4f' % v_loss, 'policy_loss': '%.2f' % pg_loss, 'loss': '%.4f' % loss})
                pbar.update(1)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(agent.state_dict(), os.path.join(model_save_path, "last.pt"))
        print(f"Iter{iteration}: save model to {model_save_path}")

        # eval and save the model
        if global_step // args.eval_frequence > eval_index:
            if args.env_id == "Minesweeper-v1":
                mkEnv = make_env(MinesweeperEnv_v1, grid_size=args.grid_size, num_mines=args.num_mines, is_train=False)
            elif args.env_id == "Minesweeper-v2":
                mkEnv = make_env(MinesweeperEnv_v2, grid_size=args.grid_size, num_mines=args.num_mines, is_train=False)
            else:
                raise NotImplementedError

            mr, wr = eval_model(
                agent,
                gym.vector.SyncVectorEnv(
                    [mkEnv for _ in range(args.num_levels)]
                ),
                device=device
            )

            eval_index += 1
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            writer.add_scalar("charts/eval_mr", mr, global_step)
            writer.add_scalar("charts/eval_wr", wr, global_step)

            # torch.save(agent.state_dict(), os.path.join(model_save_path, "last.pt"))

            if mr > best_mr:
                best_mr = mr
                shutil.copy(os.path.join(model_save_path, "last.pt"),
                            os.path.join(model_save_path, "best_mr.pt"))

            if wr > best_wr:
                best_wr = wr
                shutil.copy(os.path.join(model_save_path, "last.pt"),
                            os.path.join(model_save_path, "best_wr.pt"))

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
