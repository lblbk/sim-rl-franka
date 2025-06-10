import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import wandb
import swanlab
from swanlab.integration.sb3 import SwanLabCallback
import numpy as np

import panda_mujoco_gym  # 注册环境
print(panda_mujoco_gym.__file__)

from datetime import datetime
import argparse

import torch

def get_formatted_time(format="%Y-%m-%d-%H-%M-%S"):
    return datetime.now().strftime(format)


def main(args):
    run_dir = f"{args.run_dir}/{args.exp_name}-{get_formatted_time()}"
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.device_id}"
    torch.cuda.set_device(args.device_id)

    # 自定义包装器
    class TerminateOnTruncatedWrapper(gym.Wrapper):
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if truncated:
                terminated = True
            return obs, reward, terminated, truncated, info

    env = DummyVecEnv([lambda: TerminateOnTruncatedWrapper(gym.make(args.env_id, reward_type=args.reward_type))])

    # 初始化 wandb
    # run = wandb.init(
    #     entity="***",
    #     project="***",
    #     config={
    #         "policy": "MultiInputPolicy",
    #         "learning_rate": 0.001,
    #         "buffer_size": 1_000_000,
    #         "batch_size": 512,
    #         "learning_starts": 2000,
    #         "policy_kwargs": {"net_arch": [256, 256, 256], "n_critics": 2},
    #         "replay_buffer_class": "HerReplayBuffer",
    #         "replay_buffer_kwargs": {"n_sampled_goal": 4, "goal_selection_strategy": "future"},
    #         "tau": 0.05,
    #         "gamma": 0.95,
    #         "verbose": 1,
    #         "top_quantiles_to_drop_per_net": 2
    #     }
    # )

    swanlab_callback = SwanLabCallback(
        project=args.exp_name,

        config={
            "policy": "MultiInputPolicy",
            "learning_rate": args.lr,
            "buffer_size": 1_000_000,
            "batch_size": args.batch_size,
            "learning_starts": 2000,
            "policy_kwargs": {"net_arch": [256, 256, 256], "n_critics": 2},
            "replay_buffer_class": "HerReplayBuffer",
            "replay_buffer_kwargs": {"n_sampled_goal": 4, "goal_selection_strategy": "future"},
            "tau": 0.05,
            "gamma": 0.95,
            "verbose": 1,
            "top_quantiles_to_drop_per_net": 2
        }
    )

    model = TQC(
        policy="MultiInputPolicy",
        env=env,
        device=device,
        learning_rate=args.lr,
        buffer_size=1_000_000,
        batch_size=args.batch_size,
        learning_starts=2000,
        policy_kwargs=dict(net_arch=[256, 256, 256], n_critics=2),
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        tau=0.05,
        gamma=0.95,
        verbose=1,
        top_quantiles_to_drop_per_net=2
    )


    class CustomEvalCallback(EvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.success_rates = []

        def _on_step(self) -> bool:
            result = super()._on_step()

            # 添加空列表保护
            if len(self.evaluations_results) == 0:
                return result

            # 仅在评估完成后记录
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                successes = []
                for episode_data in self.evaluations_results[-1]:
                    _, episode_infos = episode_data
                    if len(episode_infos) > 0:
                        success = episode_infos[-1].get('is_success', False)
                        successes.append(success)

                if len(successes) > 0:
                    success_rate = np.mean(successes)
                    swanlab.log({
                        "eval/success_rate": success_rate,
                        "global_step": self.num_timesteps
                    })

            return result

    eval_env = DummyVecEnv([lambda: Monitor(TerminateOnTruncatedWrapper(gym.make(args.env_id, reward_type=args.reward_type)), 
                                            os.path.join(run_dir, "eval_logs"))])
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        eval_freq=2000,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, swanlab_callback],
        tb_log_name="log_"+args.exp_name
    )

    model.save(os.path.join(run_dir, args.exp_name + "_final"))
    # run.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="Franka_TQC_DP", type=str, help="exp name")
    parser.add_argument("--env_id", default="FrankaPickAndPlaceSparse-v0", type=str, help="env id")
    parser.add_argument("--reward_type", default="sparse", type=str, help="reward type, can be 'dense' or 'sparse'")

    parser.add_argument("--device_id", default=1, type=int, help="gpu id")
    parser.add_argument("--batch_size", default=512, type=int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--total_timesteps", default=500_000, type=int, help="total_timesteps")

    parser.add_argument("--n_eval_episodes", default=15, type=int, help="n_eval_episodes")

    parser.add_argument("--run_dir", default="runs", type=str, help="this epoch runs dir name for save logs")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    '''
    nohup python scripts/train_tqc.py > logs/nuhup.out 2>&1 &
    '''
    args = parse_args()
    main(args)
