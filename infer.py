import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
import numpy as np
import time
import panda_mujoco_gym
import argparse

def infer_naive():
    '''
    simple infer
    '''
    # 创建环境（注意渲染模式改为'human'以便实时显示）
    env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")

    # 加载训练好的模型
    model = PPO.load("model/PPO_pick_and_place.zip")

    # 重置环境
    obs, _ = env.reset()
    done = False
    success_count = 0
    episode_count = 0

    # 运行10个episode的推理演示
    while episode_count < 10:
        # 使用模型预测动作（deterministic=True 表示使用确定性策略）
        action, _ = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 渲染环境（自动显示在窗口）
        env.render()
        
        # 检查任务是否成功完成（根据环境特性调整）
        if reward > 0:  # 稀疏奖励环境中reward>0通常表示成功
            success_count += 1
            print(f"任务成功! 累计成功次数: {success_count}")
        
        # 重置环境当episode结束
        if done:
            episode_count += 1
            obs, _ = env.reset()
            print(f"Episode {episode_count} 完成")

    # 关闭环境
    env.close()
    print("推理演示结束")


def infer_dp(args):
    '''
    infer script for train_dp
    '''
    stats_path = os.path.join(args.eval_dir, "vec_normalize.pkl")
    if not os.path.exists(stats_path):
        print(f"[REEOR] no file in {stats_path}!!!")
        return 
    
    # 更稳妥的方式是创建一个 n_envs=1 的 VecEnv
    render_vec_env = DummyVecEnv([lambda: gym.make(args.env_id, render_mode="human")])
    render_env = VecNormalize.load(stats_path, render_vec_env)
    render_env.training = False # 必须设置为 False
    render_env.norm_reward = False # 通常在评估/渲染时设置为 False
    
    # 加载模型
    # model = PPO.load(best_model_path, env=render_env) # 把包装好的env传给模型
    best_model_path = os.path.join(args.eval_dir, "best_model/best_model.zip")
    model = PPO.load(best_model_path, env=render_env) # 把包装好的env传给模型
    print(f"Model loaded from {best_model_path}")

    # 运行和渲染
    max_steps_per_episode_render = 100
    obs = render_env.reset()
    for episode in range(args.n_eval_episodes):
        terminated = False
        truncated = False # For Gymnasium v0.26+
        total_reward = 0
        current_steps = 0 # 当前回合的步数计数器
        print(f"\nStarting rendering episode {episode + 1}")
        while not terminated or not truncated:
            action, _states = model.predict(obs, deterministic=True)
            # obs, reward, terminated, info = render_env.step(action) # 对于 VecEnv，返回的是数组
            obs, reward, terminated, info = render_env.step(action) # 对于 VecEnv，返回的是数组
            total_reward += reward[0] # VecEnv 返回的 reward 是数组

            current_steps += 1

            try:
                render_env.render()
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during rendering: {e}")
                break # 停止渲染循环

            terminated = terminated[0]
            truncated = info[0]["is_success"]
            
            if current_steps >= max_steps_per_episode_render:
                print(f"Episode reached max render steps ({max_steps_per_episode_render}), truncating.")
                truncated = True # 手动截断

        print(f"Episode {episode + 1} finished. Total reward: {total_reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        if episode < args.n_eval_episodes - 1:
            # obs_tuple_reset = render_env.reset()
            obs = render_env.reset()
    render_env.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="FrankaPickAndPlaceSparse-v0", type=str, help="env id")

    parser.add_argument("--n_eval_episodes", default=5, type=int, help="n_eval_episodes")
    parser.add_argument("--eval_dir", default="runs/Franka_PPO_DP-2025-06-05-16-36-43", type=str, help="eval model path")

    parser.add_argument("--is_train", default=True, type=bool, help="train or eval?")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    infer_dp(args)
