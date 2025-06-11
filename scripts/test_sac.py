import os
os.environ["MUJOCO_GL"] = "glfw"  # 切换到 glfw 渲染，可在窗口显示渲染结果

import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import panda_mujoco_gym
from stable_baselines3.common.vec_env import DummyVecEnv
import time  # 用于控制渲染帧率

# 自定义包装器（与训练一致）
class TerminateOnTruncatedWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if truncated:
            terminated = True
        return obs, reward, terminated, truncated, info

# 创建与训练完全一致的环境，添加 render_mode 参数
def make_env(reward_type="sparse"):
    env = gym.make("FrankaPickAndPlaceSparse-v0", reward_type=reward_type, render_mode="human")
    env = TerminateOnTruncatedWrapper(env)
    return env

# 创建测试环境（必须使用 DummyVecEnv）
test_env = DummyVecEnv([lambda: make_env()])

# 加载模型（必须传入环境）
model = SAC.load("sac_franka_push_final", env=test_env)

# 运行测试
n_episodes = 20
success_count = 0

for episode in range(n_episodes):
    # VecEnv 返回列表形式的观测值
    obs = test_env.reset()
    done = False
    episode_success = False

    while not done:
        # VecEnv 动作需要是列表形式
        action, _ = model.predict(obs, deterministic=True)
        # VecEnv step 返回列表形式的结果
        results = test_env.step(action)

        # 处理不同返回值格式
        if len(results) == 4:
            obs, reward, done, info = results
            terminated = [done]
            truncated = [False]
        else:
            obs, reward, terminated, truncated, info = results

        # 确保 terminated/truncated 是列表
        if isinstance(terminated, bool):
            terminated = [terminated]
        if isinstance(truncated, bool):
            truncated = [truncated]

        done = terminated[0] or truncated[0]

        # 渲染环境
        test_env.render()
        time.sleep(0.1)  # 控制渲染帧率，可根据需要调整

        # 解析 info（VecEnv 返回列表）
        if isinstance(info, list):
            info = info[0]
        if info.get('is_success', False):
            episode_success = True

    if episode_success:
        success_count += 1

    print(f"Episode {episode + 1}: {'Success' if episode_success else 'Fail'}")

print(f"\nSuccess rate: {success_count / n_episodes * 100:.1f}%")
test_env.close()