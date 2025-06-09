import time

import gymnasium as gym
from stable_baselines3 import PPO
import panda_mujoco_gym

env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")

# 指定使用第二块GPU (索引从0开始)
device_id = 1  # 0表示第一块GPU，1表示第二块GPU
device = f"cuda:{device_id}"
model = PPO("MultiInputPolicy", env, verbose=1, device=device)
start_time = time.time()
model.learn(total_timesteps=100000)
model.save('model/PPO_pick_and_place.zip')
end_time = time.time()
# with open('ppo_pick_and_place_train_time.txt', 'w') as opener:
#     opener.write('spend_tine:{}'.format(end_time - start_time))

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
