import os
from datetime import datetime
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import torch
import argparse
import panda_mujoco_gym

import swanlab  # ref wandb https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html
from swanlab.integration.sb3 import SwanLabCallback

def get_formatted_time(format="%Y-%m-%d-%H-%M-%S"):
    return datetime.now().strftime(format)

def get_policy_kwargs(args):
    """
    获取策略网络的参数
    这里可以根据需要返回不同的网络结构
    """
    policy_kwargs = None
    if "PPO" in args.exp_name:
        policy_kwargs = dict(
            pi=[256, 256],  # 策略网络的结构
            vf=[256, 256]   # Critic (value) network
        )
    if "SAC" in args.exp_name:
        policy_kwargs = dict(
            pi=[256, 256],  # 策略网络的结构
            qf=[256, 256]   # Q函数网络的结构 (SAC中使用)
        )
    return policy_kwargs

def get_model(env, device, run_dir, policy_kwargs, args):
    """
    获取模型的函数
    这里可以根据需要返回不同的模型
    """
    model = None
    if "PPO" in args.exp_name:
        model = PPO(
            "MultiInputPolicy", 
            env, 
            policy_kwargs=policy_kwargs,
            verbose=1, 
            device=device,
            tensorboard_log=run_dir,

            # 优化参数 (根据环境调整)
            n_steps=2048//args.num_envs,    # 每个环境收集的步数
            batch_size=args.batch_size,     # 批次大小
            n_epochs=args.n_epochs,         # 每个数据收集阶段的优化轮数
            gamma=0.99,                     # 折扣因子
            gae_lambda=0.95,                # GAE参数
            ent_coef=0.01,                  # 熵系数
            learning_rate=args.lr,
            )
    if "SAC" in args.exp_name:
        model = SAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,

            buffer_size=1_000_000,          # 一个大的回放缓冲区
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gamma=0.99,
            # train_freq 控制模型更新频率，可以是步数或 episode
            train_freq=(1, "step"), 
            gradient_steps=1,               # 每次更新的梯度步数
            learning_starts=10000,          # 在收集足够多的样本后再开始训练
            verbose=1)
        
    return model

def train(args):
    run_dir = f"{args.run_dir}/{args.exp_name}-{get_formatted_time()}"
    os.makedirs(run_dir, exist_ok=True)

    # 指定 GPU (索引从0开始)
    device = f"cuda:{args.device_id}"
    torch.cuda.set_device(args.device_id)

    # 使用更高效的方法创建向量化环境
    env = make_vec_env(
        args.env_id,
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv,  # 使用子进程并行
        vec_env_kwargs={"start_method": "fork"},  # 在Linux上使用"fork"，Windows用"spawn"
        env_kwargs={"render_mode": "rgb_array"},
        seed=args.seed
    )

    # 添加监控和标准化 (可选但推荐)
    env = VecMonitor(env, filename=os.path.join(run_dir, "monitor.csv"))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, training=True)

    # 定义策略网络的参数
    policy_kwargs = get_policy_kwargs(args)
    if policy_kwargs is None:
        print("[ERROR] policy_kwargs not initialized!")
        return

    model = get_model(env, device, run_dir, policy_kwargs=policy_kwargs, args=args)
    if model is None:
        print("[ERROR] Model not initialized, check your exp_name!")
        return

    eval_env = make_vec_env(
        args.env_id,
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": "rgb_array"},
        seed=args.seed + 100
    )

    eval_env = VecMonitor(eval_env, filename=os.path.join(run_dir, "monitor_eval.csv"))
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    # 回调函数
    class ReduceLROnPlateauCallback(BaseCallback):
        """
        当评估奖励停滞时，降低学习率。
        必须与 EvalCallback 一起使用。
        
        :param patience: (int) 连续多少次评估性能没有提升后降低学习率。
        :param factor: (float) 学习率衰减因子。
        :param min_lr: (float) 学习率的下限。
        """
        def __init__(self, patience: int, factor: float = 0.5, min_lr: float = 1e-6, verbose: int = 0):
            super(ReduceLROnPlateauCallback, self).__init__(verbose)
            self.patience = patience
            self.factor = factor
            self.min_lr = min_lr
            self.patience_counter = 0 # 耐心计数器
            # EvalCallback 会更新 self.parent.best_model_save_path
            # 我们可以通过检查它是否被更新来判断模型是否有提升
            self.last_best_path = ""

        def _on_step(self) -> bool:
            # 此回调的逻辑在 EvalCallback 之后运行
            # EvalCallback 有一个 'parent' 属性指向它自己
            assert self.parent is not None, "ReduceLROnPlateauCallback 必须作为 EvalCallback 的 callback 参数使用"

            # 检查模型是否有提升 (EvalCallback 会在找到更好的模型时更新路径)
            if self.parent.best_model_save_path != self.last_best_path:
                # 模型有提升，重置耐心计数器
                self.patience_counter = 0
                self.last_best_path = self.parent.best_model_save_path
                if self.verbose > 0:
                    print("检测到模型性能提升，重置学习率衰减耐心。")
            else:
                # 模型没有提升，增加耐心计数器
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                current_lr = self.model.policy.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * self.factor, self.min_lr)
                
                if current_lr > new_lr:
                    self.model.policy.optimizer.param_groups[0]['lr'] = new_lr
                    if self.verbose > 0:
                        print(f"评估奖励停滞 {self.patience} 次，学习率从 {current_lr:.6f} 衰减至 {new_lr:.6f}")
                    # 重置计数器，避免连续衰减
                    self.patience_counter = 0
            
            return True

    reduce_lr_callback = ReduceLROnPlateauCallback(patience=3, factor=0.5, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=max(10000 // args.num_envs, 1),  # 每10000步评估一次
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        # verbose=1,

        # 将学习率衰减回调作为 EvalCallback 的一部分
        callback_on_new_best=None, # 我们用自己的逻辑
        callback_after_eval=reduce_lr_callback # 在每次评估后运行我们的回调
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args.num_envs, 1),  # 每50000步保存一次
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix=args.exp_name
    )

    swanlab_callback = SwanLabCallback(
        project=args.exp_name,
        # experiment_name="MlpPolicy",
        verbose=2,
    )

    # 训练
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback, swanlab_callback],
        tb_log_name="tb_log",
        progress_bar=True
    )

    # 保存最终模型
    model.save(os.path.join(run_dir, args.exp_name))
    env.save(os.path.join(run_dir, "vec_normalize.pkl"))  # 保存环境标准化参数

    # 关闭环境
    env.close()
    eval_env.close()

def eval(args):
    run_dir = args.eval_dir
    print("\nLoading model and stats for evaluation...")
    # 重新创建基础环境 (不需要VecMonitor或VecNormalize包装)
    loaded_env_raw = make_vec_env(
        args.env_id,
        n_envs=args.num_eval_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork" if os.name != 'nt' else "spawn"},
        env_kwargs={"render_mode": "rgb_array"}
    )

    # 加载 VecNormalize 统计数据并应用到新环境
    stats_path = os.path.join(run_dir, "vec_normalize.pkl")
    if not os.path.exists(stats_path):
        print(f"[REEOR] no file in {stats_path}!!!")
        return 
    
    loaded_env_normalized = VecNormalize.load(stats_path, loaded_env_raw)

    # 设置为非训练模式
    loaded_env_normalized.training = False
    # 通常在评估时，我们想看到原始奖励，所以设置 norm_reward = False
    # 但如果你的模型期望归一化的奖励，则可以保持True。
    loaded_env_normalized.norm_reward = False # 或者 True，取决于你的需求
    print(f"Loaded normalized env: {type(loaded_env_normalized)}")

    # 加载模型
    model = PPO.load(os.path.join(run_dir, "best_model/best_model"), env=loaded_env_normalized)
    print("Model loaded successfully.")

    mean_reward, std_reward = evaluate_policy(model, loaded_env_normalized, n_eval_episodes=args.n_eval_episodes, deterministic=True)
    print(f"Loaded model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    loaded_env_normalized.close() # 关闭环境

    print("Script finished.")
    
def main(args):
    if args.is_train:
        train(args)
    else:
        eval(args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="Franka_PPO_DP", type=str, help="exp name")
    parser.add_argument("--num_envs", default=64, type=int, help="number of gym envs")
    parser.add_argument("--num_eval_envs", default=1, type=int, help="number of gym eval envs")
    parser.add_argument("--env_id", default="FrankaPickAndPlaceSparse-v0", type=str, help="env id")

    parser.add_argument("--run_dir", default="runs", type=str, help="this epoch runs dir name for save logs")
    # parser.add_argument("--name_prefix", default="ppo_franka", type=str, help="save ckpt prefix")

    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--device_id", default=0, type=int, help="gpu id")
    parser.add_argument("--batch_size", default=2048, type=int, help="batch size")
    parser.add_argument("--n_epochs", default=10, type=int, help="num_epochs")
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument("--total_timesteps", default=1e8, type=int, help="total_timesteps")

    parser.add_argument("--n_eval_episodes", default=5, type=int, help="n_eval_episodes")
    parser.add_argument("--eval_dir", default="runs/Franka_PPO_DP-2025-06-05-16-05-17", type=str, help="eval model path")

    parser.add_argument("--is_train", default=True, type=bool, help="train or eval?")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    '''
    nohup python train_dp.py > nuhup.out 2>&1 & 
    '''
    args = parse_args()
    main(args)
