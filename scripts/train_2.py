import os
from datetime import datetime
from stable_baselines3 import PPO
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

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,  # 激活函数
        net_arch=dict(
            pi=[256, 256],  # Actor (policy) network
            vf=[256, 256]   # Critic (value) network
        )
    )

    model = PPO("MultiInputPolicy", 
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
    class MutableLRSchedule:
        """
        一个可变的学习率调度器，其值可以被外部（如回调函数）修改。
        它本身是一个可调用对象，可以作为 learning_rate 参数传入模型。
        """
        def __init__(self, initial_value: float):
            self.current_value = initial_value

        def __call__(self, progress_remaining: float) -> float:
            # PPO 会调用这个方法，我们只需返回当前的值
            # 注意：这里的 progress_remaining 我们没有使用，因为值的更新由回调控制
            return self.current_value

        def set_value(self, new_value: float):
            """提供一个方法给回调函数来修改学习率"""
            self.current_value = new_value

    class ReduceLROnPlateauCallback(BaseCallback):
        """
        当评估奖励停滞时，降低学习率。
        这个版本与 MutableLRSchedule 配合使用。
        """
        def __init__(self, lr_scheduler: MutableLRSchedule, patience: int, factor: float = 0.5, min_lr: float = 1e-6, verbose: int = 0):
            super().__init__(verbose)
            self.lr_scheduler = lr_scheduler # 存储调度器实例
            self.patience = patience
            self.factor = factor
            self.min_lr = min_lr
            self.patience_counter = 0
            self.last_best_path = ""

        def _on_step(self) -> bool:
            assert self.parent is not None, "此回调必须与 EvalCallback 一起使用"

            if self.parent.best_model_save_path != self.last_best_path:
                self.patience_counter = 0
                self.last_best_path = self.parent.best_model_save_path
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                current_lr = self.lr_scheduler.current_value
                new_lr = max(current_lr * self.factor, self.min_lr)
                
                if current_lr > new_lr:
                    # 直接修改调度器的值！
                    self.lr_scheduler.set_value(new_lr)
                    if self.verbose > 0:
                        print(f"UP TO {self.patience}! LR form {current_lr:.7f} to {new_lr:.7f}")
                    self.patience_counter = 0
            
            return True
        
    lr_schedule = MutableLRSchedule(initial_value=args.lr)
    reduce_lr_callback = ReduceLROnPlateauCallback(lr_scheduler=lr_schedule, patience=5, factor=0.5, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=max(100000 // args.num_envs, 1),  # 每10000步评估一次
        n_eval_episodes=10,
        deterministic=True,
        render=False,

        callback_after_eval=reduce_lr_callback # 在每次评估后运行我们的回调
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args.num_envs, 1),  # 每50000步保存一次
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix=args.name_prefix
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
    model.save(os.path.join(run_dir, "ppo_franka_final"))
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
    parser.add_argument("--name_prefix", default="ppo_franka", type=str, help="save ckpt prefix")

    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--device_id", default=1, type=int, help="gpu id")
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
