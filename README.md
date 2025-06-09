# RL-FRANKA-PANDA 

本文档主要是说明该仓库使用信息

## 安装

正常按照 `requirements.txt` 安装环境即可，没啥特殊要求

需要安装日志记录库

```
pip install swanlab
```

## 运行

1. 首先注册自己 swanlab 账户

2. 在虚拟环境中，登陆账户, 根据提示输入 API 就可以

  ```
  swanlab login
  ```

3. 修改 `parse_args()` 参数，按照需求更改，一定要更改 `device_id` 

4. 直接运行就可以，可以在 swanlab 看到相应训练数据

## 代码说明

```txt
- panda_mujoco_gym    # env
- scripts             # train 或者 test 后面表示使用算法
  - test_0            # ⚠️最简单测试方式
  - train_tqc         # tqc 训练环境
  - test_tqc          # RGB 环境下渲染查看推理结果
  - train_0           # ⚠️最简单训练方式
  - train_1           # ⚠️增加多环境并行训练方式
  - train_2           # ⚠️增加多网络 多环境
- test                # 测试代码
```

## faq

1. ⚠️ 表示训练脚本可能存在问题，须谨慎使用

2. 想要运行 scripts 文件夹下脚本时如果报错, 是因为暂时找不到这个包

```bash
ModuleNotFoundError: No module named 'panda_mujoco_gym'
```

添加以下路径就可以

```py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## ref

本仓库借鉴来自 https://github.com/lsp-yh/RL-Robot-Manipulation
