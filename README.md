# RL-FRANKA-PANDA 

本文档主要是说明该仓库使用信息

[安装](#安装)
[运行](#运行)
[faq](#faq)
[ref]()


## 安装

正常按照 `requirements.txt` 安装环境即可，没啥特殊要求

需要安装日志记录库

```
pip install -r requirements.txt
pip install swanlab
```

## 运行

### 开始训练

1. 首先注册自己 swanlab 账户

2. 在虚拟环境中，登陆账户, 根据提示输入 API 就可以

  ```
  swanlab login
  ```

3. 修改 `parse_args()` 参数，按照需求更改，一定要更改 `device_id` 

4. 直接运行就可以，可以在 swanlab 看到相应训练数据

### 进阶

#### 训练圆柱

1. 首先需要修改 `panda_mujoco_gym/assets/pick_and_place.xml#36` 这样可以加载一个圆柱体 

2. 正常训练即可

#### 柔性体？

训练柔性体有点复杂，需要更改环境，因为 `flexcomp` 这个属性在 `mujoco=3.0.0` 才引入，但是我的环境中强依赖 `mujoco==2.3.3`

1. 查看了一些依赖 mujoco 环境的是 gymnasium-robotics 这个包，这个包在 `1.2.2` 的时候确实强依赖 `mujoco==2.3.3` [在这里](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/317620e76b1f33e80105c5ca79220da5566e0674/pyproject.toml), 但是在 `1.2.3` 的时候就改为了 `>=2.3.3` [在这里](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/3a579d6efa2056cce394ffeb5d1ef83a58fea23a/pyproject.toml)

2. 其他环境尽量保持不变，安装以下包, `pettingzoo` 包的安装务必小心 

```txt
gymnasium-robotics==1.3.0
gymnasium==1.0.0
mujoco==3.1.6
pettingzoo==1.24.3
```

> mujoco 版本改成 `3.1.6`, 因为 `gymnasium-robotics 1.3.0 requires mujoco<3.2.0,>=2.2.0, but you have mujoco 3.2.6 which is incompatible.` [看这里](https://github.com/Farama-Foundation/Gymnasium-Robotics/releases/tag/v1.3.0)

3. 修改为柔性体, 改为柔性体可以正常渲染，但是时间一长会堆栈溢出

改为柔性体需要增加两部分代码

```xml
<!-- 添加内存上线 -->
<size memory="50M"/>

<!-- #39 添加这个柔性圆柱体 -->
<geom size=".001" contype="0" conaffinity="0" group="4"/>
<flexcomp name="obj_geom" type="cylinder" count="5 5 5" spacing=".01 .01 .01" mass="0.1" rgba="0.5 0.7 0.5 1" radius="0.001" dim="3" flatskin="false">
  <contact internal="true" selfcollide="auto"/>
  <edge equality="true"/>
</flexcomp>
```

**QA**

```bash
mujoco.FatalError: mj_stackAlloc: insufficient memory: max = 3131760, available = 495312, requested = 524320 (ne = 392, nf = 0, nefc = 1160, ncon = 201)
```

[解决方案看这里](https://github.com/google-deepmind/mujoco/issues/1328)


## 代码说明

```txt
- panda_mujoco_gym    # env
- scripts             # train 或者 test 后面表示使用算法
  - test_dp           # ⚠️ DP 方式测试
  - test_sac          # sac 方式测试
  - test_tqc          # tqc 方式测试
  - train_dp          # ⚠️ 多环境并行训练方式
  - train_sac         # sac 训练环境
  - train_tqc         # tqc 训练环境
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

简单方法：

```sh
pip install -e .
```

> 将当前仓库作为包安装会自动搜索子包。

## ref

本仓库借鉴来自 https://github.com/lsp-yh/RL-Robot-Manipulation
