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

- train_dp    多环境训练代码，同时兼顾 eval 功能
- infer       RGB 环境下渲染查看推理结果

## 其他

本仓库借鉴来自 https://github.com/JFan5/RL-Franka-Panda
