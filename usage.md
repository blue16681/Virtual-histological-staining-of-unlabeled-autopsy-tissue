# 使用说明

本项目基于 RegiStain 虚拟染色框架，当前适配的训练任务是：

```text
HE 图像 patch -> IHC 图像 patch
```

也就是说：

```text
input  = HE RGB 图像
target = IHC RGB 图像
```

## 1. 拉取代码

在服务器上进入你想放项目的目录：

```bash
cd /root/siton-tmp/lzj
git clone https://github.com/blue16681/Virtual-histological-staining-of-unlabeled-autopsy-tissue.git
cd Virtual-histological-staining-of-unlabeled-autopsy-tissue
```

如果服务器上 `git clone` 很慢或失败，可以下载 zip：

```bash
cd /root/siton-tmp/lzj
wget -O project.zip https://github.com/blue16681/Virtual-histological-staining-of-unlabeled-autopsy-tissue/archive/refs/heads/main.zip
unzip project.zip
mv Virtual-histological-staining-of-unlabeled-autopsy-tissue-main Virtual-histological-staining-of-unlabeled-autopsy-tissue
cd Virtual-histological-staining-of-unlabeled-autopsy-tissue
```

如果服务器上已经有旧代码，进入项目目录后更新：

```bash
git pull origin main
```

## 2. 配置 Conda 环境

创建环境：

```bash
conda env create -n autopsy-vs -f tf2_env.yaml
conda activate autopsy-vs
```

如果创建环境时 pip 下载超时，不需要删除环境重来。先进入已经部分创建好的环境：

```bash
conda activate autopsy-vs
```

然后单独安装缺失包。`keras-nightly==2.5.0.dev2021032900` 很老，国内镜像可能没有，建议从官方 PyPI 装：

```bash
pip install --default-timeout=1000 --retries 10 \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  keras-nightly==2.5.0.dev2021032900
```

其余包可以用清华源：

```bash
pip install --default-timeout=1000 --retries 10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  tensorflow-gpu==2.5.0 tensorflow-estimator==2.5.0 \
  keras-preprocessing==1.1.2 \
  tensorflow-addons==0.13.0 typeguard==2.13.3 typing-extensions==3.7.4.3 \
  h5py==3.1.0 protobuf==3.20.3 tensorboard==2.11.0
```

验证 TensorFlow 和 GPU：

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow_addons as tfa; print(tfa.__version__)"
```

期望 TensorFlow 版本是：

```text
2.5.0
```

如果 GPU 正常，会看到类似：

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 3. 数据集目录结构

训练脚本默认读取处理后的 256 patch 数据集：

```text
dataset/DermaRepo_processed_256/
  train/
    input/
      train_000001.png
      train_000002.png
    target/
      train_000001.png
      train_000002.png
  test/
    input/
      test_000001.png
      test_000002.png
    target/
      test_000001.png
      test_000002.png
```

当前任务中：

```text
train/input  = HE patch
train/target = IHC patch
test/input   = HE patch
test/target  = IHC patch
```

同一对 input 和 target 必须同名，例如：

```text
train/input/train_000001.png
train/target/train_000001.png
```

GitHub 仓库不会包含 `dataset/`，所以需要你把数据集单独上传到服务器项目目录下：

```text
Virtual-histological-staining-of-unlabeled-autopsy-tissue/dataset/DermaRepo_processed_256
```

## 4. 从原始整图生成训练数据

如果你手里是原始 HE/IHC 整图，先放成：

```text
dataset/DermaRepo/
  HE/
  IHC/
```

然后运行预处理脚本：

```bash
python preprocess_image_pairs.py \
  --input-dir dataset/DermaRepo/HE \
  --target-dir dataset/DermaRepo/IHC \
  --output-dir dataset/DermaRepo_processed_256 \
  --patch-size 256 \
  --test-ratio 0.2 \
  --white-threshold 245 \
  --min-tissue 0.05 \
  --local-registration translation \
  --local-max-shift 32
```

这个脚本会完成：

```text
匹配 HE/IHC 整图
将 HE 刚性配准到 IHC
在每个 256 patch 上做局部平移微调
切成 256 x 256 patch
去除大面积白背景 patch
划分 train/test
保存成成对 PNG 文件
```

如果局部仍有明显旋转偏差，可以尝试：

```bash
--local-registration rigid
```

如果局部配准反而引入错误，可以关闭 patch 级微调：

```bash
--local-registration none
```

## 5. 开始训练

如果数据集就在默认路径：

```text
dataset/DermaRepo_processed_256
```

直接运行：

```bash
conda activate autopsy-vs
python train_stage2_seperate_train_by_iters.py
```

如果想显式指定服务器路径：

```bash
python train_stage2_seperate_train_by_iters.py \
  --data-root dataset/DermaRepo_processed_256 \
  --model-path runs/DermaRepo_processed_256 \
  --gpu 0 \
  --batch-size 4 \
  --n-epoch 150 \
  --initial-alternate-steps 6000 \
  --valid-steps 100
```

第一次跑建议先做一个快速测试：

```bash
python train_stage2_seperate_train_by_iters.py --smoke-test
```

`--smoke-test` 会把训练轮数和验证数量临时调小，用来检查路径、数据读取、模型构建和 GPU 是否正常。

## 6. 常用训练参数

```text
--data-root
```

数据集根目录，里面需要包含 `train/input`、`train/target`、`test/input`、`test/target`。

```text
--model-path
```

模型权重、日志、训练中间图像保存目录。

```text
--gpu
```

设置 `CUDA_VISIBLE_DEVICES`。例如：

```bash
--gpu 0
```

表示使用第 0 张 GPU。

```text
--batch-size
```

训练 batch size。显存不够时优先减小它，例如改成：

```bash
--batch-size 2
```

```text
--valid-batch-size
```

验证 batch size。不指定时默认等于 `--batch-size`。

```text
--image-size
```

输入 patch 尺寸。当前数据是 256，所以默认是：

```bash
--image-size 256
```

```text
--input-channels
--label-channels
```

输入和输出通道数。当前 HE/IHC 都是 RGB，因此默认都是 3。

```text
--n-channels
```

G/D 网络的基础通道数。默认是 32。显存不足时可以改成：

```bash
--n-channels 16
```

```text
--lambda-adv
```

GAN 对抗损失权重。默认是 50。

```text
--initial-alternate-steps
```

每一轮交替训练中，G/D 和 R 各训练多少 step。默认是 6000。

```text
--valid-steps
```

每隔多少 step 做一次验证。默认是 100。

```text
--n-epoch
```

交替训练总轮数。默认是 150。

```text
--train-q-limit
--valid-q-limit
```

训练/验证循环中每次取多少个 batch。验证太慢时可以调小 `--valid-q-limit`。

```text
--prev-checkpoint-path
```

从已有训练目录恢复训练。该目录下应包含：

```text
model_G_latest.h5
model_D_latest.h5
model_R_latest.h5
```
### 大致流程

epoch 0:
  训练 G/D 6000 step
    每 100 step 验证一次
    每 100 step 保存 latest 权重
    如果验证变好，额外保存 best 权重

  训练 R 6000 step
    中间不按 valid_steps 验证

  这一轮结束后，再验证一次
  保存 latest 权重


epoch 0:
  G/D 训练 1000 step
    step 200: 验证 + 保存 latest
    step 400: 验证 + 保存 latest
    step 600: 验证 + 保存 latest
    step 800: 验证 + 保存 latest
  R 训练 1000 step
  epoch 结束: 验证 + 保存 latest

epoch 1:
  G/D 训练 900 step
    step 200: 验证 + 保存 latest
    step 400: 验证 + 保存 latest
    step 600: 验证 + 保存 latest
    step 800: 验证 + 保存 latest
  R 训练 900 step
  epoch 结束: 验证 + 保存 latest

epoch 2:
  G/D 训练 810 step
    step 200: 验证 + 保存 latest
    step 400: 验证 + 保存 latest
    step 600: 验证 + 保存 latest
    step 800: 验证 + 保存 latest
  R 训练 810 step
  epoch 结束: 验证 + 保存 latest


## 7. 输出文件

训练结果会保存在 `--model-path` 下，例如：

```text
runs/dermarepo_he_to_ihc/
  output/
  train_log.txt
  model_G_latest.h5
  model_D_latest.h5
  model_R_latest.h5
```

其中最重要的是生成器：

```text
model_G_latest.h5
```

后续推理主要使用 G 模型。

## 8. 常见问题

### pip 下载超时

加长 timeout 和 retries：

```bash
pip install --default-timeout=1000 --retries 10 package_name
```

### 找不到 keras-nightly 版本

不要用国内源安装这个包，改用官方 PyPI：

```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org keras-nightly==2.5.0.dev2021032900
```

### 显存不够

优先减小：

```bash
--batch-size
```

如果还不够，再减小：

```bash
--n-channels
```

### 数据路径找不到

确认目录结构是：

```text
dataset/DermaRepo_processed_256/train/input
dataset/DermaRepo_processed_256/train/target
dataset/DermaRepo_processed_256/test/input
dataset/DermaRepo_processed_256/test/target
```

并且 input 和 target 文件名一一对应。

## 9. 推荐的首次运行命令

先做快速测试：

```bash
python train_stage2_seperate_train_by_iters.py \
  --data-root dataset/DermaRepo_processed_256 \
  --model-path runs/debug_smoke \
  --gpu 0 \
  --smoke-test
```

确认没问题后正式训练：

```bash
python train_stage2_seperate_train_by_iters.py \
  --data-root dataset/DermaRepo_processed_256 \
  --model-path runs/dermarepo_he_to_ihc \
  --gpu 0 \
  --batch-size 4 \
  --n-epoch 150 \
  --initial-alternate-steps 6000 \
  --valid-steps 100
```
