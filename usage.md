# Usage

This project adapts the RegiStain virtual staining workflow for paired image training.
The current default task is:

```text
HE image patch -> IHC image patch
```

## 1. Clone The Code

```bash
cd /root/siton-tmp/lzj
git clone https://github.com/blue16681/Virtual-histological-staining-of-unlabeled-autopsy-tissue.git
cd Virtual-histological-staining-of-unlabeled-autopsy-tissue
```

If `git clone` is slow or blocked, download the zip archive instead:

```bash
cd /root/siton-tmp/lzj
wget -O project.zip https://github.com/blue16681/Virtual-histological-staining-of-unlabeled-autopsy-tissue/archive/refs/heads/main.zip
unzip project.zip
mv Virtual-histological-staining-of-unlabeled-autopsy-tissue-main Virtual-histological-staining-of-unlabeled-autopsy-tissue
cd Virtual-histological-staining-of-unlabeled-autopsy-tissue
```

## 2. Configure Environment

Create and activate the Conda environment:

```bash
conda env create -n autopsy-vs -f tf2_env.yaml
conda activate autopsy-vs
```

If pip times out while installing large packages, activate the partially created environment and install the missing packages manually:

```bash
conda activate autopsy-vs

pip install --default-timeout=1000 --retries 10 \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  keras-nightly==2.5.0.dev2021032900

pip install --default-timeout=1000 --retries 10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  tensorflow-gpu==2.5.0 tensorflow-estimator==2.5.0 \
  keras-preprocessing==1.1.2 \
  tensorflow-addons==0.13.0 typeguard==2.13.3 typing-extensions==3.7.4.3 \
  h5py==3.1.0 protobuf==3.20.3 tensorboard==2.11.0
```

Verify TensorFlow and GPU:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow_addons as tfa; print(tfa.__version__)"
```

Expected TensorFlow version:

```text
2.5.0
```

## 3. Dataset Structure

The training script expects this processed dataset structure:

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

For the current task:

```text
input  = HE RGB patch
target = IHC RGB patch
```

Input and target files must have the same filename inside each split.

The GitHub repository does not include `dataset/`, so upload the processed data separately to:

```text
Virtual-histological-staining-of-unlabeled-autopsy-tissue/dataset/DermaRepo_processed_256
```

## 4. Preprocess Raw Paired Images

If you start from raw whole images, place them like this:

```text
dataset/DermaRepo/
  HE/
  IHC/
```

Then generate registered 256 x 256 paired patches:

```bash
python preprocess_image_pairs.py \
  --input-dir dataset/DermaRepo/HE \
  --target-dir dataset/DermaRepo/IHC \
  --output-dir dataset/DermaRepo_processed_256 \
  --patch-size 256 \
  --test-ratio 0.2 \
  --white-threshold 245 \
  --min-tissue 0.05
```

This will:

```text
pair HE/IHC whole images
rigidly register HE to IHC
tile into 256 x 256 patches
remove mostly white patches
split into train/test
save paired PNG files
```

## 5. Train

If the dataset is located at the default path, run:

```bash
conda activate autopsy-vs
python train_stage2_seperate_train_by_iters.py
```

Run with explicit server paths:

```bash
python train_stage2_seperate_train_by_iters.py \
  --data-root /root/siton-tmp/lzj/Virtual-histological-staining-of-unlabeled-autopsy-tissue/dataset/DermaRepo_processed_256 \
  --model-path /root/siton-tmp/lzj/runs/dermarepo_he_to_ihc \
  --gpu 0 \
  --batch-size 4 \
  --n-epoch 150 \
  --initial-alternate-steps 6000 \
  --valid-steps 100
```

For a quick pipeline check:

```bash
python train_stage2_seperate_train_by_iters.py --smoke-test
```

## 6. Important Training Arguments

```text
--data-root
```

Dataset root containing `train/input`, `train/target`, `test/input`, and `test/target`.

```text
--model-path
```

Directory for checkpoints, logs, and preview images.

```text
--gpu
```

Sets `CUDA_VISIBLE_DEVICES`. Use `--gpu 0` for the first GPU.

```text
--batch-size
```

Training batch size. Reduce it if GPU memory is insufficient.

```text
--n-channels
```

Base channel count for G/D. Default is `32`. Use `16` if memory is tight.

```text
--initial-alternate-steps
```

Number of G/D steps and R steps in each alternating training round.

```text
--valid-steps
```

Validation interval.

```text
--n-epoch
```

Number of alternating training rounds.

```text
--prev-checkpoint-path
```

Resume from a previous run directory containing latest model weights.

## 7. Outputs

Training outputs are saved under `--model-path`, for example:

```text
runs/dermarepo_he_to_ihc/
  output/
  train_log.txt
  model_G_latest.h5
  model_D_latest.h5
  model_R_latest.h5
```

The generator checkpoint is:

```text
model_G_latest.h5
```

## 8. Notes

- The training code defaults to RGB input and RGB target.
- The default direction is HE to IHC.
- The dataset and generated model weights are ignored by git.
- If training crashes due to GPU memory, first reduce `--batch-size`, then reduce `--n-channels`.
