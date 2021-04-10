Baseline code copied from https://www.kaggle.com/pasewark/pytorch-resnet-lstm-with-attention.


## How to Use

1. Download the dataset into the `input` folder.

```
input/
    - train/
        - 0/
        - ...
    - test/
        - 0/
        - ...
```

2. Set up the environment.

```bash
conda env create -f environment.yml
conda activate venv
```

3. Change the training configs in `config.yaml`.

4. Start training.

```
python train.py
```