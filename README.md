## Dependencies

```bash
conda create -n ai1003 python=3.8 -y
conda activate ai1003
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop
```

## Datasets

Download training and testing datasets and put them into the corresponding folders of `datasets/`. See [datasets](datasets/README.md) for the detail of the directory structure.

## Training

- Run the following scripts. The training configuration is in `options/train/`.

  ```bash
  python basicsr/train.py -opt options/train/train_simpleCNN.yml
  ```

- The training experiment is in `experiments/`.

## Testing

- Run the following scripts. The testing configuration is in `options/test/`.

  Note 1:  You can set `use_chop: True` (default: False) in YML to chop the image for testing.

  ```bash
  python basicsr/test.py -opt options/Test/test_simpleCNN.yml
  ```

- The output is in `results/`.

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR).
