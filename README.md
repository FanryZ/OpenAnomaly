### Installation

- Prepare experimental environments

  ```shell
  pip install -r requirements.txt
  ```

- Install the environment for Segment-Anything

  ```shell
  git clone git@github.com:facebookresearch/segment-anything.git
  cd segment-anything; pip install -e .
  ```

- Download pre-trained model weights and the configuration files from [Segment-Anything](https://github.com/facebookresearch/segment-anything). Save them under `checkpoints`.

## Dataset Preparation 
### MVTec AD
- Download and extract [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) into `data/mvtec`
- run`python data/mvtec.py` to obtain `data/mvtec/meta.json`
```
data
├── mvtec
    ├── meta.json
    ├── bottle
        ├── train
            ├── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
        ├── ground_truth
            ├── anomaly1
                ├── 000.png
```

### VisA
- Download and extract [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) into `data/visa`
- run`python data/visa.py` to obtain `data/visa/meta.json`
```
data
├── visa
    ├── meta.json
    ├── candle
        ├── Data
            ├── Images
                ├── Anomaly
                    ├── 000.JPG
                ├── Normal
                    ├── 0000.JPG
            ├── Masks
                ├── Anomaly
                    ├── 000.png
```

## Model Training

  Run the following command to train the projection layer

  ```shell
  bash scripts/train.sh
  ```

  The pre-trained weights are saved under `./exps/mvtec/vit_large_14_518` and `./exps/mvtec/vit_large_14_518`.

## Testing

  Test our method for anomaly classification

  ```shell
  bash scripts/test.sh
  ```

  Test our method for anomaly segmentation

  ```shell
  bash scripts/test_pixel_visa.sh
  bash scripts/test_pixel_mvtec.sh
  ```

  Caution that replace the "path-to-pretrained-porjection-layer" by the path of pre-trained projection layer. The weights pre-trained on MVTec-AD is utilized to test images in VisA dataset, and vise versa.

## Acknowledgements
We thank  [VAND-APRIL-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN), [segment-anything](https://github.com/facebookresearch/segment-anything), [open_clip](https://github.com/mlfoundations/open_clip), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for providing assistance for our research.

We will publish our revised code for better readability.
