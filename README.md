# CSE498 Project

YOLO11 detection, bottle fine-tuning, and YOLOE on two apartment photos.

## Setup

Use Python 3.12 and run these commands from the project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python setup_models.py
```

The last command downloads the pretrained models from Ultralytics. The fine-tuned model is included as `models/best_bottles.pt`.

## 1. Detect objects

```bash
python src/experiment1.py --baseline  # YOLO11n from the original example
python src/experiment1.py             # YOLO11m with a larger input size
```

For the original webcam example, run `python original/main.py` and press **q** to quit. The apartment experiments use saved JPGs in `data/apartment/`.

## 2. Fine-tune and compare

The dataset has 40 training images and 10 validation images. To train again:

```bash
python src/train_bottles.py
```

CPU training is slow; use `--device 0` with a CUDA-enabled PyTorch installation for GPU training. New weights go to `runs/training/`.

To compare the included model with pretrained YOLO11m:

```bash
python src/compare_bottles.py
```

The comparison uses the same photos and validation set for both models. It handles the different bottle class IDs (39 before training, 0 after). Fine-tuning improved validation AP but also caused more false positives in the apartment. The validation set was used to select the best model, so it is not a separate test set.

## 3. Detect objects with YOLOE

```bash
python src/experiment3.py
```

This uses `yoloe-11m-seg.pt` and 33 saved text prompts from `data/prompts/`. Only detection boxes are shown. The prompt vectors are included; changing the words requires generating new vectors.

All results are saved in `runs/experiment1/`, `runs/experiment2/`, and `runs/experiment3/`. The report is submitted separately on CourseSite.

## References

- Original code: [webcam-object-recognition](https://github.com/valeriouberti/webcam-object-recognition)
- Tutorials: [YOLO11 fine-tuning](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolo11-object-detection-on-custom-dataset.ipynb) and [YOLOE](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-and-segmentation-with-yoloe.ipynb)
- Models: [Ultralytics](https://github.com/ultralytics/assets/releases/tag/v8.4.0), under its applicable AGPL-3.0 or commercial license
- Dataset: [Open Images bottle subset](data/bottles/README.md)

YOLO11m is an extra comparison; the original example uses YOLO11n. For YOLOE, this project uses the 11m model instead of the tutorial's v8l model. Fine-tuning uses ordinary training with some layers frozen, not LoRA.
