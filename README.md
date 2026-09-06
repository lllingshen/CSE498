# CSE498: object detection

Three experiments on two apartment photos: pretrained YOLO11, bottle fine-tuning, and text-prompt detection with YOLOE.

## Setup

Use Python 3.12. Run all commands from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python setup_models.py
```

The setup script downloads the three official pretrained models. The trained bottle model is already in `models/best_bottles.pt`.

## Run the experiments

```bash
# 1. Original YOLO11n settings, then the larger YOLO11m comparison.
python src/experiment1.py --baseline
python src/experiment1.py

# 2. Compare YOLO11m before and after bottle fine-tuning.
python src/compare_bottles.py

# 3. YOLOE with the saved 33 text prompts.
python src/experiment3.py
```

Images, detection tables, and metrics are saved under `runs/experiment1/`, `runs/experiment2/`, and `runs/experiment3/`. All three demonstrations default to CPU.

To repeat fine-tuning on the included 40 training / 10 validation images:

```bash
python src/train_bottles.py
```

CPU training is slow. For GPU training, use a CUDA-enabled PyTorch installation and add `--device 0`. The recorded run used PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, and an RTX 5090. New checkpoints go to `runs/training/`; the included checkpoint is not overwritten.

## Code and data

- `original/main.py`: unchanged webcam example. Run `python original/main.py` from this folder; press **q** to quit. The saved experiments used existing JPGs, not webcam captures.
- `src/experiment1.py`: read the two photos, detect objects, and save boxes.
- `src/train_bottles.py`: fine-tune YOLO11m on the bottle dataset.
- `src/compare_bottles.py`: compare bottle boxes and validation AP. It maps COCO bottle class 39 to dataset class 0 for a fair comparison.
- `src/experiment3.py`: run YOLOE using the text embeddings in `data/prompts/`. Changing the words requires regenerating the embeddings.
- `src/common.py`: shared paths and dataset configuration.
- `data/apartment/`: the two test photos. `data/bottles/`: 50 labeled images and [source attribution](data/bottles/README.md).

Fine-tuning is ordinary partial fine-tuning, **not LoRA**. The small validation score improved, but the apartment outputs gained false positives. The validation split also selected the best checkpoint; it is not an independent test set. Detection counts are not accuracy.

## Sources

Based on [valeriouberti/webcam-object-recognition](https://github.com/valeriouberti/webcam-object-recognition), commit `e1279fa`. Course tutorials: [YOLO11 fine-tuning](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolo11-object-detection-on-custom-dataset.ipynb) and [YOLOE](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-and-segmentation-with-yoloe.ipynb).

YOLO11m is an added comparison. Experiment 3 uses `yoloe-11m-seg.pt` instead of the tutorial's `yoloe-v8l-seg.pt`; only boxes are displayed. [Model download sources](models/sources.json) are from Ultralytics. Its models and code use AGPL-3.0 or the applicable Ultralytics license. The report is submitted separately on CourseSite.
