# CSE498: object detection in my apartment

This project compares three ways to detect objects in two apartment photographs: pretrained YOLO11, bottle-specific fine-tuning, and text-prompt detection with YOLOE. The original photographs, the 50-image training dataset, the trained bottle checkpoint, and recorded results are included.

Read the [six-page report](report/report.pdf), download the [Overleaf ZIP with all figures](report/overleaf.zip), edit its [LaTeX source](report/main.tex), or follow the [short presentation guide](DEMO_GUIDE.md). The scripts below create new outputs under `runs/`; the recorded evidence under `results/` stays unchanged.

The portable scripts were rerun before submission: all four saved prediction tables and the bottle validation metrics matched the original records exactly. A separate one-epoch GPU training check also completed. See [verification.json](verification.json) for the checks; the report uses the original completed training run, not the one-epoch execution check.

## Recorded results

| Experiment | Main result |
|---|---|
| 1: pretrained YOLO11 | The original YOLO11n settings produced 2 desk / 8 kitchen boxes. YOLO11m with larger inputs produced 7 / 24. |
| 2: fine-tune bottles | Bottle validation AP50 changed from 0.7630 to 0.9290, and AP50–95 from 0.5586 to 0.6052. Apartment detections also gained false positives. |
| 3: text prompts | YOLOE returned 24 desk / 36 kitchen boxes for the fixed 33 prompts, including useful new categories and several incorrect labels. |

Box counts are not accuracy. The 10 validation images also selected the best fine-tuned checkpoint, so they are not an independent test set. The apartment photographs have no complete human bounding-box ground truth. Fine-tuning uses ordinary training with the first 10 layers frozen; it is **not LoRA**. The fine-tuned model predicts only `bottle`, not all 80 COCO classes.

## Install and download the pretrained models

Run these commands from the repository root. Python 3.12 and the CPU packages below match the inference environment used for the recorded experiments.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python setup_models.py
python src/prepare_dataset.py
```

`setup_models.py` downloads three official Ultralytics checkpoints, about 107 MB total, and checks their sizes and SHA-256 hashes against `models/sources.json`. To download only one, use `python setup_models.py --models yolo11m.pt`. `models/best_bottles.pt` is already included and has SHA-256 `509348c6c7d4c93542b7812ef712a384ded66b15bdb5afe33b270db079c33c79`.

The main library is pinned to Ultralytics 8.4.142. The recorded CPU run used NumPy 2.5.2 and OpenCV 5.0.0.93. Installation ranges in `requirements.txt` allow a separate GPU environment; package and hardware differences can cause small numeric differences. The requirements file does not replace an already compatible PyTorch installation.

## Run the three experiments

```bash
# Original settings: YOLO11n, image size 640, confidence 0.5.
python src/experiment1.py --baseline

# Improved settings: YOLO11m, image size 1920, confidence 0.25, IoU 0.45.
python src/experiment1.py

# Evaluate the recorded fine-tuned checkpoint and compare both apartment photos.
python src/compare_bottles.py

# YOLOE with the exact 33 cached prompt vectors from the recorded experiment.
python src/experiment3.py
```

All three demonstrations default to CPU and accept `--device` and `--output`. New files appear in `runs/experiment1/baseline/`, `runs/experiment1/improved/`, `runs/experiment2/`, and `runs/experiment3/`. Annotated images and CSV detections are saved there. The comparison script computes bottle-only validation AP by mapping the original COCO bottle class 39 and the new model class 0 to the same evaluation class before NMS.

YOLOE loads `data/prompts/prompt_embeddings.npz` with the official `load_prompt_embeddings()` API. These are the exact recorded text vectors, not newly chosen prompts or new predictions. This avoids downloading the roughly 600 MB MobileCLIP encoder for the demonstration. To change the vocabulary, the new prompts must be encoded again with the matching text encoder; editing `prompts.json` alone is insufficient and is rejected. The scene-informed vocabulary was chosen before viewing YOLOE predictions, so this is a demonstration rather than a blind benchmark.

## Repeat training, if needed

Training is unnecessary for presenting the saved results. To repeat it on a compatible CUDA GPU, create a separate environment. The recorded training used Python 3.10, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, and an RTX 5090.

```bash
python3.10 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python src/train_bottles.py --device 0
```

The defaults match the original run: at most 30 epochs, patience 10, image size 960, batch size 4, `freeze=10`, AdamW with learning rate 0.0005, seed 42, workers 0, and AMP disabled. New training goes into `runs/training/`; another run receives an incremented directory. It never replaces `models/best_bottles.pt`. To evaluate a new checkpoint, use `python src/compare_bottles.py --weights runs/training/weights/best.pt --output runs/new_model_comparison`. A [Colab reproduction template](notebooks/fine_tuning_colab.ipynb) is also provided; it has not been executed in Colab.

## Files and sources

- `data/apartment/`: the two user-provided photographs, unchanged from the supplied JPGs. These were existing photographs, not newly captured webcam frames.
- `data/bottles/`: 40 training / 10 validation images and 81 bottle boxes. The fixed random subset contains many drink and cosmetic bottles and is not a specialized thermos dataset. See its [README](data/bottles/README.md) for selection and attribution.
- `original/main.py`: the upstream webcam example from [valeriouberti/webcam-object-recognition](https://github.com/valeriouberti/webcam-object-recognition), commit `e1279fa04c0c263be4c7c4117f0e1c6e6c4b4365`. The original file is retained for attribution and comparison, not used as a portable experiment entry point.
- `results/`: actual recorded outputs and a [source manifest](results/source_manifest.json). Descriptions and copied text logs replace machine-specific home/workspace prefixes with placeholders. The manifest records both original and submitted file hashes and every translation or redaction; numeric predictions and images are preserved.
- `results/executed_scripts/`: original experiment source snapshots corresponding to historical hashes. Use `src/` to rerun the portable version.

The immutable trained checkpoint retains its historical training metadata; the portable entry points explicitly supply current dataset and output paths. No pretrained base checkpoints, MobileCLIP weight, Python environments, or training caches are committed.

Course tutorials: [YOLO11 fine-tuning](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolo11-object-detection-on-custom-dataset.ipynb), [YOLOE text-prompt detection](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-and-segmentation-with-yoloe.ipynb). The YOLOE run used the current supported `yoloe-11m-seg.pt` checkpoint instead of the tutorial's older `yoloe-v8l-seg.pt`; this difference is recorded in the report and provenance. Upstream models and code remain subject to their respective upstream terms.
