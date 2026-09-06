# CSE498: five-minute demo guide

My main point: **inference settings, fine-tuning, and text prompts solve different problems. A higher validation score does not guarantee better detections in my apartment.**

This demo uses two saved apartment photos. It does not measure live webcam performance. Use the saved results for a smooth presentation; the CPU commands below can regenerate them without retraining.

## Set up before the presentation

Run these commands from the `CSE498-submission` folder on Linux. The [README](README.md) is the installation reference if its instructions change.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python setup_models.py
python src/prepare_dataset.py
```

Download the official weights before class. `models/best_bottles.pt` is the saved result of the completed fine-tuning run; it does not need to be trained again. Experiment 3 uses the supplied embeddings for the fixed 33 text prompts, so its normal demo does not need to regenerate them with the large text encoder.

Rehearse these commands once:

```bash
python src/experiment1.py --baseline --device cpu
python src/experiment1.py --device cpu
python src/compare_bottles.py --device cpu
python src/experiment3.py --device cpu
```

New outputs go under `runs/`. The submitted historical results are under `results/`. The scripts also accept `--output` if you want a separate rehearsal folder. Runtime depends on the computer; use the historical images if a live run takes too long.

GPU training is optional. The separate environment below follows the CUDA 12.8/PyTorch versions used for the recorded training run and needs a compatible NVIDIA GPU and driver. Keep the CPU demo environment available:

```bash
deactivate
python3.10 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
python src/train_bottles.py --device 0
```

Only start GPU training if the check prints `True`. Training creates a new run under `runs/`; it does not replace the submitted `models/best_bottles.pt`. The comparison command continues to use that submitted checkpoint. **Do not retrain during the five-minute demo.**

## Five-minute speaking order

| Time | What to show | What to say |
|---|---|---|
| 0:00–0:30 | The two apartment photos | “I tested ordinary object detection, bottle fine-tuning, and open-vocabulary detection on these scenes.” |
| 0:30–1:30 | Experiment 1: baseline and adjusted desk images | “The baseline uses YOLO11n. I then used YOLO11m with a larger input, a lower confidence cutoff, and different suppression settings. More small objects appear, but the class vocabulary is still fixed.” |
| 1:30–2:45 | Experiment 2: dataset overview, AP table, desk before/after | “I trained on 40 images and validated on 10. Validation AP improved, but apartment false positives increased. More boxes are not necessarily better.” |
| 2:45–4:00 | Experiment 3: desk and kitchen previews | “YOLOE lets me specify object names. It found the computer tower and electric kettle, but still missed or mislabeled other objects.” |
| 4:00–5:00 | One success and one failure from each method | “The next step is better target-scene data and an independent labeled test set. I would not replace the general detector with this bottle-only model.” |

The original photos are in `data/apartment/`. Open these result images before class:

| Purpose | Saved image |
|---|---|
| Experiment 1 baseline, desk | `results/experiment1/baseline/1P9A1048_annotated.jpg` |
| Experiment 1 adjusted settings, desk | `results/experiment1/improved/1P9A1048_annotated.jpg` |
| Experiment 2: desk before/after | `results/experiment2/comparison/1P9A1048_comparison.jpg` |
| Experiment 2: kitchen before/after | `results/experiment2/comparison/1P9A1049_comparison.jpg` |
| Experiment 3: desk | `results/experiment3/1P9A1048_preview.jpg` |
| Experiment 3: kitchen | `results/experiment3/1P9A1049_preview.jpg` |

Fresh Experiment 1 images go to `runs/experiment1/baseline/` and `runs/experiment1/improved/`. Fresh Experiment 2 images are `runs/experiment2/before_1P9A1048.jpg`, `after_1P9A1048.jpg`, and the corresponding `1049` images; its `metrics.json` records the AP comparison. Fresh Experiment 3 previews go to `runs/experiment3/`.

## Results I need to remember

**Experiment 2:** the dataset contains 50 images and 81 bottle boxes: 40 training images/66 boxes, and 10 validation images/15 boxes. Training stopped after 17 epochs; epoch 7 supplied the best checkpoint.

| Bottle metric on the same validation split | Before | After |
|---|---:|---:|
| AP50 | 0.763 | 0.929 |
| AP50–95 | 0.559 | 0.605 |

The validation images also selected the best checkpoint. These are **validation results, not independent test results**. In the apartment, bottle predictions increased from **1 to 12 on the desk** and **15 to 26 in the kitchen**, with obvious new false positives on furniture and appliances. The previously missed white insulated bottle appeared, but that one success did not make the overall output reliable.

**Experiment 3:** the fixed 33-name vocabulary produced **24 desk boxes and 36 kitchen boxes**. Those numbers are output counts, not accuracy. Good examples include the computer tower, electric kettle, and faucet. Failure examples include the electronic piano labeled as a computer keyboard/desk, the humidifier labeled as an insulated bottle, the dishwasher labeled as an oven, and missed headphones. All returned boxes were retained.

## Short answers to likely questions

1. **What does the model output?**

   “For each detection, it returns a class, a confidence score, and a bounding box. Detection tells me both what the object might be and where it is. I save the boxes on the photo and in a CSV.”

2. **What are the backbone, neck, and head in YOLO11?**

   “The backbone extracts image features. The neck combines features at different resolutions. The head predicts boxes and class scores at multiple scales. YOLO11 uses C3k2 blocks, SPPF pooling, and C2PSA attention; its detection head is anchor-free.” [Architecture reference](https://docs.ultralytics.com/guides/yolo-architecture)

3. **Why change YOLO11n to YOLO11m and increase the input size?**

   “The medium model has more capacity, and a larger input preserves more detail for small objects. Both increase computation. I changed the input from 640 to 1920 and lowered confidence from 0.50 to 0.25. I changed several settings together, so I cannot isolate their individual effects. I did not train new weights in Experiment 1.”

4. **What do input size, confidence, and NMS mean?**

   “The photo is resized, padded, and converted to an RGB tensor with pixels scaled to 0–1. Input size does not change the saved output-photo resolution. With a 1920 target, these photos used a 1312-by-1920 tensor. Confidence removes low-scoring predictions. NMS removes strongly overlapping lower-scoring boxes using an IoU threshold. It can reduce duplicates, but cannot correct a wrong label.” [Prediction settings](https://docs.ultralytics.com/modes/predict/)

5. **How did you collect and split the bottle data?**

   “I sampled Open Images bottle photos using seed 42, after excluding images with grouped or depicted bottle annotations. I used 40 images for training and 10 for validation, with no shared image IDs. The source pool was Open Images validation, but this 40/10 split is my assignment split. The apartment photos were excluded.” Each YOLO label row is `class_id center_x center_y width height`, with coordinates normalized to the image dimensions; bottle is class `0`.

6. **What exactly did you fine-tune? Was it LoRA?**

   “I started from pretrained YOLO11m, froze the first ten indexed layers, and trained the remaining trainable parameters for bottle detection. I used AdamW, learning rate 0.0005, image size 960, and batch size four. The maximum was 30 epochs with patience ten. This was ordinary fine-tuning; freezing layers is not LoRA.”

7. **Why does the fine-tuned model stop showing chairs and monitors?**

   “Its output head was configured for one class: bottle. It is no longer an 80-class general detector. Non-bottle objects can still trigger false bottle predictions. This comparison measures bottle detection, not retention of all the original categories.”

8. **What do AP50 and AP50–95 mean, and was the comparison fair?**

   “AP summarizes the precision–recall curve. AP50 matches boxes at IoU 0.50; AP50–95 averages across thresholds from 0.50 to 0.95. This evaluation overlap is separate from the NMS threshold. I used the same images and settings for both models and mapped the original bottle class 39 to class 0. AP uses confidence 0.001 to keep the score curve; displayed apartment boxes use 0.25.” The paired AP runs both use `imgsz=960` and NMS `iou=0.45`.

9. **Can you call 0.929 the test accuracy?**

   “No. It is bottle AP50 on ten validation images containing fifteen labeled bottles. Those images also selected the best checkpoint. I need a separate labeled test set to estimate generalization. The apartment photos have no complete ground-truth boxes, so I report qualitative errors instead of accuracy.”

10. **Why did apartment detections get worse, and what would you do next?**

    “The training images and apartment photos differ in viewpoint, lighting, clutter, and bottle types. That is domain shift. The small dataset, label noise, and single-class training may also contribute; I did not isolate their effects. I would add representative apartment images and negative examples, check labels, and reserve an independent test set.”

11. **How does YOLOE use text prompts? Did you train it on these photos?**

    “MobileCLIP encodes class names as text features. YOLOE's RepRTA module refines their alignment with image-region features. I cached the resulting embeddings for 33 names and reused them. I chose the vocabulary after inspecting the photos but before viewing YOLOE predictions, then kept it fixed. I did not fine-tune YOLOE.” [YOLOE text-prompt workflow](https://docs.ultralytics.com/models/yoloe/)

12. **What are SAVPE and LRPC? Did you run those modes?**

    “SAVPE supports visual examples such as a box around an object. LRPC supports the separate prompt-free model and its built-in vocabulary. They are parts of the YOLOE architecture, but my experiment used only text prompts. I did not execute visual prompting or prompt-free inference. I used YOLOE-11m, not the tutorial's YOLOE-v8l or YOLO26.” [YOLOE prompting modes](https://docs.ultralytics.com/models/yoloe/)

If asked whether a result is correct, point to the actual object and its box. If you are uncertain, say so. A confidence score and a plausible label are model outputs, not ground truth.
