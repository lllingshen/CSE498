# Small Bottle dataset

The dataset contains 50 independent Open Images photographs: 40 training images with 66 bottle boxes, and 10 validation images with 15 bottle boxes. Every YOLO label uses class `0: bottle`, corresponding to Open Images class `/m/04dr76w`.

Selection was fixed before training. From the official Open Images **validation** pool, all images containing a Bottle box marked `IsGroupOf=1` or `IsDepiction=1` were excluded. This left 165 eligible images. Sorted IDs were shuffled with Python `random.Random(42)`, the first 50 were selected, and the first 40 / last 10 formed the course training / validation split. All downloads succeeded, with no replacements. This is a custom course split, not the official Open Images train/validation split.

All publisher-provided Bottle boxes in each selected image were retained. The original fields are in `source_bottle_boxes.csv`; normalized YOLO labels are under `labels/`. The manifest files contain each image's ID, split, original page, author, license, and SHA-256. The 50 image files and the 40/10 assignment are unchanged from the training run. The two apartment photographs are not in this dataset.

This random subset mainly contains drink, wine, and cosmetic bottles; it is not a thermos-specific training set. Some publisher annotations appear noisy: for example, `a1119d540ba829e4` looks like a beer glass but is labelled Bottle. Such labels were retained and the sample was not selected based on model performance. This limits what the small experiment can establish.

Validation checks found 50 unique IDs, original URLs, and file hashes, legal coordinates, and no candidate near duplicates at pHash distance at most 8. The check is not a guarantee against all possible near duplicates. See `validation_checks.json` and `selection.json`.

Run `python src/prepare_dataset.py` from the repository root to verify the bundled data and create a YAML with resolved local paths under `runs/runtime/`. The entry points do this automatically. No data resampling or internet access is needed. The original download/conversion source is preserved in `results/executed_scripts/prepare_dataset.py` as a historical snapshot.

Sources:

- [Official Open Images download page](https://storage.googleapis.com/openimages/web/download_v7.html)
- [Full validation box CSV](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv)
- [Class names](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)
- [Per-image authors and license metadata](https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv)
- [Official CVDF image mirror](https://github.com/cvdfoundation/open-images-dataset)

Annotations are published by Google LLC under CC BY 4.0. The selected image metadata lists CC BY 2.0; preserve each image's author and source attribution in the manifests. The mirror images have at most 1024 pixels on the longest side. [Official license description](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses).

Reference: Kuznetsova et al., *The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale*, IJCV, 2020.
