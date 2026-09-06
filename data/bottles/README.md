# Bottle dataset

50 images from Open Images: 40 for training (66 boxes) and 10 for validation (15 boxes). All labels use class `0: bottle`. The two apartment photos are separate test inputs.

Images were selected from the Open Images validation pool with seed 42, after excluding group and depiction annotations. All bottle boxes in each selected image were converted to YOLO format. Most examples are drink or cosmetic bottles, with some noisy labels.

Image authors, links, and licenses are listed in [sources.csv](sources.csv). Images are CC BY 2.0; bounding-box annotations are by Google LLC under CC BY 4.0.

Sources: [Open Images downloads](https://storage.googleapis.com/openimages/web/download_v7.html), [bounding-box annotations](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv), and [license details](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses).
