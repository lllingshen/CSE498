# Bottle dataset

50 Open Images photographs: **40 training images (66 boxes)** and **10 validation images (15 boxes)**. Every label uses class `0: bottle`. The apartment test photos are separate.

The images came from the official Open Images validation pool. Images with bottle group or depiction annotations were excluded; eligible IDs were shuffled with seed 42. The first 50 were selected and split 40/10. This is our course split, not the official Open Images split. All supplied bottle boxes were converted to normalized YOLO coordinates.

`manifest.csv` lists each image's ID, split, author, original URL, license, and hash. Keep this attribution with the images. The small dataset mainly contains drink and cosmetic bottles, with some noisy source labels; it is not specific to insulated bottles or apartment scenes.

Images retain their authors' CC BY 2.0 licenses. Bounding-box annotations are by Google LLC under CC BY 4.0. Sources: [Open Images downloads](https://storage.googleapis.com/openimages/web/download_v7.html), [annotation CSV](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv), and [license details](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses).

Reference: Kuznetsova et al., *The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale*, IJCV, 2020.
