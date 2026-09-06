# Recorded experiment evidence

These files come from the completed local experiments. Running the portable `src/` programs writes new files to `runs/` instead of replacing this directory.

`source_manifest.json` maps each copied file to its original relative source path and records both hashes. Images, dataset manifests, label files, and numeric results are retained. Copied text replaces the original absolute workspace and home prefixes with `<ORIGINAL_WORKSPACE>` and `<ORIGINAL_HOME>`; translated review comments are identified in the manifest. Historical hashes inside a copied log or metadata file still refer to the original source bytes, not to a redacted or translated copy.

`experiment1/baseline/` records the original YOLO11n inference; `experiment1/improved/` records YOLO11m with the improved inference settings. `experiment2/training/` contains training curves and arguments, while `experiment2/comparison/` contains the corrected bottle-only comparison. `experiment3/` contains the original YOLOE predictions, the vocabulary registration, and model/text-encoder provenance. Its first failed setup attempt is retained and clearly separate from the successful run.

`executed_scripts/` contains original source snapshots for provenance. Their historical folder layout is not the portable submission layout. Use the entry points in `src/` when rerunning the project.
