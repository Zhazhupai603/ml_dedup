# ML-based UMI deduplication

Machine-learning pipeline for read deduplication decisions on RNA-seq / scRNA-seq
alignments, replacing heuristics with a gradient-boosted classifier trained on
per-read features extracted from BAM files.

## Layout

| File | Role |
|------|------|
| `01_extract_features.py` | Extract per-read features from BAM (single sample) |
| `02_eda.py` | Exploratory data analysis on extracted features |
| `03_train.py` | Train + evaluate classifier on a single sample |
| `extract_features_wg.py` | Whole-genome feature extraction (multi-BAM, CLI) |
| `train_wg.py` | Whole-genome training (HistGBM + baselines) |
| `train_wg_v2.py` | Improved whole-genome training (extended features, SHAP-style importance) |
| `reformat_qname.py` | Utility: collapse Illumina comment, preserve UMI suffix |

## Pipeline

```
BAM ─► 01_extract_features.py ─► features.parquet
                                       │
                                       ├─► 02_eda.py        (diagnostics)
                                       └─► 03_train.py      (model)
```

Whole-genome variant substitutes `extract_features_wg.py` + `train_wg(_v2).py`.

## Install

```bash
pip install -r requirements.txt
```

Requires `pysam` (htslib) system library; see [pysam docs](https://pysam.readthedocs.io/).

## Notes

* Default I/O paths inside scripts point at `/home/eagle/try3/bench/results_*`
  (hard-coded from the original workspace). Override via the exposed CLI flags
  in `*_wg.py`; the `0*` numbered scripts use module-level `FEATURES_PATH` /
  `OUT_DIR` constants at the top of each file.
* No model weights are shipped. Retrain from BAM inputs.

## License

None specified. Internal research code.
