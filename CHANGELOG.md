# Changelog

## 2026-11-27: CSRNet inference preprocessing fixed

`methods_comparison/csrnet_infer.py` fed the network grayscale input without the
ImageNet normalization used in training (`csrnet/train_csrnet.py`:
`ToTensor` + `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`).
With the trained checkpoint this compressed the count predictions to roughly
300 to 380 strips regardless of the true count.

- `CSRNetInfer` now uses the training preprocessing by default (`preproc="train"`).
  `preproc="legacy"` keeps the old path only to reproduce earlier numbers.
- New method `CSRNetInfer.preprocess()`; `qualitative_synth.py` uses it instead of
  its own grayscale transform.
- Regression test `test_preprocess_matches_training_pipeline` checks that the
  inference input equals the training input.

All CSRNet counts, density-regime errors, late-fusion results and temporal
consistency values produced with this repository before this change were
computed with the faulty preprocessing. On the 100-frame synthetic test split the
CSRNet MAE changes from 121.3 to 20.4 strips.
