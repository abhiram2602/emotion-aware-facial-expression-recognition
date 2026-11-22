# Execution status

- Uploaded source dataset inspected: approximately 49,779 images across seven emotion classes.
- Small PyTorch baseline previously ran for two epochs; validation accuracy was 14.29% on the 630-image sample.
- Full original TensorFlow/Keras 15-epoch training has **not** been completed here.
- Original report: **62.64% accuracy and 1.1862 loss**, not independently reproduced.
- To run: `python src/train.py --data data/processed_data --output outputs/full_run --epochs 15`.
- The original dataset is not included because its redistribution license is unverified.
