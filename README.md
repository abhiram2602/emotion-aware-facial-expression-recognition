# Emotion-Aware Safety System Using Facial Expression Recognition

Seven-class CNN image classification academic project. The original method uses **96×96 images**, TensorFlow/Keras and **15 epochs**.

## Run
```bash
pip install -r requirements.txt
# Extract your original Dataset(1).zip so data/processed_data has seven class folders
python src/train.py --data data/processed_data --output outputs/full_run --epochs 15
```

The academic report states **62.64% accuracy** and **1.1862 loss**. These values have **not** been independently reproduced. The script saves measured validation accuracy, loss, model, confusion matrix and learning curves.

The dataset is not redistributed because its license has not been established. Facial expression is not a reliable proxy for internal emotion or personal safety; this is a research demonstration.

Original notebooks, reports, presentation and demo video are available in the full ChatGPT project archive.