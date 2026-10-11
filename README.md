# Deep-learning models for 24-hour movement-behaviour classification

This repository contains the preprocessing and deep-learning model definitions used for wrist-worn accelerometer movement-behaviour classification in the Capture-24 study.

The repository intentionally focuses on the two methodological components needed to inspect the deep-learning pipeline:

1. preprocessing of raw accelerometer signals into model-ready 10-second windows; and
2. the LSTM, BiLSTM, GRU, and 1D-CNN architectures.

Evaluation and result-generation code are not included.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── figures/
│   └── deep_learning_architectures.png
└── src/
    ├── __init__.py
    ├── preprocessing.py
    └── models.py
```

## Preprocessing

Raw wrist accelerometer data are represented by the three acceleration axes (`x`, `y`, and `z`). Vector magnitude is calculated as:

`VM = sqrt(x^2 + y^2 + z^2)`

The signals are divided into non-overlapping 10-second windows. At a sampling frequency of 100 Hz, a complete window contains 1,000 samples. Windows containing missing values or an incomplete number of samples are excluded. The movement-behaviour label for a valid window is obtained from the modal label within that window.

The resulting input shape for the deep-learning models is:

```text
(number of windows, 1000 samples, 4 channels)
```

where the four channels are `x`, `y`, `z`, and `VM`.

The class order used in the code is:

```text
0  light
1  moderate-vigorous
2  sedentary
3  sleep
```

## Models

Four deep-learning architectures are provided in `src/models.py`:

- LSTM
- Bidirectional LSTM (BiLSTM)
- GRU
- 1D-CNN

The implementations retain the layer sizes and dropout setting used in the study. The functions return uncompiled Keras models so that training configuration is kept separate from the architecture definitions.

![Deep-learning architectures](figures/deep_learning_architectures.png)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Example: preprocessing one participant

```python
from src.preprocessing import (
    extract_windows,
    load_annotation_dictionary,
    load_participant,
)

annotation_dictionary = load_annotation_dictionary(
    "path/to/annotation-label-dictionary.csv"
)

data = load_participant(
    "path/to/participant.csv",
    annotation_dictionary,
)

X, y = extract_windows(data)
print(X.shape)
print(y.shape)
```

## Example: build a model

```python
from src.models import build_lstm

model = build_lstm(input_shape=(1000, 4), num_classes=4)
model.summary()
```

The other architectures can be created with `build_bilstm`, `build_gru`, and `build_cnn`.

## Data

The Capture-24 dataset is not redistributed in this repository. Users should obtain the data from its official source and provide local paths to the participant files and annotation dictionary.

## Citation

If you use this code, please cite the associated paper:

> Sameh, A., Rostami, M., Oussalah, M., & Farrahi, V. *Classification of 24-hour movement behaviors from wrist-worn accelerometer data: From handcrafted features to deep learning techniques.*


