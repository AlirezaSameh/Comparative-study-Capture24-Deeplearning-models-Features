# Deep-learning models and handcrafted feature extraction for 24-hour movement-behaviour classification

This repository contains preprocessing utilities, handcrafted feature extraction, and deep-learning model definitions for classifying 24-hour movement behaviours from wrist-worn accelerometer data in the **CAPTURE-24** study. It documents two separate approaches: extracting established accelerometer features for conventional machine-learning methods and using windowed acceleration signals as inputs to deep-learning architectures.

The provided files cover:

1. Preprocessing accelerometer recordings into non-overlapping 10-second windows.
2. Extracting handcrafted time-domain, frequency-domain features from each window.
3. Defining LSTM, bidirectional LSTM (BiLSTM), GRU, and 1D-CNN architectures.

**Scope:** The supplied repository files contain preprocessing, feature extraction, and model definitions.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── Feature extraction.ipynb
├── Deep_ Learning_Structures.pdf
├── __init__.py
├── preprocessing.py
└── models.py
```


## Dataset: CAPTURE-24

**CAPTURE-24** is a publicly available free-living human activity recognition dataset collected in Oxfordshire, United Kingdom. The publicly released dataset contains accelerometer signals and activity annotations, not the camera images [1].

The original activity annotations can be mapped into broader movement-behaviour categories using the accompanying annotation dictionary. This repository uses the `label:Walmsley2020` mapping and the following four labels:

| Encoded label | Movement behaviour |
| ---: | --- |
| 0 | Light physical activity (`light`) |
| 1 | Moderate-to-vigorous physical activity (`moderate-vigorous`) |
| 2 | Sedentary behaviour (`sedentary`) |
| 3 | Sleep (`sleep`) |

**Data access:** Obtain the dataset and its annotation-label dictionary from the official CAPTURE-24 release described in [1]. The raw data are not redistributed in this repository.

## Preprocessing

The raw accelerometer recordings contain three axes (`x`, `y`, `z`). The preprocessing code calculates vector magnitude (`VM`):

```text
VM = sqrt(x² + y² + z²)
```

The recordings are segmented into **non-overlapping 10-second windows**. At 100 Hz, each complete window contains **1,000 samples**. Windows containing missing values or an incomplete number of samples are excluded. Each valid window receives the most frequent movement-behaviour label within that window; windows without a unique modal label are skipped.

The raw-signal deep-learning input has shape:

```text
(number of windows, 1000 samples, 4 channels)
```

The four channels are `x`, `y`, `z`, and `VM`. The supplied `preprocessing.py` provides `load_annotation_dictionary`, `load_participant`, `extract_windows`, and `encode_labels`.

## Handcrafted feature extraction

The supplied **`Feature extraction.ipynb`** notebook defines `extract_features(xyz, sample_rate=100)` for one triaxial acceleration window. It constructs a dictionary of **104 named features** derived from the `x`, `y`, `z`, and calculated `VM` signals. These feature families are established in accelerometer-based human activity recognition research, including time-domain statistics, axis correlations, FFT-derived measures, dominant spectral components, and orientation-related angles [2–5]. The **specific 104-feature combination is the implementation in this notebook**, not a feature set attributed in its entirety to any one publication.

| Feature group | Implemented features | Count | Relevant literature |
| --- | --- | ---: | --- |
| Mean and standard deviation | Mean and standard deviation for `x`, `y`, `z`, `VM` | 8 | [2–4] |
| Distribution statistics | Minimum, 25th percentile, median, 75th percentile, maximum for each signal | 20 | [2–4] |
| Inter-axis correlation | Pearson correlations `xy`, `yz`, `zx` | 3 | [2–4] |
| FFT magnitudes | 15 named FFT magnitudes (`fft1`–`fft15`) for each of `x`, `y`, `z`, `VM` | 60 | [2–5] |
| Spectral peaks | First and second peak magnitudes and corresponding frequencies for `VM` | 4 | [2, 4, 5] |
| Summed spectral magnitude | `totpower_VM`, nominally for the 0.3–15 Hz band | 1 | [2, 4] |
| Dominant frequency band | Dominant frequency and magnitude within the nominal 0.6–2.5 Hz band for `VM` | 2 | [2, 4] |
| Angular features | Mean and standard deviation of roll, pitch, yaw, calculated from acceleration axes | 6 | [2, 3] |
| **Total** | | **104** | |

### Feature definitions

**Time-domain features.** Mean and standard deviation summarize acceleration level and variability. Quantiles describe the distribution within a window, and inter-axis correlations describe linear relationships between acceleration axes. These are commonly used features in activity-recognition research [2–4].

**Frequency-domain features.** The notebook applies the fast Fourier transform (FFT) to each axis and vector magnitude to characterize oscillatory movement. It names 15 FFT-magnitude features per signal and extracts additional vector-magnitude spectral peaks, summed magnitudes, and dominant-band components. Frequency-domain and dominant-frequency features are established accelerometer descriptors [2, 4, 5].

### Important implementation notes

The table describes the **intended feature names and categories**, not a verified numerical output of the notebook. The original notebook has not been changed. Before using these features in reproducible experiments, note that:

- `min_VM`, `q25_VM`, `med_VM`, `q75_VM`, and `max_VM` are currently calculated from `z` rather than `VM` in the notebook.
- The FFT frequency indices are obtained from the full frequency vector but applied to a sliced half-spectrum array, which can shift the selected bins; some selected indices may be incompatible with the available array depending on the implementation.
- `totpower_VM` is implemented as a **sum of FFT magnitudes**, not squared spectral power or a formally estimated power spectral density. The name should not be interpreted as physical power without further verification.
- The notebook imports SciPy; its use requires `scipy` even though the supplied `requirements.txt` does not list it. The notebook also assumes NumPy is available as `np`.
- The `sample_rate` argument is declared, but the notebook internally fixes `dt=0.01` and `n=1000`, corresponding to 100 Hz and 1,000 samples.

These are documentation notes only; no scientific or computational changes have been made to the notebook.

### Example: feature extraction

```python
# Run the feature-extraction notebook to define extract_features first.
# X is a set of complete windows with x, y, z channels.

features_one_window = extract_features(X[0, :, :3])
print(len(features_one_window))  # 104 named entries
```

The handcrafted-feature approach yields one feature dictionary per valid window, suitable for constructing a tabular matrix for conventional machine-learning models. A complete feature-matrix builder and conventional model training scripts are not supplied in the uploaded files.

## Deep-learning models

Four architectures are implemented in `models.py`:

| Model | Main layers | Output |
| --- | --- | --- |
| LSTM | Four LSTM layers (256, 128, 64, 32 units) | Four-class softmax |
| BiLSTM | Four bidirectional LSTM layers (256, 128, 64, 32 units per direction) | Four-class softmax |
| GRU | Four GRU layers (256, 128, 64, 32 units) | Four-class softmax |
| 1D-CNN | Four Conv1D layers (256, 128, 64, 32 filters), each followed by max pooling | Four-class softmax |

The architectures use dropout (`0.2` by default) and batch normalization after their principal layers, followed by a 16-unit dense layer and softmax classification. Their default input shape is `(1000, 4)`. The functions return **uncompiled Keras models**; compilation, training, and evaluation are not included.

The supplied `Deep_ Learning_Structures.pdf` illustrates the four model structures.


## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To run the feature-extraction notebook, install its additional imported dependency if needed:

```bash
pip install scipy notebook
```

## Example: preprocessing one participant

```python
from preprocessing import (
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
from models import build_lstm

model = build_lstm(input_shape=(1000, 4), num_classes=4)
model.summary()
```

Other architectures are available as `build_bilstm`, `build_gru`, and `build_cnn`.

## Data

CAPTURE-24 data are not distributed with this repository. The source dataset, annotation dictionary, and data documentation should be obtained from the original dataset release [1]. The study's participant-level split or any subsequent experimental splits are **not implemented in the supplied repository files**; avoid interpreting the examples as a complete evaluation protocol.

## References

1. Chan, S., Hang, Y., Tong, C., et al. (2024). *CAPTURE-24: A large dataset of wrist-worn activity tracker data collected in the wild for human activity recognition.* Scientific Data, 11, 1135. DOI: `10.1038/s41597-024-03960-3`. [Google Scholar search](https://www.google.com/search?q=Chan+Hang+Tong+2024+CAPTURE-24+Scientific+Data).
2. *The Key Factors in Physical Activity Type Detection Using Real-Life Data: A Systematic Review.* (2019). Frontiers in Physiology. DOI: `10.3389/fphys.2019.00075`. [Google Scholar search](https://www.google.com/search?q=The+Key+Factors+in+Physical+Activity+Type+Detection+Using+Real-Life+Data+systematic+review).
3. *A systematic review of smartphone-based human activity recognition methods for health research.* (2021). npj Digital Medicine. DOI: `10.1038/s41746-021-00514-4`. [Google Scholar search](https://www.google.com/search?q=systematic+review+smartphone+based+human+activity+recognition+methods+health+research+2021).
4. *Human Activity Recognition using Inertial, Physiological and Environmental Sensors: A Comprehensive Survey.* (2020). [Google Scholar search](https://www.google.com/search?q=Human+Activity+Recognition+using+Inertial+Physiological+and+Environmental+Sensors+A+Comprehensive+Survey).
5. *Energy-Efficient Real-Time Human Activity Recognition on Smart Mobile Devices.* (2016). [Google Scholar search](https://www.google.com/search?q=Energy-Efficient+Real-Time+Human+Activity+Recognition+on+Smart+Mobile+Devices+2016).

These references establish the dataset and the published precedent for the **types** of handcrafted features; they do not establish that the notebook reproduces any individual paper's exact feature-extraction implementation.

## Citation

If you use this research code, please cite the associated manuscript (bibliographic details to be updated when available):

> Sameh, A., Rostami, M., Oussalah, M., & Farrahi, V. *Classification of 24-hour movement behaviors from wrist-worn accelerometer data: From handcrafted features to deep learning techniques.*

Please also cite the CAPTURE-24 dataset paper [1] when using the dataset.
