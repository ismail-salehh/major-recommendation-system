# University Major Recommendation System
A machine learning recommendation platform that guides high school (Tawjihi) graduates toward optimal university academic majors based on GPA, personality traits, thinking styles, and area interest ratings.

---

##  Process & System Features

- **Data Collection**: A public form was shared with current/previous university students. They were questioned about their current/previous university major - which would later form our target - as well as other questions. A total of **1390** separate answers were collected that would later form the dataset.
- **Data Preprocessing & Arabic Normalization**: Robust text cleaner that unifies Arabic letter variants, handles diacritics, strips noise strings, cleans GPA numerical outliers, and standardizes 61 canonical university majors.
- **Improved Machine Learning Ensemble**: Evaluates multiple architectures (Random Forest, Extra Trees, XGBoost, LightGBM, KNN) and combines them into a **Soft Voting Ensemble**.
- **Interactive Web Interfaces**: Offers both a modern **Streamlit UI** (`app.py`) and a lightweight **Gradio UI** (`gradio_app.py`).
- **CLI & Python API**: Command-line interface and modular Python engine for easy integration into external applications.

---

## Quick Start & Installation

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Launching the Streamlit UI
Run the Streamlit web application from the terminal:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

> For other setup options, please refer to [SETUPU.md](SETUPU.md).
---


### Benchmark Results Comparison

| Model Architecture | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Top-10 Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Notebook Baseline)** | 62.33% | 82.43% | 87.94% | 94.48% |
| **KNN Classifier (n=5)** | 67.45% | 85.97% | 88.57% | 89.44% |
| **LightGBM Classifier** | 72.03% | 87.00% | 91.49% | **96.45%** |
| **XGBoost Classifier** | 73.68% | 87.71% | 91.80% | 96.14% |
| **Extra Trees (Tuned)** | 74.70% | 85.97% | 89.36% | 94.41% |
| **Soft Voting Ensemble (ET + RF + XGB + LGB)** | **75.81%** | **87.63%** | **91.80%** | 96.14% |

---

## Data Cleaning & Preprocessing Workflow

1. **Text Normalization**: `src/text_cleaner.py` converts raw Arabic text to canonical form, removes Tashkeel (diacritics), unifies alef/ta-marbouta variants, and strips garbage noise phrases.
2. **Canonical Mapping**: `src/major_mapper.py` maps hundreds of raw text variations into 61 standard university majors and categorizes them into 10 broad academic domains (Tech, Engineering, Business, Health, Science, Languages, Humanities, Education, Arts, Tourism).
3. **Synthetic Oversampling**: Applies **SMOTE** on minority major classes to overcome severe class imbalance.
4. **Artifact Serialization**: Saves encoders, metadata, and the trained ensemble model into `models/` for low-latency production serving.
