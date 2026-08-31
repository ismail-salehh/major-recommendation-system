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

---

## Running the Recommendation Web UIs

### Option A: Launch Streamlit UI (Recommended)
Run the full interactive Streamlit web application:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Launch Gradio UI
Run the lightweight Gradio interface:

```bash
python gradio_app.py
```
Open your browser at `http://localhost:7860`.

---

## Programmatic Usage & CLI

### 1. Using the Command Line Interface (CLI)
You can test predictions directly from your terminal:

```bash
python scripts/predict.py --gpa 91.5 --tech 5 --health 1 --business 3 --arts 4
```

**Available CLI Flags:**
- `--gpa`: Tawjihi GPA percentage (60.0 to 100.0)
- `--tech`: Interest level in Technology (1 to 5)
- `--health`: Interest level in Health & Medicine (1 to 5)
- `--business`: Interest level in Business & Management (1 to 5)
- `--arts`: Interest level in Arts & Creativity (1 to 5)
- `--thinking`: `"تفكير منطقي"` or `"تفكير تحليلي"`
- `--personality`: `"اجتماعي (بتحب تختلط بالناس بكثرة )"` or `"منطوي (ما بتحب كثير تختلط بالناس)"`
- `--study`: `"مزيج بين النظري والعملي"`, `"دراسة فيها عملي (تطبيق)"`, or `"نظري"`
- `--work`: `"الاثنين معا"`, `"العمل الجماعي (ضمن فريق)"`, or `"العمل الفردي"`
- `--top_k`: Number of recommendations to return (default: `5`)

### 2. Using the Python API
You can load the trained model into any Python script:

```python
from src.recommender import MajorRecommender

# Initialize engine (loads default best ensemble model)
recommender = MajorRecommender()

# Define student profile
student_profile = {
    'GPA': 90.0,
    'Interest in Technology': 5,
    'Interest in Health and Medicine': 1,
    'Interest in Business and Management': 3,
    'Interest in Arts and Creativity': 2,
    'Thinking Style': 'تفكير منطقي',
    'Personality Type': 'منطوي (ما بتحب كثير تختلط بالناس)',
    'Preferred Study Method': 'دراسة فيها عملي (تطبيق)',
    'Preferred Work Style': 'العمل الفردي'
}

# Generate top 5 recommendations
recommendations = recommender.recommend(student_profile, top_k=5)

for rec in recommendations:
    print(f"Rank #{rec['rank']}: {rec['major']} ({rec['category']}) - Match: {rec['confidence_pct']}")
```

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
