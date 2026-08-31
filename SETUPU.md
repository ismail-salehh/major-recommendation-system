# Set Up
## 1. Create Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 2. Install Dependencies
Install all required libraries using `pip`:

```bash
pip install -r requirements.txt
```

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