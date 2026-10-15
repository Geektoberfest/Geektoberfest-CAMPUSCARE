# CAMPUSCARE — Intelligent Complaint Triage

CAMPUSCARE is a Django-based smart campus complaint portal that uses machine learning and text similarity to analyze student complaints.

## Live Demo

🌐 **[Open CAMPUSCARE](https://campuscare-025j.onrender.com)**

## Current Prototype

Students can:
- Submit a campus complaint in English, Hindi, Hinglish, or Marathi
- Get an automatically detected complaint language
- Get an automatically predicted complaint category
- Get a calibrated prediction confidence score
- Get an intelligent priority level with hazard and negation handling
- See the department routed for the complaint
- Get a duplicate/similar-complaint warning

Complaints are stored in SQLite for local development and PostgreSQL when deployed with the included Render blueprint.

## How It Works

```text
Student Complaint (English / Hindi / Hinglish / Marathi)
       ↓
Language Detection & Unicode Normalization
       ↓
Preprocessing & Contraction Expansion
       ↓
Script-Aware, Sublinear TF-IDF Text Features (Unigrams & Bigrams)
       ↓
Category Prediction & Confidence Estimation
       ↓
Multilingual, Negation-Aware Priority Triage
       ↓
Department Routing
       ↓
Duplicate Detection
       ↓
SQLite / PostgreSQL Database
```

For detailed architectural trade-offs, language detection heuristics, and limitations, refer to [MULTILINGUAL.md](MULTILINGUAL.md).

For detailed benchmark comparisons, classifier experiments, and stated limitations, see [EVALUATION.md](EVALUATION.md).

## Categories

- IT & Wi-Fi
- Hostel
- Academic
- Infrastructure
- Transport

## Technology

Python, Django, Pandas, NumPy, scikit-learn, TF-IDF, Logistic Regression, HTML/CSS, SQLite and GitHub.

## Dataset

The repository includes a 1,000-row synthetic campus complaint dataset in `data/complaints.csv` covering English, Hindi, Hinglish, and Marathi complaints across 5 categories.

The dataset is provided for development and competition purposes. It is synthetic and should not be treated as real student feedback.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Geektoberfest/Geektoberfest-CAMPUSCARE.git
cd Geektoberfest-CAMPUSCARE
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Apply migrations

```bash
python manage.py migrate
```

### 5. Start the server

```bash
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## Deploy on Render

The repository includes `build.sh` and `render.yaml` for a Render deployment. The blueprint provisions a PostgreSQL database, installs production dependencies, collects static files, runs migrations, and starts Django with Gunicorn.

In Render, create a new Blueprint Instance from this repository and apply the included `render.yaml`. Render's documentation describes this Blueprint workflow for Django deployments. The deployed service will receive a generated `DJANGO_SECRET_KEY` and `DJANGO_DEBUG=false`.

## Competition Task

This repository intentionally provides a working prototype rather than a finished product.

### Open Challenges

- **Easy:** [Add complaint evaluation tests](https://github.com/Geektoberfest/Geektoberfest-CAMPUSCARE/issues/4)
- **Medium:** [Build a staff complaint dashboard](https://github.com/Geektoberfest/Geektoberfest-CAMPUSCARE/issues/1)
- **Medium:** [Improve complaint classification and triage](https://github.com/Geektoberfest/Geektoberfest-CAMPUSCARE/issues/2)
- **Hard:** [Add multilingual complaint support](https://github.com/Geektoberfest/Geektoberfest-CAMPUSCARE/issues/3)

Participants should choose an issue, keep changes focused, run the tests, and link their Pull Request to the issue. Difficulty labels describe expected scope rather than a guaranteed completion time.

## Testing

Run:

```bash
python manage.py test
```

The included tests cover core complaint creation and basic ML predictions.

## Project Structure

```text
CAMPUSCARE/
├── complaints/
│   ├── migrations/
│   ├── templates/complaints/
│   │   └── home.html
│   ├── admin.py
│   ├── apps.py
│   ├── ml.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── config/
├── data/
│   ├── complaints.csv
│   └── generate_dataset.py
├── db.sqlite3
├── manage.py
├── requirements.txt
└── README.md
```

## Note

CAMPUSCARE is a competition prototype. The included Django development configuration and SQLite database are intended for local development and experimentation, not production deployment.
