# Voyage Analytics — Integrating MLOps in Travel
---

## Project Overview

Production-grade ML system for a travel booking platform built on 271,888 records from three linked datasets. Three distinct ML models are developed and deployed using MLOps best practices:

1. **Flight Price Prediction** — Regression model (Random Forest, R² ≈ 1)
2. **Gender Classification** — Name-to-gender via Sentence Transformers + PCA
3. **Hotel Recommendation Engine** — Content-based filtering (cosine similarity)

All three models are tracked via MLflow, containerised with Docker, and served through a Streamlit web interface.

---

## Dataset

| File | Records | Description |
|---|---|---|
| Flights | 271,888 | Route, price (TARGET), flight type, agency, time, distance |
| Users | 271,888 | Name, gender (TARGET for classification), age, company |
| Hotels | 271,888 | Hotel name, place, days, price — 70% null (flight-only bookings) |

Join key: `travelCode` + `userCode`.

---

## EDA Key Findings

- Top route: Aracaju (SE) → Florianopolis (SC) — 8,000+ flights; high volume does not reduce price
- Price strongly positively correlated with distance and time (~+0.64)
- Time and distance are perfectly correlated (r=1.0) — same information, different units
- FlyingDrops agency has widest price spread; Rainbow/CloudFy cluster at fixed price points
- User age: approximately normal distribution, median ~42 years (range 30–55)
- 190,784 null hotel rows (70%) — valid: most bookings are flight-only

---

## Model Details

### Model 1 — Flight Price Prediction (Regression)

| Model | R² | Notes |
|---|---|---|
| Linear Regression | Moderate | Cannot capture non-linear relationships |
| Ridge / Lasso / ElasticNet | Lower | Linear — limited by price↔distance non-linearity |
| Decision Tree | ~1.0 | Overfits without pruning |
| **Random Forest** | **≈ 1.0** | **Best — handles non-linear distance/price interactions** |

- Top features: distance, time, flightType, agency
- SHAP confirms: distance dominates; first class adds a consistent fixed premium

### Model 2 — Gender Classification from Name

- Sentence Transformer (`all-MiniLM-L6-v2`) encodes names as 384-dim vectors
- PCA reduces to 2–3 components while retaining gender-discriminating signal
- Classifier (Logistic Regression / Random Forest) trained on embeddings
- Errors concentrated on genuinely unisex/ambiguous names

### Model 3 — Hotel Recommendation Engine

- Content-based filtering: TF-IDF vectors of hotel attributes (location, price range, type)
- User trip profile encoded identically; ranked by cosine similarity
- Top-K (K=5) recommendations returned with similarity scores
- Cold-start capable — no prior booking history required

---

## Data Preprocessing

- Dropped duplicate `code` column after merge
- `pd.to_datetime()` on date columns; extracted year, month, day-of-week
- Hotel nulls: hotel_name → 'No Hotel', hotel_price → 0; `has_hotel` binary flag added
- `log10(price)` transformation to reduce right skewness
- VIF analysis: time and distance (VIF=∞) handled via Ridge regularisation
- StandardScaler on all numeric features

---

## MLOps Architecture

| Component | Tool | Purpose |
|---|---|---|
| Experiment Tracking | MLflow | Log hyperparameters, metrics, artifacts per run |
| Containerisation | Docker | Reproducible deployment across environments |
| Web Interface | Streamlit | Browser UI for all 3 models — no coding required |
| Pipeline Orchestration | Apache Airflow | Weekly retraining DAG |

---

## Tech Stack

```
Python | Pandas | NumPy | Scikit-learn | Sentence Transformers | MLflow
Docker | Streamlit | Airflow | Matplotlib | Seaborn | SHAP
```

---

## Repository Contents

```
Voyage-Analytics-Integrating-MLOps-in-Travel/
├── Capstone_Project_Productionization_of_ML_Systems.ipynb   # Main notebook
├── Dockerfile                                                 # Docker container config
├── requirements.txt                                          # Python dependencies
├── Voyage_Analytics_MLOps_Presentation.pptx                 # Presentation (16 slides)
├── Flight price regression/                                  # Regression model files
├── Gender classification files/                              # Classification model files
├── Hotel recommender system/                                 # Recommendation engine
└── ml flow/                                                  # MLflow tracking artifacts
```

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Capstone_Project_Productionization_of_ML_Systems.ipynb
# or via Docker:
docker build -t voyage-analytics .
docker run -p 8501:8501 voyage-analytics
```

---

*AlmaBetter M.Sc. Data Science | 2026 | Vishal Kumar Singh*
