# 🛡️ Phishing Website Detection with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)](https://xgboost.readthedocs.io/)
[![LIME](https://img.shields.io/badge/LIME-0.2.0-purple.svg)](https://github.com/marcotcr/lime)
[![SHAP](https://img.shields.io/badge/SHAP-0.42.1-red.svg)](https://shap.readthedocs.io/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Available-success)](https://phishguard-n52t.onrender.com)

**🚀 Live Demo:** [https://phishguard-n52t.onrender.com](https://phishguard-n52t.onrender.com)

A machine learning-powered web application that detects phishing websites in real-time and provides **human-readable explanations** using Explainable AI (LIME & SHAP). Built with a Flask backend and a modern, interactive web interface.

---

## ✨ Key Features

- **Multi-Model Ensemble** — Random Forest, XGBoost, and Neural Network (MLP) classifiers
- **Explainable AI** — LIME and SHAP integration for transparent, interpretable predictions
- **46+ Feature Extraction** — URL structure, page content, host/DNS, and SSL analysis
- **Key Analysis Factors** — Feature-driven risk explanations with actionable security advice
- **Modern Web UI** — Dark/Light theme toggle, animated results, and auto-scroll
- **REST API** — JSON endpoints for single URL, batch analysis, and health checks
- **Batch Processing** — Analyze up to 100 URLs simultaneously

---

## 📁 Project Structure

```
phishing-detection-xai/
├── app.py                    # Flask web application (routes & API)
├── feature_extractor.py      # URL & page feature extraction (46+ features)
├── explainer.py              # LIME/SHAP explainable AI engine
├── predictor.py              # Prediction pipeline & model orchestration
├── model_trainer.py          # ML model training with GridSearchCV
├── uni_model_trainer.py      # Unified training pipeline
├── util.py                   # Utility functions
├── requirements.txt          # Python dependencies
│
├── models/                   # Pre-trained model files
│   ├── xgboost_model.joblib
│   ├── random_forest_model.joblib
│   ├── neural_network_model.joblib
│   ├── scalers.joblib
│   └── metadata.json
│
├── templates/
│   └── index.html            # Main web interface
│
├── static/images/            # Favicon and static assets
│
├── data/
│   ├── raw/                  # Raw datasets (UCI, PhishTank)
│   ├── test_urls.csv         # Sample test URLs
│   └── sample_urls.txt       # Quick-test URL list
│
├── create_sample_data.py     # Generate sample training data
├── download_datasets.py      # Download public phishing datasets
├── synthesize_data.py        # Synthesize additional training data
├── full_training.py          # Full model training entry point
└── quick_training.py         # Quick training with fewer samples
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### 1. Clone & Install

```bash
git clone https://github.com/riteshnarkar/Phishing-Website-Detection-with-Explainable-AI.git
cd Phishing-Website-Detection-with-Explainable-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

Open your browser and navigate to **http://localhost:5000**

### 3. Analyze a URL

1. Enter any URL in the input field
2. Select an ML model (XGBoost recommended)
3. Choose an explanation level (Quick / Detailed / Comprehensive)
4. Click **Analyze URL** and view the results with AI explanations

---

## 🤖 Machine Learning Models

| Model | Type | Strengths |
|-------|------|-----------|
| **XGBoost** *(recommended)* | Gradient Boosting | Best overall performance, handles imbalanced data |
| **Random Forest** | Ensemble | Robust, interpretable, handles missing values |
| **Neural Network** | MLP Classifier | Complex pattern recognition |

### Retraining Models

```bash
# Quick training (smaller dataset)
python quick_training.py

# Full training with GridSearchCV
python full_training.py
```

---

## 🔍 Explainable AI (XAI)

Every prediction is accompanied by a clear, feature-driven explanation consisting of two parts:

| Section | Description |
|---------|-------------|
| **Key Analysis Factors** | Top 4–5 risk signals (or safety signals) identified from the extracted features |
| **Security Recommendation** | Actionable advice based on the prediction |

The explanations are dynamically generated from the actual ML features — not generic templates. Each bullet maps directly to a detected risk indicator (e.g., new domain registration, IP-based URL, suspicious JavaScript).

### Explanation Techniques

- **LIME** — Generates local, interpretable explanations for individual predictions by perturbing input features
- **SHAP** — Provides globally consistent feature importance using Shapley values from game theory

### Example Output — Phishing

```
⚠️ DANGER: This website (suspicious-site.tk) looks definitely unsafe.

Key Analysis Factors:
  ⚠️ New Domain: Registered only 3 days ago. Phishing sites are often brand new.
  ⚠️ No Encryption: Site does not use HTTPS/SSL. Data is sent in plain text.
  ⚠️ Suspicious Keywords: URL contains alarm word(s): 'login', 'verify'.
  ⚠️ Long URL (127 chars): Excessively long URLs are often used to hide suspicious patterns.
  ⚠️ Suspicious TLD (.tk): '.tk' is a high-risk domain extension often abused by scammers.

Security Recommendation:
  • Do not enter your password or credit card info.
  • Close this page immediately.
  • If you are unsure, search for the official website on Google.
```

### Example Output — Legitimate

```
✅ SAFE: This website (github.com) looks real and safe.

Key Analysis Factors:
  ✅ Valid SSL: Connection is encrypted and certificate is valid.
  ✅ Established Domain: Registered over 17 years ago.
  ✅ Email Verified: Domain is set up to receive emails (MX record found).
  ✅ Clean URL: No suspicious keywords found.

Security Recommendation:
  • You can likely browse this site safely.
  • Still, never share your password unless you are sure.
  • Make sure the website address looks correct.
```

---

## 🧠 Feature Engineering

**46+ features** extracted across four categories:

| Category | Count | Examples |
|----------|-------|---------|
| **URL Structure** | 19 | URL length, subdomain count, special characters, entropy |
| **Page Content** | 14 | Login forms, iframe count, external link ratio, scripts |
| **Host & DNS** | 9 | Domain age, MX/SPF/DMARC records |
| **SSL/TLS** | 4 | SSL validity, certificate age |

---

## 🌐 API Reference

### Single URL Analysis

```http
POST /analyze
Content-Type: application/json

{
    "url": "https://example.com",
    "model": "xgboost",
    "explain": true,
    "explanation_level": "detailed"
}
```

### Batch Analysis

```http
POST /batch-analyze
Content-Type: application/json

{
    "urls": ["https://site1.com", "https://site2.com"],
    "model": "xgboost",
    "explain": true,
    "explanation_level": "basic"
}
```

### All Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/analyze` | POST | Analyze single URL (configurable explanation level) |
| `/analyze-quick` | POST | Quick analysis with basic explanation |
| `/analyze-detailed` | POST | Detailed multi-section analysis |
| `/analyze-comprehensive` | POST | Full analysis with statistics |
| `/batch-analyze` | POST | Batch URL analysis (up to 100) |
| `/api/v1/predict` | POST | REST API endpoint |
| `/health` | GET | Health check |
| `/models` | GET | Available models info |

---

## 📊 Datasets

The project supports training with:

- **UCI Phishing Dataset** — 11,000+ labeled samples with 30 features
- **PhishTank** — Real-time phishing URL feeds
- **Custom Synthetic Data** — Generated via `synthesize_data.py`

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask |
| ML Models | scikit-learn, XGBoost, TensorFlow |
| Explainability | LIME, SHAP |
| Feature Extraction | BeautifulSoup, python-whois, dnspython |
| Frontend | HTML5, CSS3, JavaScript |
| Visualization | Matplotlib, Seaborn, Plotly |

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [LIME](https://github.com/marcotcr/lime) & [SHAP](https://github.com/slundberg/shap) for Explainable AI
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for benchmark datasets
- [PhishTank](https://phishtank.org/) for phishing URL data
- [scikit-learn](https://scikit-learn.org/) & [XGBoost](https://xgboost.readthedocs.io/) for ML frameworks
