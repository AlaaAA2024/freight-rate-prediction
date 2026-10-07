# Freight Load Rate Prediction & Dynamic Scheduling

An end-to-end Machine Learning solution for predicting spot-market freight load rates (`posted_rate`) using 5-Fold Cross-Validated LightGBM models with dynamic temporal feature engineering.

---

## 📋 Detailed File & Directory Breakdown

```text
.
├── data/                                # Directory containing raw and template datasets
│   ├── december-chart-inputs.csv        # Inputs for the 31-day fixed route control scenario (Lexington to Fort Wayne)
│   ├── train-test.csv                   # Historical training dataset (48,000 loads) containing target posted_rate
│   ├── validation.csv                   # Validation dataset (12,000 unseen loads) for prediction evaluation
│   └── validation-predictions-template.csv # Official output template detailing required schema and load_id formatting
├── scorer_results/                      # Output directory created by score.py
│   └── candidate_december.png           # Rendered 31-day December rate seasonality prediction chart
├── .gitignore                           # Git configuration file specifying untracked local and temporary files
├── README.md                            # Comprehensive project guide, setup instructions, and execution workflow
├── Technical_Report.pdf                 # Final compiled LaTeX technical report covering validation methodology & analysis
├── december_predictions.csv             # Model output containing 31 predicted load rates for December
├── validation_predictions.csv           # Model output containing 12,000 predicted load rates for the validation set
├── preprocess.py                        # Preprocessing module for data cleaning, coordinate mapping, and feature extraction
├── train.py                             # Main training pipeline implementing 5-Fold CV and LightGBM ensembling
├── score.py                             # Evaluation script verifying CSV schema integrity and generating output charts
└── requirements.txt                     # List of Python library dependencies required to execute the pipeline
```

---

## 🛠️ Installation & Setup

### Prerequisites

- Anaconda or Miniconda installed.
- Git installed.

### Step 1: Open Terminal / Anaconda Prompt

- **Windows:** Open Anaconda Prompt from the Start menu.
- **macOS / Linux:** Open your standard Terminal.

### Step 2: Clone the Repository

Clone the project repository to your local computer and navigate into the project directory:

```bash
git clone https://github.com/AlaaAA2024/freight-rate-prediction.git
cd freight-rate-prediction
```

### Step 3: Create an Anaconda Environment

Create a clean Python 3.10 virtual environment named `freight_env`:

```bash
conda create -n freight_env python=3.10 -y
```

### Step 4: Activate the Anaconda Environment

Activate the newly created environment:

```bash
conda activate freight_env
```

> **Note:** Upon activation, your terminal prompt line will update from `(base)` to `(freight_env)`.

### Step 5: Install Project Dependencies

Install all required Python libraries specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Pipeline

Execute the scripts sequentially from your terminal within the active `(freight_env)` environment.

### Step 1: Run Model Training & Prediction Generation

Execute `train.py` to preprocess data, perform 5-fold cross-validation training, and generate prediction files:

```bash
python train.py
```

**Expected Terminal Output:**

- Loads datasets from `data/` directory.
- Completes 5-Fold CV training with LightGBM.
- Saves `validation_predictions.csv` (12,000 rows).
- Saves `december_predictions.csv` (31 rows).

### Step 2: Validate Predictions & Render Evaluation Chart

Execute `score.py` to verify output CSV formats and build the December route rate chart:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

**Expected Terminal Output:**

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
```

> Final validation metrics are calculated by Spotter after submission.

**Generated Visual:**

```text
scorer_results/candidate_december.png
```

---

## 📝 Notes

- Ensure you are inside the `(freight_env)` environment before running any scripts.
- The `data/` directory must contain the required datasets before running `train.py`.
- All output files will be generated in the project root or `scorer_results/` directory.
