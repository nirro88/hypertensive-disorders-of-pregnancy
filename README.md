[![Project Logo](https://img.shields.io/badge/Clinical%20Screening-Prioritization-blue)](https://github.com/yourrepo)

# High-Risk Women Screening & Prioritization

This project provides a robust pipeline for clinical risk screening and prioritization, focusing on high-risk women. It includes data preprocessing, feature engineering, topic modeling (BERTopic), and two-track modeling (Logistic Regression & XGBoost) with calibration and budget-constrained evaluation.

## Main Components

- **HWPreprocessing.py**: Contains the `Preprocess` class for:
  - Data loading
  - Leakage column identification
  - Stratified train/validation/test splits
  - Diagnosis feature engineering (collapse columns)
  - Missing value filtering
  - Clinical note cleaning (Hebrew, regex)
  - Topic modeling (BERTopic)
  - Adding topic probabilities to tabular data
- **HWEDA.ipynb**: Exploratory Data Analysis (EDA) notebook
  - Data overview, missingness, feature distributions
  - Key takeaways at each step
- **HWModeling.ipynb**: Modeling notebook
  - Two-track models: Logistic Regression & XGBoost
  - Budget-constrained evaluation (10%, 20% referral)
  - Calibration (Platt, Isotonic)
  - Key findings and practical implications
- **requirements.txt**: All dependencies for reproducibility
- **barchart.html**, **highrisk_barchart.html**: Plotly visualizations (exported charts)

## Workflow Overview

1. **Preprocessing**
    - Use `HWPreprocessing.py` to clean and prepare the data.
    - Remove leakage columns, collapse diagnosis features, filter high-missing columns, clean clinical notes, and generate topic features.
2. **EDA**
    - Run `HWEDA.ipynb` to explore data, visualize distributions, and summarize key findings.
3. **Modeling**
    - Run `HWModeling.ipynb` for:
      - Logistic Regression (with imputation/standardization)
      - XGBoost (handles NaNs natively)
      - Evaluation: AUC, AUPR, sensitivity/PPV at referral budgets
      - Calibration: Platt scaling, Isotonic regression
      - Calibration-aware thresholding
4. **Visualization**
    - Review exported Plotly charts in `barchart.html` and `highrisk_barchart.html`.

## Key Takeaways (from EDA & Modeling)

- **Discrimination**: XGBoost slightly stronger on validation, Logistic better on test. Both below target AUC band (0.75–0.85).
- **Budget-Constrained Evaluation**: At 10% referral, XGBoost catches more positives; at 20%, both models similar but with lower precision.
- **Calibration**: Calibration (especially Isotonic) greatly improves probability reliability. Calibrated scores are interpretable for clinical decision-making.
- **Practical Implications**: Use calibrated probabilities for deployment; thresholds should be set post-calibration.
- **Next Steps**: Feature refinement, improved imbalance handling, calibration-aware thresholding, and clinical framing (rule-in vs rule-out).

## Updates to Workflow

### Text Handling
- Preprocessed clinical notes by cleaning, tokenizing, and removing stop words.
- Extracted numerical features from text using TF-IDF and word embeddings.
- Analyzed word frequencies and created topic models for better insights.

### SHAP Analysis
- Computed SHAP values to interpret model predictions and understand feature contributions.
- Visualized SHAP results using summary plots, force plots, and dependence plots.
- Identified key features driving predictions and validated their importance in the context of clinical decision-making.

These updates enhance the interpretability and robustness of the pipeline, ensuring better insights for clinical prioritization.

## Usage

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run preprocessing:
    ```python
    from HWPreprocessing import Preprocess
    pre = Preprocess()
    # ...see HWPreprocessing.py for method usage...
    ```
3. Run EDA and modeling notebooks in Jupyter:
    ```bash
    jupyter notebook HWEDA.ipynb
    jupyter notebook HWModeling.ipynb
    ```

## Dependencies

See `requirements.txt` for all required packages:
- numpy, pandas, matplotlib, seaborn
- scikit-learn, xgboost, shap
- bertopic, sentence-transformers, umap-learn, hdbscan
- plotly

## File References

- `HWPreprocessing.py`: Preprocessing and feature engineering
- `HWEDA.ipynb`: Exploratory data analysis
- `HWModeling.ipynb`: Modeling, evaluation, calibration
- `requirements.txt`: Dependencies
- `barchart.html`, `highrisk_barchart.html`: Visualizations

## Contact

For questions, please contact [your.email@domain.com].
