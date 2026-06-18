# Project Brief: Game User LTV Prediction and UA Value Analysis

## 1. Original Project: What TransForeCaster Does

The original repository is based on TransForeCaster, a deep learning framework for user representation learning and behavior prediction. It combines user metadata, portrait features, and behavior time-series features to predict future user outcomes such as purchase amount and churn probability.

Its main modeling idea is:

- Use in-category integration to compress behavior and portrait time-series features.
- Use cross-category integration to combine different feature groups with user metadata.
- Train a multi-task neural network for purchase prediction and churn prediction.

This design is closer to a research-oriented deep learning project than a beginner-friendly business analytics project.

## 2. Why We Will Not Fully Reproduce the Deep Learning Version

For this repository's new goal, fully reproducing the original deep learning model is not the best choice.

Reasons:

- The original model contains custom TensorFlow layers, Transformer blocks, LSTM/VAE-style pretraining, and multi-task loss design, which makes the project hard to explain in an entry-level interview.
- The current sample data is synthetic and does not justify a complex neural architecture.
- A complete deep learning reproduction would spend too much effort on model internals instead of business understanding, feature engineering, model evaluation, and marketing analysis.
- For new graduate data analyst, data scientist, or business analyst roles, a clear end-to-end LTV project is more valuable than a complicated model that is hard to interpret.

Therefore, the transformed project will keep the useful business idea from TransForeCaster but simplify the technical route.

## 3. New Project Goal

This repository will be transformed into a portfolio project named:

**Game User LTV Prediction and User Acquisition Value Analysis**

The goal is to build an end-to-end project that simulates a real game business scenario:

- Use early user behavior data to predict future LTV, such as D7 and D30 revenue.
- Analyze user quality by acquisition channel, country, device, and early engagement behavior.
- Build interpretable machine learning models for LTV prediction.
- Evaluate model performance with standard regression metrics.
- Combine predicted LTV with acquisition cost to analyze ROI and ROAS.
- Provide actionable marketing recommendations, such as which channels or user segments deserve more budget.

The final project should be understandable to recruiters and interviewers, and it should be easy for a new graduate to explain from business problem to data, features, model, evaluation, and decision analysis.

## 4. Job Description Capability Mapping

This project is designed to demonstrate the following job-relevant skills.

### LTV and Revenue Prediction

- Define user LTV in a game business context.
- Predict future revenue such as D7 LTV and D30 LTV.
- Understand skewed revenue distribution and high-value payer behavior.

### D7 and D30 Analysis

- Use early lifecycle behavior to estimate later user value.
- Compare D1, D3, D7, and D30 indicators.
- Explain why early user signals are useful for marketing decisions.

### User Behavior Features

- Build features from login activity, session behavior, level progress, payment behavior, ad engagement, and retention signals.
- Compare user groups by channel, country, device, and payment status.

### Feature Engineering

- Create RFM-style features, early payment features, engagement features, retention flags, and channel-level aggregates.
- Handle categorical variables and numerical variables in a clean modeling pipeline.

### Model Evaluation

- Train baseline and machine learning models for LTV prediction.
- Evaluate models with MAE, RMSE, R2, and prediction error analysis.
- Compare model performance against simple baselines.

### ROI and ROAS

- Combine predicted LTV with CPI or user acquisition cost.
- Calculate ROI and ROAS by channel and segment.
- Identify profitable and unprofitable acquisition sources.

### User Acquisition Value Analysis

- Translate model outputs into business decisions.
- Recommend budget adjustment, channel optimization, and high-value user targeting strategies.
- Present findings through tables, charts, and concise business conclusions.

## 5. Planned Final Directory Structure

The final project is planned to use the following structure:

```text
data/
  raw/                  # Raw or generated sample data
  processed/            # Cleaned modeling datasets

notebooks/
  01_eda.ipynb          # Exploratory data analysis
  02_feature_engineering.ipynb
  03_model_training.ipynb
  04_roi_analysis.ipynb

src/
  data_generation.py    # Generate realistic game user sample data
  preprocessing.py      # Data cleaning and feature preparation
  features.py           # Feature engineering logic
  train_model.py        # Model training workflow
  evaluate.py           # Model metrics and evaluation helpers
  roi_analysis.py       # ROI/ROAS calculation and segment analysis

outputs/
  figures/              # Charts for README and analysis reports
  reports/              # Summary tables and business conclusions

models/
  ltv_model.pkl         # Saved trained model artifacts

README.md
PROJECT_BRIEF.md
requirements.txt
```

This is a target structure. In the current stage, it is only a plan. We will not create unnecessary empty folders or complex code yet.

## 6. Final Deliverables

The final transformed project should include:

- A clear README explaining the business background, project workflow, how to run the project, and key results.
- A generated or sample game user dataset with acquisition channel, user behavior, payment, retention, and cost fields.
- EDA charts showing LTV distribution, payer behavior, channel quality, and retention/revenue relationships.
- A feature engineering pipeline for early user behavior and monetization features.
- Baseline and machine learning LTV prediction models.
- Model evaluation results using MAE, RMSE, R2, and segment-level error analysis.
- ROI/ROAS analysis by channel, country, and user segment.
- Business conclusions suitable for resume and interview discussion.

## 7. Current Stage

Current stage: **Stage 1 - Project Repositioning and Directory Planning**

This stage only defines the transformation direction and documentation structure. It does not modify the original deep learning model, does not train any model, and does not introduce a large amount of new code.
