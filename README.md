# Breast Cancer Classification using k-NN

This project implements a **k-Nearest Neighbors (k-NN)** classification pipeline to predict **breast cancer diagnosis (Benign vs Malignant)** using the **Wisconsin Diagnostic Breast Cancer dataset** from the UCI Machine Learning Repository.

The workflow follows a structured **data science pipeline**, including data ingestion, preprocessing, visualization, model training, evaluation, and cross-validation.

## Dataset

- **Source:** UCI Machine Learning Repository  
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
- **UCI Repository ID:** 17  
- **Instances:** 569  
- **Attributes:** 30 real-valued features  
- **Target Classes:**  
  - `B` → Benign  
  - `M` → Malignant  

## Libraries Required

```bash
pip install numpy pandas matplotlib scikit-learn ucimlrepo
```

## Project Pipeline Overview

### 1. Data Ingestion
- The Breast Cancer Wisconsin (Diagnostic) dataset is fetched directly from the **UCI Machine Learning Repository**.
- Non-informative identifier attributes are removed to prevent bias in model learning.

### 2. Exploratory Data Analysis (EDA)
- Boxplots are generated for all 30 numerical attributes to visualize distributions and detect outliers.
- Summary statistics (Q1, median, Q3, minimum, maximum) are computed for each feature.
- Individual plots are used to avoid scale dominance by high-range attributes.

### 3. Data Cleaning and Preprocessing
- Missing values are checked across all features.
- Mean imputation is applied **class-wise** to preserve the statistical properties of each diagnosis class.
- All features are normalized using **Min–Max scaling** to ensure fair distance calculations for k-NN.

### 4. Label Transformation
- Diagnosis labels are encoded into numerical form:
  - Benign → 0  
  - Malignant → 1  

### 5. Dataset Splitting
- The dataset is split into **training (80%)** and **testing (20%)** sets.
- Stratified sampling ensures class distributions are preserved.
- Dataset shapes and label proportions are verified for consistency.

### 6. Model Training (Single Train–Test Split)
- Multiple k-NN models are trained for values of `k = 1` to `10`.
- Each model is evaluated using:
  - F1-score
  - Accuracy
  - Precision
- Performance metrics are stored and compared across different values of `k`.

### 7. Performance Visualization
- Line plots visualize the relationship between `k` and model performance.
- The optimal value of `k` is identified based on metric trends.

### 8. K-Fold Cross-Validation
- Stratified K-Fold Cross-Validation is applied to improve evaluation reliability.
- Each model is trained and tested across multiple folds.
- Mean and standard deviation of evaluation metrics are computed for robustness analysis.

### 9. Model Selection
- The optimal k-NN model is selected based on:
  - Highest mean F1-score
  - Lowest performance variance across folds

## Conclusion
- Both 5-fold and 10-fold cross-validation yield high and consistent performance
- The 5-fold model with k = 8 is preferred due to:
- - Higher F1-score
  - Lower standard deviation
  - Better robustness and reliability
- Given the sensitive nature of medical classification tasks, statistical evaluation must be complemented with domain expertise and human interpretation before real-world deployment.

## Key Takeaways
- k-NN performs exceptionally well on normalized biomedical data
- Stratification is critical for reliable evaluation
- Cross-validation provides a more robust estimate than a single train-test split
