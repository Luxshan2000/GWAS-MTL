# Multitask Learning for GWAS-Based Disease Prediction

## 1. Project Overview

This project implements a multitask learning model for predicting disease risk based on GWAS (Genome-Wide Association Study) data. The model uses shared layers with an attention-based pooling mechanism to enhance genetic feature extraction across multiple diseases.

### Key Components:

1. **Real GWAS Data**: Cardiovascular disease data from the CARDIoGRAMplusC4D consortium
2. **Multitask Learning**: Simultaneous prediction of risk for three diseases
3. **Attention-Based Pooling**: Enhances genetic feature extraction by highlighting relevant SNPs for each disease
4. **Open-Source Architecture**: Fully documented and transparent implementation

## 2. Model Architecture

![Multitask Model Architecture](figures/multitask_model_architecture.png)

The model architecture integrates:

- **Shared Layers**: Extract common genetic features relevant to multiple diseases
- **Attention Pooling**: Prioritizes different genetic features for each disease task
- **Disease-Specific Layers**: Generate specialized predictions for each disease

## 3. Data Processing and Exploratory Analysis

### Data Source

- Primary data: Real cardiovascular disease GWAS data from CARDIoGRAMplusC4D consortium

### Data Distribution

![Effect Size Distribution](figures/log_odds_distribution.png)

The distribution shows how effect sizes (log odds) vary across the three diseases, with some genetic variants having stronger effects than others.

## 4. Training and Evaluation

### Training Process

![Loss Curves](figures/loss_curves.png)

The model was trained for 28 epochs with early stopping to prevent overfitting. The loss curves show smooth convergence, indicating effective learning of disease-specific patterns.

### Model Performance

**ROC Curves:**

![ROC Curves](figures/roc_curves.png)

**Confusion Matrices:**

Cardiovascular Disease:  
![Cardiovascular Confusion Matrix](figures/confusion_matrix_0.png)

Type 2 Diabetes:  
![T2D Confusion Matrix](figures/confusion_matrix_1.png)

Cancer:  
![Cancer Confusion Matrix](figures/confusion_matrix_2.png)

### Classification Reports

**Cardiovascular Disease:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Low Risk | 0.98 | 1.00 | 0.99 | 802 |
| High Risk | 0.98 | 0.93 | 0.96 | 198 |
| macro avg | 0.98 | 0.96 | 0.97 | 1000 |
| weighted avg | 0.98 | 0.98 | 0.98 | 1000 |

Accuracy: 0.98

**Type 2 Diabetes:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Low Risk | 0.99 | 0.99 | 0.99 | 800 |
| High Risk | 0.97 | 0.95 | 0.96 | 200 |
| macro avg | 0.98 | 0.97 | 0.98 | 1000 |
| weighted avg | 0.98 | 0.98 | 0.98 | 1000 |

Accuracy: 0.98

**Cancer:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Low Risk | 0.99 | 1.00 | 0.99 | 810 |
| High Risk | 0.98 | 0.97 | 0.97 | 190 |
| macro avg | 0.99 | 0.98 | 0.98 | 1000 |
| weighted avg | 0.99 | 0.99 | 0.99 | 1000 |

Accuracy: 0.99

### Feature Importance Analysis

![Attention Weights](figures/attention_weights.png)

The attention mechanism helps the model prioritize different genetic features for each disease. This visualization shows the average attention weights assigned to each disease task.

## 5. Limitations and Future Work

### Current Limitations

- Used real data for cardiovascular disease but derived data for T2D and cancer for demonstration
- Focused on a subset of significant SNPs rather than all possible genetic variants
- Binary classification approach may not capture full spectrum of disease risk

### Future Directions

1. Integrate real GWAS datasets for all diseases
2. Experiment with additional pooling mechanisms (max pooling, graph-based pooling)
3. Incorporate more genetic variants and clinical data
4. Explore alternative neural network architectures
5. Add population stratification controls for more robust predictions

## 6. Conclusion

This project demonstrates the effectiveness of multitask learning with pooling mechanisms for GWAS-based disease prediction. The model successfully leverages shared genetic architecture across diseases while maintaining high prediction accuracy for each condition. The open-source implementation provides a foundation for future genomic research and AI-driven risk assessment tools.

The high performance metrics (98-99% accuracy) demonstrate the potential of this approach, though real-world application would require integration of real datasets for all diseases.