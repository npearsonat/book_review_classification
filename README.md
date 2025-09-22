# Review Helpfulness Classifier

A demonstration of machine learning and neural network models that predict whether product reviews will be helpful to other customers. Achieved 73% accuracy with ML and 96% with neural network.</br>

**Dataset Source:** Kaggle Amazon Book Reviews: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews

## Key Findings

**High-impact features discovered:**
- **Review length matters most**: Reviews with 50-200 words are 3x more likely to be helpful
- **Positive sentiment wins**: Reviews mixing positive and negative words outperform purely positive ones

**Model Performance:**
- **96% accuracy** on 13,000+ review dataset
- **0.995 AUC score** with good precision-recall balance

## Business Impact

This classifier could help e-commerce platforms:
- **Prioritize helpful reviews** in search results and product pages
- **Reduce customer decision time** by surfacing quality feedback first  
- **Improve user experience** by filtering low-quality reviews automatically

## Approach

**Data Cleaning:**
- Selected rows with greater than 10 helpfullness ratings.
- Took random sample of 100k remaining rows for ease of use.
- Cleaned up columns and combined metadata and review datasets.

**Feature Engineering Innovation:**
- Created helpfull/not helpfull feature based on helpfullness scores of greater than 0.8. 
- Combined TF-IDF text vectors with 11 custom linguistic features
- Engineered sentiment ratio features that captured review nuance better than basic sentiment scores
- Created writing style indicators (caps ratio, punctuation patterns) that proved surprisingly predictive

**Model Architecture:**
- Employed logistic regression, random forest, XGBoost, and LightGBM mashine learning models.
- Multi-layer neural network (384→192→96 neurons) with ReLU activation
- Handled class imbalance through stratified sampling
- Used early stopping to prevent overfitting

## Sample Results


## Quick Demo

```python
from text_classifier import build_classifier
import pandas as pd

# Train on your review data
df = pd.read_csv('reviews.csv')
classifier = build_classifier(df)

# Get predictions with confidence scores
result = classifier.check_review("Your review text here")
print(f"Prediction: {result['prediction']} ({result['confidence']:.2f})")
```

## Installation

```bash
git clone https://github.com/yourusername/review-helpfulness-classifier
cd review-helpfulness-classifier
pip install -r requirements.txt
```

## Dataset Requirements

Works with any review dataset containing:
- Review text (`review/text` or `review/summary` columns)
- Binary helpfulness labels (`is_helpful`: 1 for helpful, 0 for not helpful)

---

*Built with Python, scikit-learn, and pandas. No heavy ML frameworks required.*
