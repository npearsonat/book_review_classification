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

## Results

![Visual](visualizations/ML_classifier_accuracy.png)
**Machine learning classification accuracy results for review helpfulness classification**

For the machine learning models, random forest achieved the highest accuracy score as well as highest F1, recall and precision. It used the maximum range of estimators, 200, and max_depth=None. 

![Visual](visualizations/ML_classifier_feature_importance.png)

**Feature importances utilized in random forest model. Engineered features are in red while TF-IDF features are in blue**

![Visual](visualizations/ML_classifier_importance_chart.png)

**Most important features and the values most associated with helpful and unhelpful reviews**

When looking at the feature importances of the random forest model, the pre-engineered features were much more prominent individually, although vastly outnumbered. Polarity score was the most important engineered feature, with reviews having higher polarity score being sorted as helpful. You can see a variety of features are more associated with helpful reviews, such as having a longer length, higher word count and lower question count. This gives us some simpler insight into what readers may be looking for in a review. 

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
