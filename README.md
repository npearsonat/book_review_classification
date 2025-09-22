# Review Helpfulness Classifier

A neural network-based text classifier that predicts whether product reviews are helpful or not. Uses TF-IDF vectorization combined with engineered features and a multi-layer perceptron to achieve high accuracy on review classification tasks.

## Features

- **Neural Network Architecture**: Multi-layer perceptron with customizable layers
- **Advanced Feature Engineering**: Combines text vectorization with sentiment signals, writing style indicators, and structural features
- **Easy to Use**: Simple API for training and prediction
- **No Heavy Dependencies**: Built with scikit-learn, avoiding transformer library issues

## Quick Start

```python
import pandas as pd
from text_classifier import build_classifier

# Load your data (assuming 'is_helpful' target column)
df = pd.read_csv('your_reviews.csv')

# Train the classifier
classifier = build_classifier(df)

# Make predictions
result = classifier.check_review("This product works great and arrived quickly!")
print(result)
# Output: {'prediction': 'Helpful', 'confidence': 0.847, ...}
```

## Installation

```bash
pip install pandas numpy scikit-learn scipy
```

## Usage

### Training a Model

```python
from text_classifier import TextAnalyzer

# Initialize with custom parameters
classifier = TextAnalyzer(
    layers=(512, 256, 128),    # Neural network architecture
    vocab_size=15000           # TF-IDF vocabulary size
)

# Train the model
classifier.train(
    df, 
    target='is_helpful',       # Target column name
    training_rounds=200,       # Number of epochs
    validation_split=0.2       # Test set percentage
)

# Analyze performance
classifier.analyze_performance()
```

### Making Predictions

```python
# Single review prediction
result = classifier.check_review("Great product, highly recommend!")

# Batch predictions
predictions, probabilities = classifier.predict([
    "Excellent quality and fast shipping",
    "Terrible product, waste of money",
    "Average product, nothing special"
])
```

## Model Architecture

The classifier combines multiple feature types:

- **Text Features**: TF-IDF vectors with 1-2 word n-grams
- **Sentiment Signals**: Positive/negative word ratios
- **Structural Features**: Word count, sentence length, writing style
- **Neural Network**: Multi-layer perceptron with ReLU activation

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layers` | `(384, 192, 96)` | Hidden layer sizes |
| `vocab_size` | `12000` | TF-IDF vocabulary size |
| `training_rounds` | `50` | Maximum training epochs |
| `validation_split` | `0.2` | Test data percentage |

## Data Format

Your dataset should be a pandas DataFrame with:
- Text columns: `review/text`, `review/summary`, or `combined_text`
- Target column: Binary values (0/1 or True/False) indicating helpfulness
- Optional: `review/helpfulness` for additional features

Example:
```csv
review/text,review/summary,is_helpful
"Great product, works as expected","Excellent",1
"Broke after one day","Poor quality",0
```

## Performance

The model typically achieves:
- **Accuracy**: 75-85% on review helpfulness prediction
- **Training Time**: 2-5 minutes on typical datasets (10k-50k reviews)
- **Features**: Automatically extracts 12k+ text features + 9 numerical features

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with scikit-learn and pandas
- Inspired by modern NLP techniques adapted for traditional ML frameworks
- Designed for reliability and ease of use in production environments
