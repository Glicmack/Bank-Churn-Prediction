# Bank-Churn-Prediction

# 🚀 Bank Customer Churn Prediction Pipeline

A production-ready **end-to-end machine learning pipeline** for predicting bank customer churn using the popular Kaggle Churn Modelling dataset. Features automated preprocessing, hyperparameter optimization, model persistence, and interactive prediction interface.

Features

- ✅ **Full sklearn Pipeline**: Preprocessing + Logistic Regression in one object
- ✅ **Automatic Hyperparameter Tuning**: GridSearchCV finds optimal `C`, `solver`
- ✅ **Class Imbalance Handling**: `class_weight="balanced"` for better churn recall
- ✅ **Model Persistence**: Save/load with `joblib` for production use
- ✅ **Interactive Prediction**: CLI interface for real-time customer predictions
- ✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, ROC/PR curves, Confusion Matrix
- ✅ **Production Optimizations**: `n_jobs=-1`, dense arrays, stratified splits

Results (Test Set)
<img width="398" height="260" alt="image" src="https://github.com/user-attachments/assets/285eecef-0cc8-45e7-b0a2-cc6d117e3222" />

python3.10 -m venv venv
source venv/bin/activate # Linux/Mac

venv\Scripts\activate # Windows
Install dependencies
pip install -r requirements.txt

### 2. Download Dataset  https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

python train_churn_model.py

*Outputs: `churn_model.joblib` + performance metrics + plots*

### 4. Make Predictions

python predict_churn.py

*Interactive CLI: Enter customer details → Get churn probability*

## 📁 Project Structure
bank-churn-prediction/ 

├── train_churn_model.py # Full training pipeline + evaluation

├── predict_churn.py # Interactive prediction interface

├── churn_model.joblib # Saved trained model (auto-generated)

├── dataset/ # Kaggle Churn_Modelling.csv

├── requirements.txt # Python dependencies

├── plots/ # Confusion matrix, ROC/PR curves

└── README.md

## 🔧 Usage Examples

### Training Output
Best parameters: {'classifier__C': 0.01, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Accuracy: 0.711 | Precision: 0.386 | Recall: 0.708 | F1 Score: 0.499
Model saved to churn_model.joblib

### Prediction Example

Enter customer details:
CreditScore: 650
Geography: France
Gender: Male
Age: 42
...
Prediction: This customer is LIKELY to churn.
Churn probability: 0.623

## 🎛️ Customization

### Adjust Business Threshold

threshold = 0.6 # Only flag as churn if prob >= 60%
y_pred_custom = (y_proba >= threshold).astype(int)

### Try Different Models

from sklearn.ensemble import RandomForestClassifier

Replace LogisticRegression() with RandomForestClassifier(n_jobs=-1)

## 🧪 Development

Activate environment
source venv/bin/activate

Install dev tools
pip install black flake8 pytest

Lint
flake8 *.py

Format
black *.py

Tests (add your own)
pytest


## 📈 Expected Improvements Over Baseline

| Metric | Baseline (no tuning) | Optimized Pipeline |
|--------|---------------------|-------------------|
| Recall | ~40-50% | **70.8%** ↑ |
| F1 Score | ~0.35-0.40 | **0.499** ↑ |
| Training Speed | Single core | **Multi-core** (`n_jobs=-1`) |

## 📚 Key Learning Concepts

- **sklearn Pipeline**: Chain preprocessing + modeling
- **ColumnTransformer**: Handle numeric/categorical features separately
- **GridSearchCV**: Automated hyperparameter optimization
- **Class Imbalance**: `class_weight="balanced"`
- **Model Persistence**: `joblib.dump/load`
- **Business Thresholding**: Custom decision boundaries

## 🤝 Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Kaggle Churn Modelling Dataset](https://www.kaggle.com/datasets/aakash50897/churn-modellingcsv)
- [scikit-learn Documentation](https://scikit-learn.org/)
- Original code inspired by standard ML best practices

---
⭐ **Star this repo if you found it helpful!** ⭐
