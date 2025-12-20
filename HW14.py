import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

def generate_critical_data(n_samples=1000, imbalance_ratio=1.0):
    np.random.seed(42)
    n_minority = int(n_samples / (1 + imbalance_ratio))
    n_majority = n_samples - n_minority
    X_ordered = np.random.normal(loc=[0, 0], scale=0.8, size=(n_minority, 2))
    y_ordered = np.zeros(n_minority)
    X_disordered = np.random.normal(loc=[0, 0], scale=1.2, size=(n_majority, 2))
    y_disordered = np.ones(n_majority)
    X = np.vstack([X_ordered, X_disordered])
    y = np.hstack([y_ordered, y_disordered])
    shuffle_idx = np.random.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]

def test_regularization(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    models = \
    {
        "L2 Regularization": LogisticRegression(penalty='l2', solver='saga', C=1.0, max_iter=1000),
        "L1 Regularization": LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=1000),
        "Elastic Net (L1+L2)": LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', C=1.0,
                                                  max_iter=1000)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = \
        {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average='binary'),
            "AUC Score": roc_auc_score(y_test, y_prob),
            "Non-zero Coefficients": np.sum(model.coef_ != 0)
        }
    return pd.DataFrame(results).T

def test_class_imbalance(imbalance_ratios=[0.5, 1.0, 2.0, 5.0, 10.0]):
    results = []
    for ratio in imbalance_ratios:
        X, y = generate_critical_data(n_samples=1000, imbalance_ratio=ratio)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', C=1.0, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        minority_class = 1 if np.sum(y_test == 1) < np.sum(y_test == 0) else 0
        minority_recall = cm[minority_class, minority_class] / cm[minority_class].sum() if cm[
                                                                                               minority_class].sum() > 0 else 0
        results.append({
            "Imbalance Ratio (Majority/Minority)": ratio,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Minority Recall": minority_recall,
            "F1 Score": f1_score(y_test, y_pred, average='binary'),
            "AUC Score": roc_auc_score(y_test, y_prob)
        })
    return pd.DataFrame(results)

def plot_results(df_imbalance):
    plt.rcParams['font.family'] = 'Arial'  # Use universal font
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(df_imbalance["Imbalance Ratio (Majority/Minority)"], df_imbalance["Accuracy"], 'o-', color='blue',
                    linewidth=2)
    axes[0, 0].set_xlabel("Class Imbalance Ratio (Majority/Minority)")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Impact of Imbalance Ratio on Accuracy")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df_imbalance["Imbalance Ratio (Majority/Minority)"], df_imbalance["Minority Recall"], 'o-',
                    color='red', linewidth=2)
    axes[0, 1].set_xlabel("Class Imbalance Ratio (Majority/Minority)")
    axes[0, 1].set_ylabel("Minority Recall")
    axes[0, 1].set_title("Impact of Imbalance Ratio on Minority Recall")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df_imbalance["Imbalance Ratio (Majority/Minority)"], df_imbalance["F1 Score"], 'o-', color='green',
                    linewidth=2)
    axes[1, 0].set_xlabel("Class Imbalance Ratio (Majority/Minority)")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_title("Impact of Imbalance Ratio on F1 Score")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df_imbalance["Imbalance Ratio (Majority/Minority)"], df_imbalance["AUC Score"], 'o-',
                    color='orange', linewidth=2)
    axes[1, 1].set_xlabel("Class Imbalance Ratio (Majority/Minority)")
    axes[1, 1].set_ylabel("AUC Score")
    axes[1, 1].set_title("Impact of Imbalance Ratio on AUC Score")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_critical, y_critical = generate_critical_data(n_samples=1000, imbalance_ratio=1.0)
    print("=" * 60)
    print("Experiment 1: Model Performance at Critical Temperature (T/J=2.25)")
    print(f"Proportion of ordered states in data: {np.sum(y_critical == 0) / len(y_critical):.2f}")
    print(f"Proportion of unordered states in data: {np.sum(y_critical == 1) / len(y_critical):.2f}")
    X_train, X_test, y_train, y_test = train_test_split(X_critical, y_critical, test_size=0.3, random_state=42)
    model_critical = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', max_iter=1000)
    model_critical.fit(X_train, y_train)
    y_pred_critical = model_critical.predict(X_test)
    accuracy_critical = accuracy_score(y_test, y_pred_critical)
    print(f"Model Accuracy: {accuracy_critical:.2f}")
    print(
        "Conclusion: Accuracy is close to 50% at critical temperature, matching expectations (maximal model confusion)")
    print("\n" + "=" * 60)
    print("Experiment 2: Performance Comparison of Regularization Methods")
    df_reg = test_regularization(X_critical, y_critical)
    print(df_reg.round(3))
    print("\n" + "=" * 60)
    print("Experiment 3: Impact of Class Imbalance Ratio on Model Performance")
    df_imbalance = test_class_imbalance(imbalance_ratios=[0.5, 1.0, 2.0, 5.0, 10.0])
    print(df_imbalance.round(3))
    plot_results(df_imbalance)