import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.inspection import permutation_importance

def cross_val_pr_curve(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    mean_recall = np.linspace(0, 1, 100)
    mean_precision_sum = np.zeros_like(mean_recall)
    auc_scores = []

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_scores = model.predict_proba(X_val_fold)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val_fold, y_scores)
        auc_score = auc(recall, precision)
        auc_scores.append(auc_score)

        interpolated_precision = np.interp(mean_recall, np.flip(recall), np.flip(precision))
        mean_precision_sum += interpolated_precision

    mean_precision = mean_precision_sum / cv

    plt.plot(mean_recall, mean_precision, label=f"Mean PR Curve (AUC={np.mean(auc_scores):.2f})", color='b')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Cross-Validated Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Mean Precision-Recall AUC: {np.mean(auc_scores):.4f}")

def tune_hyperparameters(X_train, y_train, param_grid, model, metric):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=metric)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

def evaluate_models(model, X_data, y_data, metric):
    cv_scores = cross_val_score(model, X_data, y_data, cv=3, scoring=metric)
    print("Cross-Validation Precision Scores:", cv_scores)
    print("Mean Precision Score:", np.mean(cv_scores))

    y_pred = cross_val_predict(model, X_data, y_data, cv=3, method='predict')
    print("Validation Set Classification Report:")
    print(classification_report(y_data, y_pred))
    cross_val_pr_curve(model, X_data, y_data)

def feature_importance(model, X_data, y_data, metric):
    perm_importance = permutation_importance(model, X_data, y_data, scoring=metric, n_repeats=10, random_state=42)
    feature_importance_df = pd.DataFrame({'Feature': X_data.columns, 'Importance': perm_importance.importances_mean}, index=X_data.columns)
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    ax = feature_importance_df.plot(kind='bar', figsize=(10, 5), title='Feature Importance')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    plt.show()

def cross_val_approval_matrix(model, X_data, y_data, bad_flag, cv=3):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    if isinstance(X_data, pd.DataFrame):
        X_data = X_data.reset_index(drop=True)
    if isinstance(y_data, pd.Series):
        y_data = y_data.reset_index(drop=True)
    if isinstance(bad_flag, pd.Series):
        bad_flag = bad_flag.reset_index(drop=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]
        bad_flag_val = bad_flag.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        eval_df = pd.DataFrame({
            'prediction': y_pred,
            'true_approval': y_val,
            'bad_flag': bad_flag_val
        })

        eval_df = eval_df[eval_df['bad_flag'].notna()]
        eval_df['approval_result'] = eval_df.apply(lambda row:
            'Approved Good' if row['prediction'] == 1 and row['bad_flag'] == 1 else
            'Approved Bad' if row['prediction'] == 1 and row['bad_flag'] == 0 else
            'Rejected Good' if row['prediction'] == 0 and row['bad_flag'] == 1 else
            'Rejected Bad', axis=1)

        print(f"\nFold {fold + 1} Approval Matrix:")
        print(eval_df['approval_result'].value_counts())
        results.append(eval_df)

    final_df = pd.concat(results)
    print("\n✅ Overall Approval Matrix Across Folds:")
    print(final_df['approval_result'].value_counts())
    return final_df


    
    
