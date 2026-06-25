import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, recall_score, precision_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib


print("Loading data")
df = pd.read_csv('221125_MDD_DEGs_raw_data.csv', index_col=0)
print(f"Data shape: {df.shape}")
print(f"Number of genes: {df.shape[0]}")
print(f"Number of samples: {df.shape[1]}")


df_transposed = df.T


labels = []
for idx in df_transposed.index:
    if idx.startswith('CTL'):
        labels.append(0)  # Control
    elif idx.startswith('MDD'):
        labels.append(1)  # MDD
    else:
        labels.append(-1)  # Unknown


X = df_transposed.values.astype(float)
y = np.array(labels)

# Remove any unknown samples
mask = y != -1
X = X[mask]
y = y[mask]

print('Class counts:', np.bincount(y.astype(int)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Define cross-validation and pipelines
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#  Train multiple models (Logistic Regression, Random Forest, SVM, XGBoost):
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    ),
    
    'SVM': SVC(random_state=42, probability=True, class_weight='balanced'),
    'XGBoost': XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    
    if name in ['SVM', 'Logistic Regression']:
         
        max_pca = min(X_train.shape[0], X_train.shape[1])
        pca_grid = [n for n in [10, 25, 50, 100] if n <= max_pca]
        
        pipe = Pipeline([
            ('var', VarianceThreshold(threshold=0.01)),
            ('kbest', SelectKBest(score_func=mutual_info_classif)),
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('clf', model)
        ])
        
        if name == 'SVM':
            param_grid = {
                'kbest__k': [25, 50, 100, 200, 500, 'all'],
                'pca__n_components': pca_grid,
                'clf__C': [0.1, 1, 10],
                'clf__gamma': [0.01, 0.1, 1],
                'clf__kernel': ['rbf', 'linear']
            }
        else:
            param_grid = {
                'kbest__k': [25, 50, 100, 200, 500, 'all'],
                'pca__n_components': pca_grid,
                'clf__C': [0.01, 0.1, 1, 10, 100]
            }
        # Use GridSearchCV for hyperparameter tuning
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        cv_mean = grid.best_score_
        cv_std = grid.cv_results_['std_test_score'][grid.best_index_]
        
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_proba)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
       
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_roc_auc': train_roc_auc,
            'test_roc_auc': test_roc_auc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model': best_model
        }
        
        print(f"Best GridSearchCV parameters: {grid.best_params_}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Train ROC-AUC: {train_roc_auc:.4f}")
        print(f"Test ROC-AUC: {test_roc_auc:.4f}")
        print(f"Train F1 Score: {train_f1:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Train Recall: {train_recall:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Train Precision: {train_precision:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"CV Mean Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_test_pred, target_names=['Control', 'MDD']))
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"Confusion Matrix:")
        print(cm)
        continue
    
    elif name == 'XGBoost':
        # Class imbalance: MDD = 1, CTL = 0
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale_weight = neg / pos if pos > 0 else 1.0
        
        # Base XGBoost model
        base_xgb = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_weight
        )
        
        # Pipeline: variance filter -> KBest -> XGBoost
        xgb_pipe = Pipeline([
            ('var', VarianceThreshold(threshold=0.0)),
            ('kbest', SelectKBest(score_func=mutual_info_classif)),
            ('clf', base_xgb)
        ])
        
        
        param_dist = {
            'kbest__k': [50, 100, 200],
            'clf__n_estimators': [300, 500, 800],
            'clf__learning_rate': [0.01, 0.03, 0.1],
            'clf__max_depth': [2, 3, 4],
            'clf__min_child_weight': [1, 3, 5],
            'clf__gamma': [0, 0.5, 1.0],
            'clf__subsample': [0.7, 0.9, 1.0],
            'clf__colsample_bytree': [0.5, 0.7, 0.9],
            'clf__reg_lambda': [1, 5, 10],
            'clf__reg_alpha': [0, 0.5, 1, 5],
        }
        
        # A bit lighter CV just for XGBoost due to its speed and complexity
        cv_xgb = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(
            xgb_pipe, param_distributions=param_dist, n_iter=40,
            cv=cv_xgb, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
        )
        
        print("Tuning XGBoost")
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        cv_mean = random_search.best_score_
        cv_std = random_search.cv_results_['std_test_score'][random_search.best_index_]
        # Evaluate on training and test sets and calculate metrics
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_proba)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_roc_auc': train_roc_auc,
            'test_roc_auc': test_roc_auc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model': best_model
        }
        
        print(f"Best RandomizedSearchCV parameters (XGBoost): {random_search.best_params_}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Train ROC-AUC: {train_roc_auc:.4f}")
        print(f"Test ROC-AUC: {test_roc_auc:.4f}")
        print(f"Train F1 Score: {train_f1:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Train Recall: {train_recall:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Train Precision: {train_precision:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"CV Mean Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print("\nClassification Report for XGBoost:")
        print(classification_report(y_test, y_test_pred, target_names=['Control', 'MDD']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        continue
    
    # For other models (Random Forest):
    pipe = Pipeline([
        ('var', VarianceThreshold(threshold=0.01)),
        ('kbest', SelectKBest(score_func=mutual_info_classif, k=500)),
        ('scaler', StandardScaler()),
        ('clf', model),
    ])
    
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    # Evaluation metrics will be calculated after fitting the model on the training set
    pipe.fit(X_train, y_train)
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)
    y_train_proba = pipe.predict_proba(X_train)[:, 1]
    y_test_proba = pipe.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    
    results[name] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'model': pipe
    }
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}") 
    print(f"Train ROC-AUC: {train_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Train F1 Score: {train_f1:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Train Recall: {train_recall:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Train Precision: {train_precision:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"CV Mean Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_test_pred, target_names=['Control', 'MDD']))
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Confusion Matrix:")
    print(cm)

# Compare all models

print("Model comparison")

print(f"{'Model':<20} {'Train Acc':<10} {'Test Acc':<10} {'Train AUC':<10} {'Test AUC':<10} {'Train F1':<10} {'Test F1':<10} {'Test Recall':<12} {'Test Prec':<10}")


for name, result in results.items():
    print(f"{name:<20} {result['train_accuracy']:<10.4f} {result['test_accuracy']:<10.4f} {result['train_roc_auc']:<10.4f} {result['test_roc_auc']:<10.4f} {result['train_f1']:<10.4f} {result['test_f1']:<10.4f} {result['test_recall']:<12.4f} {result['test_precision']:<10.4f}")

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
best_model = results[best_model_name]['model']

print(f"\nBest model: {best_model_name} with test accuracy of {results[best_model_name]['test_accuracy']:.4f}")


joblib.dump(best_model, "mdd_trained_model_final.pkl")
print("\nSaved best trained model as 'mdd_trained_model_final.pkl'.")

# Extract top 75 genes using SelectKBest scores
print("TOP 75 MOST IMPORTANT GENES")
# Get gene names from original dataframe (genes are columns, unchanged by sample filtering)
gene_names = df_transposed.columns.tolist()

# Apply preprocessing steps to get feature selection scores
if 'var' in best_model.named_steps and 'kbest' in best_model.named_steps:
    # Apply variance threshold
    var_threshold = best_model.named_steps['var']
    X_var = var_threshold.transform(X_train)
    kbest = best_model.named_steps['kbest']
    feature_scores = kbest.scores_
        # map the surviving features back to the original gene index positions
var_mask = var_threshold.get_support()
original_indices = np.where(var_mask)[0]

top_genes_df = pd.DataFrame({
    'Gene': [gene_names[i] for i in original_indices],
    'Score': feature_scores
}).sort_values('Score', ascending=False)
    
    
print("\nTop 75 Genes:")
print(top_genes_df.head(75).to_string(index=False))
    

