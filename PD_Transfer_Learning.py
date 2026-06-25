import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# PD file lives in the same directory as the MDD model/data
MDD_FILE = "221125_MDD_DEGs_raw_data.csv"
PD_FILE = "221125PD_RMA_unfiltered_only_overlappinggenes_with_mdd_pd_vs_ctl.xlsx"
MDD_MODEL_FILE = "mdd_trained_model_final.pkl"
PD_TRANSFER_MODEL_FILE = "pd_transfer_model_final.pkl"


# load + validate the MDD model

print("loading the trained MDD model...")
try:
    mdd_model = joblib.load(MDD_MODEL_FILE)
    print(f"loaded MDD model from '{MDD_MODEL_FILE}'")
except FileNotFoundError:
    print(f"ERROR: could not find '{MDD_MODEL_FILE}' - train the MDD model first.")
    sys.exit(1)

if not hasattr(mdd_model, "named_steps"):
    print(f"ERROR: MDD model isn't a sklearn Pipeline (got {type(mdd_model)}). Save it as a Pipeline.")
    sys.exit(1)

if "clf" not in mdd_model.named_steps:
    print(f"ERROR: MDD pipeline has no 'clf' step. Steps found: {list(mdd_model.named_steps.keys())}")
    sys.exit(1)

print(f"MDD pipeline steps: {list(mdd_model.named_steps.keys())}\n")


# recover the gene order used during MDD training

print("recovering gene order from the MDD dataset...")
try:
    mdd_df = pd.read_csv(MDD_FILE, index_col=0)  # genes x samples
    mdd_df_T = mdd_df.T  # samples x genes
    mdd_gene_order = mdd_df_T.columns.tolist()  # this is the feature order the model was trained on
    print(f"MDD dataset shape: {mdd_df.shape}, genes: {len(mdd_gene_order)}\n")
except FileNotFoundError:
    print(f"ERROR: could not find '{MDD_FILE}', can't recover gene order.")
    sys.exit(1)


# load PD dataset + build labels

print("loading PD dataset...")
try:
    pd_df = pd.read_excel(PD_FILE, index_col=0, engine="openpyxl")  # genes x samples
    print(f"PD dataset shape: {pd_df.shape}")
except FileNotFoundError:
    print(f"ERROR: could not find '{PD_FILE}'.")
    sys.exit(1)

pd_df_T = pd_df.T  # samples x genes

# HC = control, mild_pd/mod_pd/sev_pd = Parkinson's
labels_pd = []
for idx in pd_df_T.index:
    name = str(idx).upper()
    if "HC" in name:
        labels_pd.append(0)
    elif "MILD_PD" in name or "MOD_PD" in name or "SEV_PD" in name:
        labels_pd.append(1)
    else:
        labels_pd.append(-1)  # unknown, gets dropped below

y_pd = np.array(labels_pd)
mask = y_pd != -1

if mask.sum() == 0:
    print("ERROR: no valid PD samples found, check sample naming.")
    sys.exit(1)

X_pd_raw = pd_df_T.values.astype(float)[mask]
y_pd = y_pd[mask]
pd_sample_names = pd_df_T.index[mask]

print(f"PD usable samples: {X_pd_raw.shape[0]}")
print(f"class counts (0=CTL, 1=PD): {np.bincount(y_pd.astype(int))}")


# same QC checks as the MDD training run

missing_count = np.isnan(X_pd_raw).sum()
if missing_count > 0:
    print(f"found {missing_count} missing values, filling with median")
    imputer = SimpleImputer(strategy='median')
    X_pd_raw = imputer.fit_transform(X_pd_raw)
else:
    print("no missing values")

inf_count = np.isinf(X_pd_raw).sum()
if inf_count > 0:
    print(f"found {inf_count} infinite values, clipping to finite range")
    finite_vals = X_pd_raw[~np.isinf(X_pd_raw)]
    X_pd_raw = np.nan_to_num(
        X_pd_raw,
        posinf=finite_vals.max() if finite_vals.size > 0 else 1e6,
        neginf=finite_vals.min() if finite_vals.size > 0 else -1e6,
    )
else:
    print("no infinite values")


# align PD genes to the exact MDD training order

pd_gene_names = pd_df_T.columns.tolist()
common_genes = [g for g in mdd_gene_order if g in pd_gene_names]
missing_genes = [g for g in mdd_gene_order if g not in pd_gene_names]

print(f"common genes: {len(common_genes)} / {len(mdd_gene_order)}")
if missing_genes:
    print(f"{len(missing_genes)} MDD genes are missing from the PD dataset")
    if len(missing_genes) > 50:
        print(f"first 10 missing: {missing_genes[:10]}")
    else:
        print(f"missing genes: {missing_genes}")

if len(common_genes) < len(mdd_gene_order) * 0.8:
    pct = len(common_genes) / len(mdd_gene_order) * 100
    print(f"only {pct:.1f}% of MDD genes found in PD data - transfer learning may be weaker")

pd_df_aligned = pd.DataFrame(index=pd_sample_names, columns=mdd_gene_order)

for gene in common_genes:
    pd_df_aligned[gene] = pd_df_T.loc[pd_sample_names, gene].values
for gene in missing_genes:
    pd_df_aligned[gene] = np.nan

X_pd_aligned = pd_df_aligned.values.astype(float)

if missing_genes:
    print(f"filling {len(missing_genes)} missing genes with their median")
    col_medians = np.nanmedian(X_pd_aligned, axis=0)
    nan_cols = np.isnan(X_pd_aligned).any(axis=0)
    X_pd_aligned[:, nan_cols] = np.where(
        np.isnan(X_pd_aligned[:, nan_cols]),
        col_medians[nan_cols],
        X_pd_aligned[:, nan_cols],
    )

print(f"aligned PD matrix: {X_pd_aligned.shape}\n")


# train/test split

X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
    X_pd_aligned,
    y_pd,
    test_size=0.2,
    random_state=42,
    stratify=y_pd,
)

print(f"PD train: {X_train_pd.shape}, test: {X_test_pd.shape}\n")


# reuse every MDD pipeline step except the final classifier as a feature extractor

steps = list(mdd_model.named_steps.items())
feature_steps = steps[:-1]
feature_extractor = Pipeline(feature_steps)

print("MDD feature extractor steps:", [name for name, _ in feature_steps])


# push PD data through it (no refitting on PD)

print("transforming PD data through the MDD feature extractor...")
try:
    X_train_feat = feature_extractor.transform(X_train_pd)
    X_test_feat = feature_extractor.transform(X_test_pd)
    print(f"X_train_feat: {X_train_feat.shape}, X_test_feat: {X_test_feat.shape}\n")
except Exception as e:
    print(f"ERROR during feature transformation: {e}")
    raise


# train a new classifier on PD (the actual transfer learning step)

print("training PD classifier on top of MDD-derived features...")
base_clf = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42,
    solver="lbfgs",
)

param_grid_clf = {
    "C": [0.01, 0.1, 1, 10, 100],
}

cv_pd = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_clf = GridSearchCV(
    base_clf,
    param_grid=param_grid_clf,
    cv=cv_pd,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=1,
)

grid_clf.fit(X_train_feat, y_train_pd)
best_clf_pd = grid_clf.best_estimator_

print(f"best C: {grid_clf.best_params_['C']}, cv score: {grid_clf.best_score_:.4f}\n")


# evaluate

y_train_pred = best_clf_pd.predict(X_train_feat)
y_test_pred = best_clf_pd.predict(X_test_feat)
y_test_proba = best_clf_pd.predict_proba(X_test_feat)[:, 1]

train_acc = accuracy_score(y_train_pd, y_train_pred)
test_acc = accuracy_score(y_test_pd, y_test_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test_pd, y_test_pred, average='weighted'
)
roc_auc = roc_auc_score(y_test_pd, y_test_proba)

print("PD transfer model results")
print(f"train acc {train_acc:.4f} | test acc {test_acc:.4f}")
print(f"test precision {precision:.4f}, recall {recall:.4f}, f1 {f1:.4f}, auc {roc_auc:.4f}\n")
print(classification_report(y_test_pd, y_test_pred, target_names=["Control", "PD"]))
print(confusion_matrix(y_test_pd, y_test_pred))
print()


# wrap extractor + classifier into one pipeline and save

pd_transfer_model = Pipeline([
    ("features", feature_extractor),
    ("clf", best_clf_pd),
])

joblib.dump(pd_transfer_model, PD_TRANSFER_MODEL_FILE)
print(f"saved PD transfer model to '{PD_TRANSFER_MODEL_FILE}'\n")


# top 75 genes, using the MDD SelectKBest scores reused for this model

print("top 75 genes for the PD transfer model")
gene_names = pd_df_aligned.columns.tolist()

if hasattr(pd_transfer_model, 'named_steps') and 'features' in pd_transfer_model.named_steps:
    feature_extractor_used = pd_transfer_model.named_steps['features']

    if hasattr(feature_extractor_used, 'named_steps') and 'kbest' in feature_extractor_used.named_steps:
        kbest = feature_extractor_used.named_steps['kbest']

        if 'var' in feature_extractor_used.named_steps:
            var_threshold = feature_extractor_used.named_steps['var']
            var_mask = var_threshold.get_support()
            original_indices = np.where(var_mask)[0]

            if hasattr(kbest, 'scores_'):
                feature_scores = kbest.scores_

                if len(feature_scores) == len(original_indices):
                    top_genes_df = pd.DataFrame({
                        'Gene': [gene_names[i] for i in original_indices],
                        'Score': feature_scores,
                    }).sort_values('Score', ascending=False)

                    top_75 = top_genes_df.head(75)
                    print("(scores come from MDD's SelectKBest step, reused here)")
                    print(top_75.to_string(index=False))

                    top_75.to_csv('pd_top_75_genes.csv', index=False)
                    print("saved to 'pd_top_75_genes.csv'")
                else:
                    print(f"score array length ({len(feature_scores)}) doesn't match gene count ({len(original_indices)})")
            else:
                print("kbest has no scores_ - feature extractor may not be fitted")
        else:
            print("no variance threshold step in the feature extractor")
    else:
        print("no kbest step in the feature extractor")
else:
    print("can't access the feature extractor from this model")


# coefficient-based importance, on whatever space the classifier actually sees

print("\nPD classifier coefficients (transformed feature importance)")
if hasattr(best_clf_pd, 'coef_'):
    coef = best_clf_pd.coef_[0] if len(best_clf_pd.coef_.shape) > 1 else best_clf_pd.coef_
    coef_abs = np.abs(coef)

    if 'features' in pd_transfer_model.named_steps:
        feat_ext = pd_transfer_model.named_steps['features']
        if hasattr(feat_ext, 'named_steps') and 'kbest' in feat_ext.named_steps:
            kbest_used = feat_ext.named_steps['kbest']
            if hasattr(kbest_used, 'get_support'):
                selected_mask = kbest_used.get_support()
                if 'var' in feat_ext.named_steps:
                    var_mask = feat_ext.named_steps['var'].get_support()
                    var_indices = np.where(var_mask)[0]
                    selected_indices = var_indices[selected_mask]

                    # if the MDD model used PCA, these coefficients are on PCA components, not raw genes
                    print("note: model uses PCA after kbest, so this is by PCA component, not individual gene")
                    feat_importance_df = pd.DataFrame({
                        'PC_Component': range(len(coef_abs)),
                        'Coefficient_Abs': coef_abs,
                    }).sort_values('Coefficient_Abs', ascending=False)

                    print(feat_importance_df.head(75).to_string(index=False))
                    print(f"PCA components used: {len(coef_abs)}, genes selected by kbest: {selected_mask.sum()}")
                else:
                    selected_indices = np.where(selected_mask)[0]
                    print(f"features selected by kbest: {len(selected_indices)}")

print("\ngene extraction done")


# summary

print("\ntransfer learning complete")
print(f"test accuracy: {test_acc:.4f}, test auc: {roc_auc:.4f}")
print("top 75 genes saved to pd_top_75_genes.csv")