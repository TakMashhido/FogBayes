import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import shap


def plot_feature_importance(model, features):
    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    plt.figure(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=fi.sort_values('Importance', ascending=True))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

def plot_roc(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()
    plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig('roc_curve.png')

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

def plot_calibration(y_true, y_pred_proba):
    frac_true, frac_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    plt.figure()
    plt.plot(frac_pred, frac_true, marker='o')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Calibration Curve')
    plt.tight_layout()
    plt.savefig('calibration_curve.png')

def plot_error_hist(y_true, y_pred):
    error = (y_true != y_pred).astype(int)
    plt.figure()
    sns.histplot(error, bins=2)
    plt.title('Prediction Error Histogram')
    plt.tight_layout()
    plt.savefig('error_histogram.png')

def plot_pearson_correlation(df, features):
    """Pearson correlation heatmap (linear relationships)"""
    plt.figure(figsize=(10, 8))
    corr = df[features].corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Pearson Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('pearson_correlation_heatmap.png', dpi=300)
    plt.close()


def plot_spearman_correlation(df, features):
    """Spearman correlation heatmap (monotonic relationships)"""
    plt.figure(figsize=(10, 8))
    corr_matrix = df[features].corr(method='spearman')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='viridis',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Spearman Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('spearman_correlation_heatmap.png', dpi=300)
    plt.close()


def plot_kendall_correlation(df, features):
    """Kendall Tau correlation heatmap (ordinal relationships)"""
    plt.figure(figsize=(10, 8))
    corr_matrix = df[features].corr(method='kendall')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='plasma',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Kendall Tau Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('kendall_correlation_heatmap.png', dpi=300)
    plt.close()


def plot_correlation_with_target(df, features, target):
    """Correlation of features with target variable"""
    correlations = df[features].corrwith(df[target]).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    correlations.plot(kind='barh', color='teal')
    plt.title('Feature Correlation with Target', fontsize=14)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_target_correlation.png', dpi=300)
    plt.close()


def plot_clustermap_correlation(df, features):
    """Clustered correlation heatmap (hierarchical clustering)"""
    corr = df[features].corr(method='pearson')
    sns.clustermap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   center=0, linewidths=0.5, figsize=(10, 10))
    plt.title('Hierarchical Clustered Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('clustermap_correlation.png', dpi=300)
    plt.close()


def plot_all_correlations(df, features, target):
    """Generate all types of correlation heatmaps"""
    plot_pearson_correlation(df, features)
    plot_spearman_correlation(df, features)
    plot_kendall_correlation(df, features)
    plot_correlation_with_target(df, features, target)
    plot_clustermap_correlation(df, features)

def plot_shap_summary(model, X_train, features):
    """SHAP Summary Plot - Fixed to handle numpy arrays"""
    # Convert to DataFrame to preserve feature names
    if isinstance(X_train, np.ndarray):
        X_train_df = pd.DataFrame(X_train, columns=features)
    else:
        X_train_df = X_train
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_df)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train_df, show=False)
    plt.title('SHAP Summary Plot', fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_bar(model, X_train, features):
    """SHAP Bar Plot"""
    if isinstance(X_train, np.ndarray):
        X_train_df = pd.DataFrame(X_train, columns=features)
    else:
        X_train_df = X_train
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_df)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train_df, plot_type="bar", show=False)
    plt.title('SHAP Bar Plot', fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_waterfall(model, X_train, features, sample_idx=0):
    """SHAP Waterfall Plot"""
    if isinstance(X_train, np.ndarray):
        X_train_df = pd.DataFrame(X_train, columns=features)
    else:
        X_train_df = X_train
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train_df)
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    plt.title(f'SHAP Waterfall Plot (Sample {sample_idx})', fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_force(model, X_train, features, sample_idx=0):
    """SHAP Force Plot"""
    if isinstance(X_train, np.ndarray):
        X_train_df = pd.DataFrame(X_train, columns=features)
    else:
        X_train_df = X_train
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_df)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    shap.force_plot(
        explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
        shap_values[sample_idx],
        X_train_df.iloc[sample_idx],
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot (Sample {sample_idx})', fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_force_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

