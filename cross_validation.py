from sklearn.model_selection import StratifiedKFold, cross_validate
from lightgbm import LGBMClassifier

def cross_val_cv(X, y, n_folds=5):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_validate(LGBMClassifier(random_state=42), X, y, cv=cv,
                            scoring=['accuracy', 'roc_auc', 'f1'], return_train_score=True)
    return scores
