from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
import joblib
from config import FOGBAYES_SEARCH_SPACE, SEED

def train_FogBayes(X_train, y_train):
    bayes_cv = BayesSearchCV(
        LGBMClassifier(boosting_type='gbdt', random_state=SEED),
        FOGBAYES_SEARCH_SPACE,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        random_state=SEED
    )
    bayes_cv.fit(X_train, y_train)
    joblib.dump(bayes_cv.best_estimator_, 'FogBayes.joblib')
    return bayes_cv
