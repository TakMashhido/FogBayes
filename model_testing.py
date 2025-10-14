import joblib

def test_model(X_test):
    model = joblib.load('FogBayes.joblib')
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba, model
