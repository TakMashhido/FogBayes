import logging
from config import *
from data_preprocessing import load_data, clean_data, split_data
from feature_engineering import scale_features
from model_training import train_FogBayes
from model_testing import test_model
from cross_validation import cross_val_cv
from benchmarking import benchmark
from visualization import plot_feature_importance, plot_roc, plot_confusion_matrix, plot_calibration, plot_error_hist, plot_all_correlations, plot_shap_summary, plot_shap_bar, plot_shap_waterfall, plot_shap_force

def main():
    logging.basicConfig(level=logging.INFO)
    df = load_data(DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET, TEST_SIZE, SEED)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    bayes_cv = train_FogBayes(X_train_scaled, y_train)
    y_pred, y_pred_proba, model = test_model(X_test_scaled)
    benchmark(y_test, y_pred, y_pred_proba)
    plot_feature_importance(model, FEATURES)
    plot_roc(y_test, y_pred_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_calibration(y_test, y_pred_proba)
    plot_error_hist(y_test, y_pred)
    plot_all_correlations(df, FEATURES, TARGET)
    plot_shap_summary(model, X_train_scaled, FEATURES)
    plot_shap_bar(model, X_train_scaled, FEATURES)
    plot_shap_waterfall(model, X_train_scaled, FEATURES, sample_idx=0)
    plot_shap_force(model, X_train_scaled, FEATURES, sample_idx=0)
    scores = cross_val_cv(X_train_scaled, y_train)
    print('Cross-validation scores:', scores)
    logging.info("Pipeline run complete. Results saved.")

if __name__ == "__main__":
    main()
