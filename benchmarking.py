from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, log_loss
import pandas as pd

def benchmark(y_true, y_pred, y_pred_proba):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    ll = log_loss(y_true, y_pred_proba)
    report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))
    cm = confusion_matrix(y_true, y_pred)
    report.to_csv('classification_report.csv')
    pd.DataFrame(cm).to_csv('confusion_matrix.csv')
    with open('metrics.txt', 'w') as f:
        f.write(f'Accuracy:{acc}\nAUC:{auc}\nLogLoss:{ll}\n')
