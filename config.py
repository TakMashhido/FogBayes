DATA_PATH = 'Dataset/machine failure.csv'
FEATURES = ['Torque [Nm]', 'Rotational speed [rpm]', 'Process temperature [K]', 'Air temperature [K]', 'Tool wear [min]']
TARGET = 'Machine failure'
TEST_SIZE = 0.2
SEED = 42
SCALER = 'StandardScaler'
FOGBAYES_SEARCH_SPACE = {
    'num_leaves': (20, 100),
    'learning_rate': (0.01, 0.2, 'log-uniform'),
    'n_estimators': (50, 150),
    'bagging_fraction': (0.7, 1.0, 'uniform'),
}
