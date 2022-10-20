from sklearn.model_selection import KFold, StratifiedKFold

def _stratified_kfold(config, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for train_index, test_index in skf.split(X, y):
        return train_index, test_index
    