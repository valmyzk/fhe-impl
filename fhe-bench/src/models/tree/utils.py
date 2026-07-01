from src.datasets.shanghai import load_dataset, FEATURE_COLS


def train(model):
    df = load_dataset()
    X = df[FEATURE_COLS].values
    y = df["cgm"].values
    model.fit(X, y)
    return model
