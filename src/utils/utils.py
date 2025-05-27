import joblib

def save_forecaster(forecaster, filepath: str):
    joblib.dump(forecaster, filepath)


def load_forecaster(filepath: str):
    return joblib.load(filepath)
