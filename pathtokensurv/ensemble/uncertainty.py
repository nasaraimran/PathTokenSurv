
import numpy as np


def ensemble_mean(predictions):
    return np.asarray(predictions).mean(axis=0)


def ensemble_variance(predictions):
    return np.asarray(predictions).var(axis=0, ddof=1)


def patient_uncertainty(predictions):
    variance = ensemble_variance(predictions)
    return variance.mean(axis=-1)
