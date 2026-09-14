
import numpy as np
import pandas as pd


def patient_uncertainty(member_survival_curves):
    """
    member_survival_curves:
        shape = [ensemble_members, patients, time_bins]
    """
    x = np.asarray(member_survival_curves)
    variance = x.var(axis=0)
    return variance.mean(axis=1)


def uncertainty_groups(scores):
    q1, q2 = np.quantile(scores, [0.33, 0.66])
    labels = []
    for s in scores:
        if s <= q1:
            labels.append("low")
        elif s <= q2:
            labels.append("medium")
        else:
            labels.append("high")
    return labels
