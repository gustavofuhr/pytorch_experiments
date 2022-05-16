import torch.nn as nn

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

softmax = nn.Softmax(dim=1)

def eer_metric(labels, output, classes_map):
    """Compute the Equal Error Rate (EER) from the predictions and scores.
    Args:
        labels (list[int]): values indicating whether the ground truth
            value is positive (1) or negative (0).
        output (list[float]): output, logits, from model
    Return:
        (float, thresh): the Equal Error Rate and the corresponding threshold
    NOTES:
       The EER corresponds to the point on the ROC curve that intersects
       the line given by the equation 1 = FPR + TPR.
       The implementation of the function was taken from here:
       https://yangcha.github.io/EER-ROC/
    """

    # seems these are logits, lets transform them in probs using Softmax
    probs = softmax(outputs)
    import pdb; pdb.set_trace()

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh
