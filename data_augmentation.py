#! /usr/bin/env python

from random import randrange
import numpy as __np

def augment(positives, labels, negatives, n):
    lp = len(positives)
    ln = len(negatives)
    out_data = []
    out_labels = []
    for i in range(n):
        j = randrange(lp)
        k = randrange(ln)
        out_data += [positives[j] + negatives[k]]
        out_labels += [labels[j]]
    if type(positives) is __np.ndarray:
        out_data = __np.array(out_data)
    if type(labels) is __np.ndarray:
        out_labels = __np.array(out_labels)
    return out_data, out_labels

def augment_inorder(positives, labels, negatives, l):
    lp = len(positives)
    ln = len(negatives)
    out_data = []
    out_labels = []
    for j in range(lp):
        for i in range(l):
            k = randrange(ln)
            out_data += [positives[j] + negatives[k]]
            out_labels += [labels[j]]
    if type(positives) is __np.ndarray:
        out_data = __np.array(out_data)
    if type(labels) is __np.ndarray:
        out_labels = __np.array(out_labels)
    return out_data, out_labels


