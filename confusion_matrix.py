#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
from sklearn.metrics import ConfusionMatrixDisplay
import re

logfile = 'logs/jpujobs_lscpword.log'
run = 'crossval_ch_6DtF_44'

logfile = '../../../jobs/cpujobs_lscpword/log.log'
run = 'crossval_ch_6_1024nf78fs_0'

name = ''
labels = np.array([c for c in 'qwertyuiopasdfghjklzxcvbnm'])
confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
current_char_id = 0
can_read_matrix_now = False
class_labels_order = []
class_labels_read_mode = False
with open(logfile) as f:
    r = re.compile('.*: \[([0-9]*, )*[0-9]*\]\n')
    for l in f:
        if l.startswith('crossval_ch'):
            name = l.rstrip('\n')
            # reset
            class_labels_order = []
            can_read_matrix_now = False
            confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
            print(name, name==run)
        if name == run:
            class_labels_line = l.rstrip('\n')
            if l.startswith('class labels: '):
                class_labels_read_mode = True
                class_labels_order = []
                class_labels_line = l.lstrip('class labels:[').rstrip('\n')
                can_read_matrix_now = False
            if class_labels_read_mode:
                for i, u in enumerate(class_labels_line.split('"')):
                    if i % 2 == 0:
                        class_labels_order += u.split("'")[1::2]
                    else:
                        class_labels_order += [u]
                if class_labels_line.endswith(']'):
                    class_labels_read_mode = False
                    current_char_id = 0
            if l.startswith('top2'):
                can_read_matrix_now = True
            if can_read_matrix_now and r.fullmatch(l):
                v = l.split('[')[1].rstrip(']\n')
                vv = v.split(', ')
                act_labels = np.where(labels == class_labels_order[current_char_id])
                if len(act_labels) == 1:
                    act_label = act_labels[0]
                    for ch, x in zip(class_labels_order, vv):
                        pred_labels = np.where(labels == ch)
                        if len(pred_labels) == 1:
                            pred_label = pred_labels[0]
                            confusion_matrix[act_label, pred_label] += int(x)
                            #print(ch, 'classified as', class_labels_order[current_char_id], ':', x)
                current_char_id += 1
            if l.startswith('top3'):
                can_read_matrix_now = False
            if l.startswith('Crossval Accuracy:'):
                if name == run:
                    print(confusion_matrix)
                    cmd = ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
                    cmd.plot(cmap=plt.cm.gist_earth)

                    



plt.savefig('AlmosPartIII/figs/confmatrix-lscpword-lda.pgf')


