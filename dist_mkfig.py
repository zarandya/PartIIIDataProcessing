#! /usr/bin/env python

import matplotlib
matplotlib.use('pgf')
plt = matplotlib.pyplot

fig, ax = plt.subplots(nrows=2, sharex=True)

X = [15, 25, 35, 45, 55]

recall_lda_exp1 = [0.7528846153846154, 0.703076923076923, 0.5253731343283582, 0.22545454545454546, 0.17180616740088106]
recall_cnn_exp1 = [0.8125, 0.683076923076923, 0.4835820895522388, 0.11272727272727273, 0.02643171806167401]
recall_lda_exp2 = [0.7855769230769231, 0.7138461538461538, 0.6029850746268657, 0.4036363636363636, 0.34801762114537443]
recall_cnn_exp2 = [0.5682692307692307, 0.5553846153846154, 0.5761194029850746, 0.36, 0.23348017621145375]

ax[0].plot(X, recall_cnn_exp1, c='green', label='CNN, Exp. 1')
ax[0].plot(X, recall_cnn_exp2, c='blue', label='CNN, Exp. 2')
ax[0].plot(X, recall_lda_exp1, c='lime', label='LDA, Exp. 1')
ax[0].plot(X, recall_lda_exp2, c='cyan', label='LDA, Exp. 2')
ax[0].legend()
ax[0].grid()
ax[0].set_xticks(X)
ax[0].set_ylabel('Recall')

acc_lda_exp1 = [0.5336538461538461, 0.37846153846153846, 0.28059701492537314, 0.16, 0.11894273127753303]
acc_lda_exp2 = [0.5067307692307692, 0.41846153846153844, 0.33432835820895523, 0.2727272727272727, 0.21585903083700442]
acc_cnn_exp1 = [0.4673076923076923, 0.34307692307692306, 0.21194029850746268, 0.20363636363636364, 0.1762114537444934]
acc_cnn_exp2 = [0.4221153846153846, 0.41384615384615386, 0.3283582089552239, 0.28363636363636363, 0.1894273127753304]


ax[1].plot(X, acc_cnn_exp1, c='green', label='CNN, Exp. 1')
ax[1].plot(X, acc_cnn_exp2, c='blue', label='CNN, Exp. 2')
ax[1].plot(X, acc_lda_exp1, c='lime', label='LDA, Exp. 1')
ax[1].plot(X, acc_lda_exp2, c='cyan', label='LDA, Exp. 2')
ax[1].legend()
ax[1].grid()
ax[1].set_xticks(X)
ax[1].set_ylabel('Accuracy')

#plt.xticks(X)
plt.xlabel('Distance (cm)')
plt.savefig('AlmosPartIII/figs/tdist-results.pgf')
