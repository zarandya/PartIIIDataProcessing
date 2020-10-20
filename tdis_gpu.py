#! /usr/bin/env python
"""
Evaluates effect of distance on tap detection and classification
"""

import re
experimentName = re.compile('tdis.5')
from gc import collect as __gc_collect

fallback_mode = 'use_syncd_to_self'

jsons = []
tap_cand_fourier_6mic_all_features = []
tap_cand_labels = []
tap_cand_ch = []
tap_cand_json_name = []

def dependencies(symbols):
    deps = ['tap_cand_json_name']
    if 'selection_features' in symbols or 'ch_features' in symbols:
        deps += ['tap_cand_fourier_6mic_all_features']
    return deps

correctly_syncd_cands = []
cands_for_each_distance = []
ch_class_names = []
positives = []
selection_features = []
ch_features = []

def init(symbols):
    global correctly_syncd_cands
    global cands_for_each_distance
    global ch_class_names
    global positives
    global tap_cand_labels
    global tap_cand_ch
    global selection_features
    global ch_features
    ch_class_names_arr = sorted(U.np.unique(tap_cand_ch))
    ch_class_names = U.np.array(ch_class_names_arr)
    tap_cand_ch = U.np.array([ ch_class_names_arr.index(ch) for ch in tap_cand_ch ])
    positives = U.np.array(tap_cand_labels) == 1
    if 'correctly_syncd_cands' in symbols:
        correctly_syncd_cands = U.np.array([ not hasattr(jsons[i], 'taps_syncd_orig') for i in tap_cand_json_name ])
    if 'cands_for_each_distance' in symbols:
        cands_for_each_distance = U.np.array([[ jsons[i].experimentName == ('tdis%s5' % j) for i in tap_cand_json_name ] for j in range(6)]) 
    tap_cand_labels = U.np.array(tap_cand_labels)
    tap_cand_ch = U.np.array(tap_cand_ch)
    l, nd,nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    if 'selection_features' in symbols:
        selection_features = U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3)
    if 'ch_features' in symbols:
        print('creating ch_features')
        ch_features = U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1))
    print('ch_features is', type(ch_features))

def crossval_tpfp_r15():
    idcs = cands_for_each_distance[1]
    U.crossval_neural(selection_features[idcs], tap_cand_labels[idcs], verbose=True)

def crossval_tpfp_r15c():
    idcs = cands_for_each_distance[1] * correctly_syncd_cands
    U.crossval_neural(selection_features[idcs], tap_cand_labels[idcs], verbose=True)

def crossval_ch_r15():
    idcs = cands_for_each_distance[1] * positives
    features = ch_features[idcs]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[idcs]
    f = lambda fold: U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_r15c():
    idcs = cands_for_each_distance[1] * correctly_syncd_cands * positives
    features = ch_features[idcs]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[idcs]
    f = lambda fold: U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_tpfp_r25():
    idcs = cands_for_each_distance[2]
    U.crossval_neural(selection_features[idcs], tap_cand_labels[idcs], verbose=True)

def crossval_tpfp_r25c():
    idcs = cands_for_each_distance[2] * correctly_syncd_cands
    U.crossval_neural(selection_features[idcs], tap_cand_labels[idcs], verbose=True)

def crossval_ch_r25():
    idcs = cands_for_each_distance[2] * positives
    features = ch_features[idcs]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[idcs]
    f = lambda fold: U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_r25c():
    idcs = cands_for_each_distance[2] * correctly_syncd_cands * positives
    features = ch_features[idcs]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[idcs]
    f = lambda fold: U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_tpfp_r35():
    idcs = cands_for_each_distance[3]
    U.crossval_neural(selection_features[idcs], tap_cand_labels[idcs], verbose=True)

def crossval_tpfp_r35c():
    idcs = cands_for_each_distance[3] * correctly_syncd_cands
    U.crossval_neural(selection_features[idcs], tap_cand_labels[idcs], verbose=True)

def crossval_ch_r35():
    idcs = cands_for_each_distance[3] * positives
    features = ch_features[idcs]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[idcs]
    f = lambda fold: U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_r35c():
    idcs = cands_for_each_distance[3] * correctly_syncd_cands * positives
    features = ch_features[idcs]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[idcs]
    f = lambda fold: U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_tpfp_r45():
    idcs = cands_for_each_distance[4]
    U.crossval_neural(selection_features[idcs], tap_cand_labels[idcs], verbose=True)

def crossval_tpfp_r45c():
    idcs = cands_for_each_distance[4] * correctly_syncd_cands
    U.crossval_neural(selection_features[idcs], tap_cand_labels[idcs], verbose=True)

def crossval_ch_r45():
    idcs = cands_for_each_distance[4] * positives
    features = ch_features[idcs]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[idcs]
    f = lambda fold: U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_r45c():
    idcs = cands_for_each_distance[4] * correctly_syncd_cands * positives
    features = ch_features[idcs]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[idcs]
    f = lambda fold: U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_tpfp_r55():
    idcs = cands_for_each_distance[5]
    U.crossval_neural(selection_features[idcs], tap_cand_labels[idcs], verbose=True)

def crossval_tpfp_r55c():
    idcs = cands_for_each_distance[5] * correctly_syncd_cands
    U.crossval_neural(selection_features[idcs], tap_cand_labels[idcs], verbose=True)

def crossval_ch_r55():
    idcs = cands_for_each_distance[5] * positives
    features = ch_features[idcs]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[idcs]
    f = lambda fold: U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_r55c():
    idcs = cands_for_each_distance[5] * correctly_syncd_cands * positives
    features = ch_features[idcs]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[idcs]
    f = lambda fold: U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_tpfp_xdist():
    tp = 0
    fp = 0
    fn = 0
    tps = [0,0,0,0,0,0]
    fps = [0,0,0,0,0,0]
    fns = [0,0,0,0,0,0]
    l = len(selection_features)
    for i in range(0, 10):
        q1 = int(l * i / 10)
        q2 = int(l * (i+1) / 10)
        train_idcs = U.np.concatenate((U.np.arange(q1), U.np.arange(start=q2, stop=l)))
        test_idcs = U.np.arange(start=q1, stop=q2)
        pred, act = U.traintest_neural(selection_features, tap_cand_labels, None, train_idcs=train_idcs, test_idcs=test_idcs, verbose=True, epochs=10)
        p = pred.sum()
        a = act.sum()
        ctp = (pred * act).sum()
        tp += ctp
        fp += p - ctp
        fn += a - ctp
        for j in range(1, 6):
            p = ((pred == 1) * cands_for_each_distance[j, test_idcs]).sum()
            a = ((act == 1) * cands_for_each_distance[j, test_idcs]).sum()
            ctp = (((pred * act) == 1) * cands_for_each_distance[j, test_idcs]).sum()
            tps[j] += ctp
            fps[j] += p - ctp
            fns[j] += a - ctp
        __gc_collect()
    print("Precision:", tp / (tp + fp))
    print("Recall:", tp / (tp + fn))
    for j in range(1, 6):
        print("Precision %s5cm:" % j, tps[j] / (tps[j] + fps[j]))
        print("Recall %s5cm:" % j, tps[j] / (tps[j] + fns[j]))


def crossval_ch_xdist():
    features = ch_features[positives]
    tap_labels = U.np.ones(shape=len(features), dtype=int)
    tap_ch = tap_cand_ch[positives]
    tp = 0
    tps = [0,0,0,0,0,0]
    total = 0
    totals = [0,0,0,0,0,0]
    l = len(features)
    for i in range(0, 10):
        print('Fold:', i)
        q1 = int(l * i / 10)
        q2 = int(l * (i+1) / 10)
        train_idcs = U.np.concatenate((U.np.arange(q1), U.np.arange(start=q2, stop=l)))
        test_idcs = U.np.arange(start=q1, stop=q2)
        pred, act = U.traintest_ch_cnn(features, tap_labels, tap_ch, ch_class_names, train_idcs=train_idcs, test_idcs=test_idcs, verbose=True, model=U.gen_model_44)
        n = len(pred)
        ctp = (pred == act).sum()
        tp += ctp
        for j in range(1, 6):
            ctp = ((pred == act) * (cands_for_each_distance[j, positives][test_idcs])).sum()
            tps[j] += ctp
            n = cands_for_each_distance[j, positives][test_idcs].sum()
            totals[j] += n
            print('fold', i, "%s5cm" % j, ctp, '/', n)
        total += n
        __gc_collect()
    print('Crossval Accuracy:', tp/total)
    for j in range(1, 6):
        print('Crossval Accuracy %s5cm:' % j, tps[j]/totals[j])



runs = []
for r in dir():
    if r.startswith('crossval') or r.startswith('guesseval'):
        runs += [r]

