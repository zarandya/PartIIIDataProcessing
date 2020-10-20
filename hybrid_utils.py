#! /usr/bin/env python

import numpy as np
from tensorflow.keras import layers, models, losses
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from data_augmentation import augment
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from cnn_utils import gen_model_0

def traintest_tp_lda_ch_cnn(features_tp, labels, ch_features, ch_labels, class_names, fold=None, verbose=False, train_idcs=None, test_idcs=None, model=None, epochs=10):
    """
    Trains a linear discriminant analysis classifier to discriminate real taps and false positives, then a convolutional neural network to discriminate characters on a virtual keyboard.
    Tests the two classifiers, making the input of the second one the candidates that were marked as true positives by the first one. 
    """
    if fold == None:
        fold = 9
    l = len(labels)
    features = np.reshape(np.abs(features_tp), (l, int(np.prod(np.shape(features_tp)) / l)))
    labels = np.array(labels)
    ch_labels = np.array(ch_labels)
    it = np.arange(l)
    q1 = int(l * fold / 10)
    q2 = int(l * (fold+1) / 10)
    if train_idcs is None and test_idcs is None:
        train_idcs = np.concatenate((np.arange(q1), np.arange(start=q2, stop=l)))
        test_idcs = np.arange(start=q1, stop=q2)
    lda1 = LinearDiscriminantAnalysis()
    lda1.fit(features[train_idcs], labels[train_idcs])
    pred = lda1.predict(features[test_idcs])
    act = labels[test_idcs]
    if verbose:
        print("predicted positives:", pred.sum())
        print("actual positives:", act.sum())
        print("true positives:", (pred*act).sum())
        print("--------")
    features = np.array(ch_features)
    l, h, w, d = features.shape
    train_idcs2 = train_idcs[labels[train_idcs] == 1]
    test_idcs2 = test_idcs[pred == 1]
    if model is None:
        model = models.Sequential()
        model.add(layers.Conv2D(32, (min(h, 2), 3), activation='relu', input_shape=(h, w, d)))
        model.add(layers.MaxPooling2D((min(h, 2), 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (min(h, 2), 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(len(class_names)))
    else:
        model = model(h, w, d, len(class_names))
    if verbose:
        model.summary()
    model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(features[train_idcs2], ch_labels[train_idcs2], epochs=epochs, verbose=0)
    proba = model.predict(features[test_idcs2])
    chpred = class_names[np.argmax(proba, axis=1)]
    chact = class_names[ch_labels[test_idcs2]]
    pred = (chpred != 'FALSE_POSITIVE').astype(int)
    act = labels[test_idcs]
    intop1 = np.sum((chpred == chact) * (chact != 'FALSE_POSITIVE') * (chact != ''))
    n = np.sum((chact != 'FALSE_POSITIVE') * (chact != ''))
    if verbose:
        print("correctly located: ", intop1)
        print("ratio:", intop1 / n)
        print("ratio of predicted positives:", np.sum(chpred == chact) / len(chact))
    proba_sort = np.argsort(proba, axis=1)
    classes = class_names[proba_sort]
    intop2 = np.sum((chact != 'FALSE_POSITIVE') * (chact != '') * ((chact == classes[:, -1]) + (chact == classes[:, -2])))
    if verbose:
        print("class labels:", class_names)
        print("top2:", intop2);
        print("top2 ratio:", intop2 / n);
        for j in class_names:
            print("    ", j, ':', [np.sum(classes[chact == j, -1] == k) for k in class_names])
    intop3 = np.sum((chact != 'FALSE_POSITIVE') * (chact != '') * ((chact == classes[:, -1]) + (chact == classes[:, -2]) + (chact == classes[:, -3])))
    if verbose:
        print("top3:", intop3);
        print("top3 ratio:", intop3 / n);
    return chpred, chact

def find_pin_hybrid(features, labels, ch_labels, jsons, json_id, class_names, ch_features=None, model_gen=None, epochs=10, max_guess=200, tp_prob_power=1, num_digits=None, penalise_non_chosen=True, verbose=False):
    json_id = np.array(json_id)
    labels = np.array(labels)
    ch_labels = np.array(ch_labels)
    uniq_json_id = np.unique(json_id)
    json_idcs_fully_syncd = np.zeros(shape=len(uniq_json_id), dtype=bool)
    for i, json in enumerate(uniq_json_id):
        cands = json_id==json
        if np.sum(labels[cands]) == len(jsons[json].taps_syncd):
            json_idcs_fully_syncd[i] = True
    if verbose:
        print(np.sum(json_idcs_fully_syncd), 'of', len(uniq_json_id), 'reocrdings fully sync\'d')
    json_id_fully_syncd = uniq_json_id[json_idcs_fully_syncd]
    jl = len(json_id_fully_syncd)
    if verbose:
        print('fully syncd:', jl)
    guess_numbers = []
    if ch_features is None:
        ch_features = features
    shp = np.shape(ch_features)
    if len(shp) > 2:
        ch_features = np.reshape(ch_features, (shp[0], int(np.product(np.array(shp)[1:]))))
    if model_gen is None:
        model_gen = gen_model_0
    num_correctly_detected = 0
    for fold in range(10):
        l, h, w, d = np.shape(features)
        train_json_idx = np.concatenate((np.arange(int(fold / 10 * jl)), np.arange(int((fold+1) / 10 * jl), jl)))
        test_json_idx = np.arange(int(fold / 10 * jl), int((fold+1) / 10 * jl))
        train_idcs = np.isin(json_id, json_id_fully_syncd[train_json_idx]) + np.isin(json_id, uniq_json_id[json_idcs_fully_syncd == False]) # bool array
        test_idcs = np.isin(json_id, json_id_fully_syncd[test_json_idx]) # bool array
        model = model_gen(h, w, d, 2)
        if verbose:
            print('fold', fold)
            model.summary()
        model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True))
        model.fit(features[train_idcs], labels[train_idcs], epochs=epochs, verbose=0)
        pred_probs = model.predict(features[test_idcs])
        pred_probs = softmax(pred_probs, axis=1)
        act=labels[test_idcs]
        train_idcs2 = train_idcs * (labels == 1)    # bool array
        test_idcs2 = test_idcs # bool array
        lda = LinearDiscriminantAnalysis()
        lda.fit(ch_features[train_idcs2], ch_labels[train_idcs2])
        _proba = lda.predict_log_proba(ch_features[test_idcs2])
        proba = np.transpose([ _proba[:, lda.classes_==i][:, 0] if i in lda.classes_ else np.full(len(_proba), -np.inf) for i in range(len(class_names))])  ## log
        chpred = class_names[np.argmax(proba, axis=1)]
        proba = softmax(proba, axis=1)
        chact = class_names[ch_labels[test_idcs2]]
        test2_labels = labels[test_idcs2]
        for j in json_id_fully_syncd[test_json_idx]:
            jidcs = json_id[test_idcs2] == j
            jn = np.sum(jidcs)
            act_tp = test2_labels[jidcs]
            if np.sum(act_tp) < len(jsons[j].taps_syncd):
                guess_numbers += [max_guess+2]
                continue
            tap_chs = [ tap.ch for tap in jsons[j].taps_syncd ]
            if num_digits is not None and len(jsons[j].taps_syncd) > num_digits:
                tap_chs = []
                tf = 0
                tcf = 0
                for i in range(jn):
                    if act_tp[i] == 1:
                        if jsons[j].taps_syncd[tf].ch != '' and tcf < num_digits:
                            tcf += 1
                            tap_chs += [jsons[j].taps_syncd[tf].ch]
                        else:
                            act_tp[i] = 0
                        tf += 1
            j_tp_probs = np.multiply(np.log(pred_probs[jidcs]), tp_prob_power) ## log
            j_proba = (proba[jidcs])  ## log
            lik_correct_tap = np.sum(j_tp_probs[act_tp==1, 1])  ## log
            lik_correct_class = np.sum([ j_proba[act_tp==1][i, class_names==ch] for i, ch in enumerate(tap_chs)])   ## log
            lik_correct = np.add(lik_correct_tap, lik_correct_class)   ## log
            if penalise_non_chosen:
                lik_correct_fp = np.sum(j_tp_probs[act_tp==0, 0])  ## log
                lik_correct = np.add(lik_correct, lik_correct_fp)  ## log
            if verbose:
                print('likelihood of correct class:', lik_correct)
                if np.all(j_tp_probs[act_tp==0, 1] <= np.min(j_tp_probs[act_tp==1, 1])):
                    num_correctly_detected += 1
            prob_sorted_per_cand = np.argsort(j_proba, axis=1)[:, ::-1]
            most_likely_per_cand = np.add(j_tp_probs[:, 1], np.max(j_proba, axis=1))   ## log
            cand_order = np.argsort(most_likely_per_cand)[::-1]
            fp_lik_after = [ np.sum(j_tp_probs[cand_order[i:], 0]) for i in range(jn+1) ]
            def count_more_likely(num_left_to_select, i, p, acc = 0):
                #print('count_more_likely', num_left_to_select, i, p, acc)
                if p < lik_correct:
                    return acc
                if num_left_to_select == 0:
                    p1 = p
                    if penalise_non_chosen:
                        p1 = np.add(p, fp_lik_after[i])
                    return acc + (0 if p1 < lik_correct else 1)
                if i >= jn:
                    return acc
                r = False
                for k in prob_sorted_per_cand[cand_order[i]]:
                    pp = np.sum((p, j_tp_probs[cand_order[i], 1], j_proba[cand_order[i], k]))
                    s = count_more_likely(num_left_to_select - 1, i+1, pp, acc)   ## log
                    if s == acc:
                        break
                    r = True
                    acc = s
                    if acc >= max_guess:
                        return max_guess + 1
                if r:
                    p0 = p
                    if penalise_non_chosen:
                        p0 = np.add(p, j_tp_probs[cand_order[i], 0])
                    acc = count_more_likely(num_left_to_select, i+1, p0, acc)
                    if acc >= max_guess:
                        return max_guess + 1
                return acc
            guess_number = count_more_likely(len(tap_chs), 0, 0.0) ## log
            if verbose:
                print('guess:', guess_number)
            guess_numbers += [guess_number]
    if verbose:
        print('true positives were correctly detected in', num_correctly_detected, '/', len(guess_numbers), 'recordings (', num_correctly_detected / len(guess_numbers), ')')
    return guess_numbers
            

