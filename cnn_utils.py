#! /usr/bin/env python

import numpy as np
from tensorflow.keras import layers, models, losses
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from data_augmentation import augment
from gc import collect as __gc_collect
from difflib import get_close_matches

def gen_model_0(h, w, d, n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (min(h, 2), 3), activation='relu', input_shape=(h, w, d)))
    model.add(layers.MaxPooling2D((min(h, 2), 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (min(h, 2), 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n))
    return model

def gen_model_33(h, w, d, n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, d)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n))
    return model

def gen_model_44(h, w, d, n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (4, 4), activation='relu', input_shape=(h, w, d)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n))
    return model

def gen_model_55(h, w, d, n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(h, w, d)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n))
    return model

def gen_model_13(h, w, d, n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (1, 3), activation='relu', input_shape=(h, w, d)))
    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (1, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n))
    return model

def gen_model_14(h, w, d, n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (1, 4), activation='relu', input_shape=(h, w, d)))
    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (1, 4), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n))
    return model

def gen_model_15(h, w, d, n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (1, 5), activation='relu', input_shape=(h, w, d)))
    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (1, 5), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n))
    return model

def gen_model_deep445(h, w, d, n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same', input_shape=(h, w, d)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n))
    return model

def gen_model_deep447(h, w, d, n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same', input_shape=(h, w, d)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n))
    return model

def gen_model_deep449(h, w, d, n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same', input_shape=(h, w, d)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n))
    return model


def traintest_neural(features, labels, h=None, fold=9, verbose=False, train_idcs=None, test_idcs=None, epochs=10, model=None):
    """
    Trains a CNN classifier to discriminate candidates between real taps and false positives, then tests the classifier.
    """
    if fold == None:
        fold = 9
    features = np.array(features)
    labels = np.array(labels)
    if len(features.shape) == 2:
        l, f = features.shape
        w = int(f / h)
        features = np.reshape(features, (l, h, w, 1))
    l, h, w, d = features.shape
    print(l, w, h, d)
    if train_idcs is None and test_idcs is None:
        q1 = int(l * fold / 10)
        q2 = int(l * (fold+1) / 10)
        train_idcs = np.concatenate((np.arange(q1), np.arange(start=q2, stop=len(labels))))
        test_idcs = np.arange(start=q1, stop=q2)
    if model is None:
        model = gen_model_0(h, w, d, 2)
    else:
        model = model(h, w, d, 2)
    if verbose:
        model.summary()
    model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(features[train_idcs], labels[train_idcs], epochs=epochs, verbose=0)
    pred_probs = model.predict(features[test_idcs])
    pred = (pred_probs[:, 0] < pred_probs[:, 1]).astype(int)
    act=labels[test_idcs]
    if verbose:
        print("predicted positives:", pred.sum())
        print("actual positives:", act.sum())
        print("true positives:", (pred*act).sum())
    return pred, act

def crossval_neural(features, labels, h=None, epochs=10, model=None, verbose=False):
    """
    Performs 10-fold cross-evaluation of Convolutional neural network classifier that discriminates between real taps and false positives.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, 10):
        pred, act = traintest_neural(features, labels, h, fold=i, verbose=verbose, epochs=epochs, model=model)
        n = len(pred)
        p = pred.sum()
        a = act.sum()
        ctp = (pred * act).sum()
        tp += ctp
        fp += p - ctp
        tn += n - p - a + ctp
        fn += a - ctp
        __gc_collect()
    print("Precision:", tp / (tp + fp))
    print("Recall:", tp / (tp + fn))

def traintest_ch_cnn(features, labels, ch_labels, class_names, fold=None, verbose=False, train_idcs=None, test_idcs=None, epochs=10, model=None):
    """
    Trains a convolutional neural network to discriminate tap candidates between false positives and characters on virtual keyboard
    Tests the classifier. 
    """
    if fold == None:
        fold = 9
    features = np.array(features)
    l, h, w, d = features.shape
    print(l, w, h, d)
    labels = np.array(labels)
    ch_labels = np.array(ch_labels)
    if train_idcs is None and test_idcs is None:
        q1 = int(l * fold / 10)
        q2 = int(l * (fold+1) / 10)
        train_idcs = np.concatenate((np.arange(q1), np.arange(start=q2, stop=l)))
        test_idcs = np.arange(start=q1, stop=q2)
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
    model.fit(features[train_idcs], ch_labels[train_idcs], epochs=epochs, verbose=0)
    proba = model.predict(features[test_idcs])
    chpred = class_names[np.argmax(proba, axis=1)]
    chact = class_names[ch_labels[test_idcs]]
    pred = (chpred != 'FALSE_POSITIVE').astype(int)
    act = labels[test_idcs]
    if verbose:
        print("predicted positives:", pred.sum())
        print("actual positives:", act.sum())
        print("true positives:", (pred*act).sum())
        print("--------")
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
        #for i in range(0, len(test_idcs2)):
        #    if (chact[i] != 'FALSE_POSITIVE'):
        #        print(i, "act", chact[i], "pred", classes[i, -1], int(100*proba[i, proba_sort[i, -1]]), '|', classes[i, -2], int(100*proba[i, proba_sort[i, -2]]), '|', classes[i, -3], int(100*proba[i, proba_sort[i, -3]]), '||', chact[i], int(100*proba[i, class_names==chact[i]]), len(class_names) - np.arange(len(class_names))[classes[i, :] == chact[i]])
    return chpred, chact

def find_pin_cnn(features, labels, ch_labels, jsons, json_id, class_names, ch_features=None, model_gen=None, ch_model_gen=None, epochs=10, max_guess=200, tp_prob_power=1, use_log_probs=False, num_digits=None, penalise_non_chosen=False, ngram_model=None, dictionary=None, verbose=False):
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
    if model_gen is None:
        model_gen = gen_model_0
    if ch_model_gen is None:
        ch_model_gen = model_gen
    prob_identity = 1.0
    prob_lift_fn = np.array
    prob_combine_fn = np.product
    prob_arr_combine_fn = np.multiply
    prob_power_fn = np.power
    prob_ratio_fn = np.divide
    if use_log_probs:
        prob_identity = 0.0
        prob_lift_fn = np.log
        prob_combine_fn = np.sum
        prob_arr_combine_fn = np.add
        prob_power_fn = np.multiply
        prob_ratio_fn = np.subtract
    ngram_str = '^^'
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
        l, h, w, d = np.shape(ch_features)
        ch_model = ch_model_gen(h, w, d, len(class_names))
        if verbose:
            ch_model.summary()
        ch_model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True))
        ch_model.fit(ch_features[train_idcs2], ch_labels[train_idcs2], epochs=epochs, verbose=0)
        proba = ch_model.predict(ch_features[test_idcs2])
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
            j_tp_probs = prob_power_fn(prob_lift_fn(pred_probs[jidcs]), tp_prob_power) ## log
            j_proba = prob_lift_fn(proba[jidcs])  ## log
            #lik_correct_tap = prob_combine_fn(j_tp_probs[act_tp==1, 1])  ## log
            #lik_correct_class = prob_combine_fn([ j_proba[act_tp==1][i, class_names==ch] for i, ch in enumerate(tap_chs)])   ## log
            #lik_correct = prob_arr_combine_fn(lik_correct_tap, lik_correct_class)   ## log
            #if penalise_non_chosen:
            #    lik_correct_fp = prob_combine_fn(j_tp_probs[act_tp==0, 0])  ## log
            #    lik_correct = prob_arr_combine_fn(lik_correct, lik_correct_fp)  ## log
            prob_sorted_per_cand = np.argsort(j_proba, axis=1)[:, ::-1]
            most_likely_per_cand = prob_arr_combine_fn(j_tp_probs[:, 1], np.max(j_proba, axis=1))   ## log
            if penalise_non_chosen:
                most_likely_per_cand = prob_ratio_fn(most_likely_per_cand, j_tp_probs[:, 0])
            cand_order = np.argsort(most_likely_per_cand)[::-1]
            fp_lik_after = [ prob_combine_fn(j_tp_probs[cand_order[i:], 0]) for i in range(jn+1) ]
            def count_lik_correct_alternative(num_left_to_select, i, p):
                if num_left_to_select == 0:
                    if penalise_non_chosen:
                        return prob_arr_combine_fn(p, fp_lik_after[i])
                    return p
                if i == jn:
                    return p
                current_cand = cand_order[i]
                if act_tp[current_cand]:
                    k = np.where(class_names == tap_chs[np.sum(act_tp[:current_cand])])[0][0]
                    print('    count_lik_correct_alternative', i, k, p, current_cand)
                    pp = prob_combine_fn((p, j_tp_probs[current_cand, 1], j_proba[current_cand, k]))
                    return count_lik_correct_alternative(num_left_to_select - 1, i+1, pp)
                else:
                    p0 = p
                    if penalise_non_chosen:
                        p0 = prob_arr_combine_fn(p, j_tp_probs[current_cand, 0])
                    return count_lik_correct_alternative(num_left_to_select, i+1, p0)
            lik_correct = count_lik_correct_alternative(len(tap_chs), 0, prob_identity)
            lik_ngram = prob_identity
            if ngram_model is not None:
                ngram_str = '^' * ngram_model['__len_ngrams__']
                ngram = ngram_str
                for ch in tap_chs:
                    ngram = ngram[1:]
                    if ngram in ngram_model.keys():
                        m = ngram_model[ngram]
                        if ch in m.keys():
                            lik_ngram = prob_arr_combine_fn(lik_ngram, prob_lift_fn(m[ch]))
                    ngram += ch
                lik_correct = prob_arr_combine_fn(lik_correct, lik_ngram)
            if verbose:
                print(tap_chs, (len(tap_chs), len(jsons[j].taps_syncd), num_digits))
                print('likelyhood of correct class:', lik_correct)
            def count_more_likely(num_left_to_select, i, p, ngram, acc):
                #print('  '*(len(tap_chs) - num_left_to_select), 'count_more_likely', num_left_to_select, i, p, ngram, acc)
                if p < lik_correct:
                    return acc
                if num_left_to_select == 0:
                    p1 = p
                    if penalise_non_chosen:
                        p1 = prob_arr_combine_fn(p, fp_lik_after[i])
                    return acc + (0 if p1 < lik_correct else 1)
                if i >= jn:
                    return acc
                r = ngram_model is not None
                ngps = np.full(shape=len(class_names), fill_value=prob_identity)
                current_cand = cand_order[i]
                prob_sorted = prob_sorted_per_cand[current_cand]
                if ngram_model is not None and ngram in ngram_model.keys():
                    current_cand = i
                if ngram_model is not None and ngram in ngram_model.keys():
                    m = ngram_model[ngram]
                    ngps = prob_lift_fn([m[ch] for ch in class_names])
                    prob_sorted = np.argsort(j_proba[current_cand] + ngps)[::-1]
                for k in prob_sorted:
                    pp = prob_combine_fn((p, j_tp_probs[current_cand, 1], j_proba[current_cand, k], ngps[k]))
                    s = count_more_likely(num_left_to_select - 1, i+1, pp, ngram[1:] + class_names[k], acc)   ## log
                    if s == acc:
                        break
                    r = True
                    acc = s
                    if acc >= max_guess:
                        return max_guess + 1
                if r and not np.all(labels==1):
                    p0 = p
                    if penalise_non_chosen:
                        p0 = prob_arr_combine_fn(p, j_tp_probs[current_cand, 0])
                    acc = count_more_likely(num_left_to_select, i+1, p0, ngram, acc)
                    if acc >= max_guess:
                        return max_guess + 1
                return acc
            def dictionary_guessing():
                cand_order = np.argsort(j_tp_probs[:, 1])[::-1]
                string = ''.join(tap_chs)
                if string not in dictionary:
                    print('not in dictionary:', string)
                    m = get_close_matches(string, dictionary, n=1)
                    if len(m) == 0:
                        print('no similar words found either')
                        return max_guess+1
                    string = m[0]
                    print('instead using:', string)
                def get_likelihood_of_word(word):
                    if len(cand_order) < len(word):
                        return -np.inf
                    cands = np.array(sorted(cand_order[:len(word)]))
                    cands_fp = cand_order[len(word):]
                    lik_ch = prob_combine_fn([j_proba[ca, class_names==ch] for ca, ch in zip(cands, word)])
                    lik_tp = prob_combine_fn(j_tp_probs[cands, 1])
                    lik_fp = prob_combine_fn(j_tp_probs[cands_fp, 0])
                    return prob_combine_fn((lik_ch, lik_fp, lik_tp))
                correct_prob = get_likelihood_of_word(string)
                gn = 0
                for w in dictionary:
                    if get_likelihood_of_word(w) > correct_prob:
                        gn += 1
                return gn
            if dictionary is not None:
                guess_number = dictionary_guessing()
            else:
                guess_number = count_more_likely(len(tap_chs), 0, prob_identity, ngram_str[1:], 0) ## log
            if verbose:
                print('guess:', guess_number)
            guess_numbers += [guess_number]
        __gc_collect()
    return guess_numbers
            


                    
            



        







