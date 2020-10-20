#! /usr/bin/env python

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from multiprocessing import Process, Queue
from time import sleep
from difflib import get_close_matches

def traintest(features, labels, discriminant_analysis=LinearDiscriminantAnalysis, fold=None, verbose=False, train_idcs=None, test_idcs=None):
    """
    Trains a discriminant analysis classifier to discriminate candidates between real taps and false positives, then tests the classifier.
    """
    if fold == None:
        fold = 9
    l = len(labels)
    features = np.reshape(features, (l, int(np.prod(np.shape(features)) / l)))
    labels = np.array(labels)
    q1 = int(len(labels) * fold / 10)
    q2 = int(len(labels) * (fold+1) / 10)
    if train_idcs is None and test_idcs is None:
        train_idcs = np.concatenate((np.arange(q1), np.arange(start=q2, stop=len(labels))))
        test_idcs = np.arange(start=q1, stop=q2)
    lda = discriminant_analysis()
    lda.fit(features[train_idcs], labels[train_idcs])
    pred=lda.predict(features[test_idcs])
    act=labels[test_idcs]
    if verbose:
        print("predicted positives:", pred.sum())
        print("actual positives:", act.sum())
        print("true positives:", (pred*act).sum())
    return pred, act

def crossval(features, labels, discriminant_analysis=LinearDiscriminantAnalysis, verbose=False):
    """
    Performs 10-fold cross-evaluation of discriminant analysis classifier that discriminates between real taps and false positives.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    l = len(labels)
    features = np.reshape(features, (l, int(np.prod(np.shape(features)) / l)))
    labels = np.array(labels)
    for i in range(0, 10):
        pred, act = traintest(features, labels, discriminant_analysis, fold=i, verbose=verbose)
        n = len(pred)
        p = pred.sum()
        a = act.sum()
        ctp = (pred * act).sum()
        tp += ctp
        fp += p - ctp
        tn += n - p - a + ctp
        fn += a - ctp
    print("Precision:", tp / (tp + fp))
    print("Recall:", tp / (tp + fn))
        
def traintest_ch(features, labels, ch_labels, class_names, fold=None, verbose=False, train_idcs=None, test_idcs=None):
    """
    Trains a linear discriminant analysis classifier to discriminate real taps and false positives, then another one to discriminate characters on a virtual keyboard.
    Tests the two classifiers, making the input of the second one the candidates that were marked as true positives by the first one. 
    """
    if fold == None:
        fold = 9
    l = len(labels)
    features = np.reshape(features, (l, int(np.prod(np.shape(features)) / l)))
    labels = np.array(labels)
    ch_labels = np.array(ch_labels)
    it = np.arange(l)
    q1 = int(l * fold / 10)
    q2 = int(l * (fold+1) / 10)
    if train_idcs is None and test_idcs is None:
        train_idcs = np.concatenate((np.arange(q1), np.arange(start=q2, stop=l)))
        test_idcs = np.arange(start=q1, stop=q2)
    if train_idcs.dtype == 'bool':
        train_idcs = np.where(train_idcs)[0]
    if test_idcs.dtype == 'bool':
        test_idcs = np.where(test_idcs)[0]
    lda1 = LinearDiscriminantAnalysis()
    lda1.fit(features[train_idcs], labels[train_idcs])
    pred = lda1.predict(features[test_idcs])
    act = labels[test_idcs]
    if verbose:
        print("predicted positives:", pred.sum())
        print("actual positives:", act.sum())
        print("true positives:", (pred*act).sum())
        print("--------")
    train_idcs2 = train_idcs[labels[train_idcs] == 1]
    test_idcs2 = test_idcs[pred == 1]
    lda2 = LinearDiscriminantAnalysis()
    lda2.fit(features[train_idcs2], ch_labels[train_idcs2])
    chpred = class_names[lda2.predict(features[test_idcs2])]
    chact = class_names[ch_labels[test_idcs2]]
    intop1 = np.sum((chpred == chact) * (chact != 'FALSE_POSITIVE') * (chact != ''))
    n = np.sum((chact != 'FALSE_POSITIVE') * (chact != ''))
    if verbose:
        print("correctly located: ", intop1)
        print("ratio:", intop1 / n)
        print("ratio of predicted positives:", np.sum(chpred == chact) / len(chact))
    _proba = lda2.predict_proba(features[test_idcs2])
    proba = np.transpose([ _proba[:, lda2.classes_==i][:, 0] if i in lda2.classes_ else np.full(len(_proba), 0.0) for i in range(len(class_names))])  ## log
    proba_sort = np.argsort(proba, axis=1)
    classes = class_names[proba_sort]
    intop2 = np.sum((chact != 'FALSE_POSITIVE') * (chact != '') * ((chact == classes[:, -1]) + (chact == classes[:, -2])))
    if verbose:
        print("class labels:", class_names)
        print("top2:", intop2);
        print("top2 ratio:", intop2 / n);
        for j in class_names:
            print("    ", j, ':', [np.sum(classes[chact == j, -1] == k) for k in class_names])
    intop3 = np.sum((chact != None) * (chact != '') * ((chact == classes[:, -1]) + (chact == classes[:, -2]) + (chact == classes[:, -3])))
    if verbose:
        print("top3:", intop3);
        print("top3 ratio:", intop3 / n);
        #for i in range(0, len(test_idcs2)):
        #    if (chact[i] != None):
        #        print(i, "act", chact[i], "pred", classes[i, -1], int(100*proba[i, proba_sort[i, -1]]), '|', classes[i, -2], int(100*proba[i, proba_sort[i, -2]]), '|', classes[i, -3], int(100*proba[i, proba_sort[i, -3]]), '||', chact[i], int(100*proba[i, class_names==chact[i]]), len(class_names) - np.arange(len(class_names))[classes[i, :] == chact[i]])
    return chpred, chact


def find_pin_lda(features, labels, ch_labels, jsons, json_id, class_names, ch_features=None, max_guess=200, tp_prob_power=1, num_digits=None, num_threads=4, dictionary=None, verbose=False, penalise_non_chosen=False, return_probs=False, ngram_model=None):
    features = np.array(features)
    shp = np.shape(features)
    if len(shp) > 2:
        features = np.reshape(features, (shp[0], int(np.product(np.array(shp)[1:]))))
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
    #for fold in range(10):
    def qput(Q, v):
        if num_threads == 1:
            Q.append(v)
        else:
            Q.put(v)
    def validate_fold(fold, guess_numbers):
        R = None
        if return_probs:
            R = guess_numbers[1]
            guess_numbers = guess_numbers[0]
        if verbose:
            print('fold', fold)
        train_json_idx = np.concatenate((np.arange(int(fold / 10 * jl)), np.arange(int((fold+1) / 10 * jl), jl)))
        test_json_idx = np.arange(int(fold / 10 * jl), int((fold+1) / 10 * jl))
        train_idcs = np.isin(json_id, json_id_fully_syncd[train_json_idx]) + np.isin(json_id, uniq_json_id[json_idcs_fully_syncd == False]) # bool array
        test_idcs = np.isin(json_id, json_id_fully_syncd[test_json_idx]) # bool array
        if np.all(labels[train_idcs == 1]):
            pred_probs = np.array([[-np.inf, 0]] * np.sum(test_idcs))
        else:
            lda = LinearDiscriminantAnalysis()
            lda.fit(features[train_idcs], labels[train_idcs])
            pred_probs = lda.predict_log_proba(features[test_idcs])
        #pred_probs = softmax(pred_probs, axis=1)
        act=labels[test_idcs]
        train_idcs2 = train_idcs * (labels == 1)    # bool array
        test_idcs2 = test_idcs # bool array
        lda = LinearDiscriminantAnalysis()
        lda.fit(ch_features[train_idcs2], ch_labels[train_idcs2])
        _proba = lda.predict_log_proba(ch_features[test_idcs2])
        proba = np.transpose([ _proba[:, lda.classes_==i][:, 0] if i in lda.classes_ else np.full(len(_proba), -np.inf) for i in range(len(class_names))])  ## log
        #proba = softmax(proba, axis=1)
        test2_labels = labels[test_idcs2]
        if return_probs:
            qput(R, (fold, pred_probs, proba, train_idcs, test_idcs, train_idcs2, test_idcs2, train_json_idx, test_json_idx))
        lik_correct_fp = 0
        ngram_str = '^^'
        for j in json_id_fully_syncd[test_json_idx]:
            jidcs = json_id[test_idcs2] == j
            jn = np.sum(jidcs)
            act_tp = test2_labels[jidcs]
            tap_chs = [ tap.ch for tap in jsons[j].taps_syncd ]
            if np.sum(act_tp) < len(jsons[j].taps_syncd):
                guess_numbers += [max_guess+2]
                continue
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
            j_tp_probs = (pred_probs[jidcs]) * tp_prob_power ## log
            j_proba = (proba[jidcs])  ## log
            lik_correct_tap = np.sum(j_tp_probs[act_tp==1, 1])  ## log
            lik_correct_class = np.sum([ j_proba[act_tp==1][i, class_names==ch] for i, ch in enumerate(tap_chs)])   ## log
            lik_correct = lik_correct_tap + lik_correct_class   ## log
            if penalise_non_chosen:
                lik_correct_fp = np.sum(j_tp_probs[act_tp==0, 0])  ## log
                lik_correct = np.add(lik_correct, lik_correct_fp)  ## log
            lik_ngram = 0
            if ngram_model is not None:
                ngram_str = '^' * ngram_model['__len_ngrams__']
                ngram = ngram_str
                for ch in tap_chs:
                    ngram = ngram[1:]
                    if ngram in ngram_model.keys():
                        m = ngram_model[ngram]
                        if ch in m.keys():
                            lik_ngram += np.log(m[ch])
                    ngram += ch
                lik_correct += lik_ngram
            prob_sorted_per_cand = np.argsort(j_proba, axis=1)[:, ::-1]
            most_likely_per_cand = j_tp_probs[:, 1] + np.max(j_proba, axis=1)   ## log
            if penalise_non_chosen:
                most_likely_per_cand = most_likely_per_cand - j_tp_probs[:, 0]
            cand_order = np.argsort(most_likely_per_cand)[::-1]
            fp_lik_after = [ np.sum(j_tp_probs[cand_order[i:], 0]) for i in range(jn+1) ]
            if verbose:
                print('likelyhood of correct class:', lik_correct, '=', lik_correct_fp,'+',lik_correct_tap,'+',lik_correct_class,'+',lik_ngram)
                print(np.sum(j_tp_probs[act_tp==1, 1] > j_tp_probs[act_tp==1, 0]), 'of', np.sum(act_tp==1), 'taps correctly detected')
            def count_lik_correct_alternative(num_left_to_select, i, p):
                print('    count_lik_correct_alternative', i, p)
                if num_left_to_select == 0:
                    return np.add(p, fp_lik_after[i])
                if i == jn:
                    return p
                current_cand = cand_order[i]
                if act_tp[current_cand]:
                    k = np.where(class_names == tap_chs[np.sum(act_tp[:current_cand])])[0][0]
                    print('        ', k, current_cand)
                    return count_lik_correct_alternative(num_left_to_select - 1, i+1, p + j_tp_probs[current_cand, 1] + j_proba[current_cand, k] + 0.0)
                else:
                    p0 = p
                    if penalise_non_chosen:
                        p0 = np.add(p, j_tp_probs[current_cand, 0])
                    return count_lik_correct_alternative(num_left_to_select, i+1, p0)
            def count_more_likely(num_left_to_select, i, p, ngram, acc):
                #print('count_more_likely', num_left_to_select, i, jn, p, fp_lik_after[i], lik_correct, acc)
                if p < lik_correct:
                    return acc
                if num_left_to_select == 0:
                    p1 = p
                    if penalise_non_chosen:
                        p1 = np.add(p, fp_lik_after[i])
                    return acc + (0 if p1 < lik_correct else 1)
                if i >= jn:
                    return acc
                r = ngram_model is not None
                current_cand = cand_order[i]
                prob_sorted = prob_sorted_per_cand[current_cand]
                ngps = np.zeros(len(class_names))
                if ngram_model is not None and ngram in ngram_model.keys():
                    current_cand = i
                    m = ngram_model[ngram]
                    ngps = np.log([m[ch] for ch in class_names])
                    prob_sorted = np.argsort(j_proba[i] + ngps)[::-1]
                for k in prob_sorted:
                    s = count_more_likely(num_left_to_select - 1, i+1, p + j_tp_probs[current_cand, 1] + j_proba[current_cand, k] + ngps[k], ngram[1:] + class_names[k], acc)   ## log
                    if s == acc:
                        break
                    r = True
                    acc = s
                    if acc >= max_guess:
                        return max_guess + 1
                if r and not np.all(labels==1):
                    p0 = p
                    if penalise_non_chosen:
                        p0 = np.add(p, j_tp_probs[current_cand, 0])
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
                    lik_ch = np.sum([j_proba[ca, class_names==ch] for ca, ch in zip(cands, word)])
                    lik_tp = np.sum(j_tp_probs[cands, 1])
                    lik_fp = np.sum(j_tp_probs[cands_fp, 0])
                    return np.sum((lik_ch, lik_fp, lik_tp))
                correct_prob = get_likelihood_of_word(string)
                gn = 0
                for w in dictionary:
                    if get_likelihood_of_word(w) > correct_prob:
                        gn += 1
                return gn
            if dictionary is not None:
                guess_number = dictionary_guessing()
            else:
                guess_number = count_more_likely(len(tap_chs), 0, 0.0, ngram_str[1:], 0) ## log
                if verbose:
                    print('guess:', guess_number)
                    #print('recompute lik_correct')
                lik_correct = count_lik_correct_alternative(len(tap_chs), 0, 0.0)
                guess_number = count_more_likely(len(tap_chs), 0, 0.0, ngram_str[1:], 0) ## log
                if verbose:
                    print('lik_correct: ', lik_correct)
                    #print('guess:', guess_number)
            if verbose:
                print('guess:', guess_number)
            qput(guess_numbers, guess_number)
    def validate_folds(folds, guess_numbers):
        try:
            for f in folds:
                validate_fold(f, guess_numbers)
        finally:
            if return_probs:
                Q, R = guess_numbers
                Q.put(None)
                R.put(None)
            else:
                guess_numbers.put(None)
    if num_threads == 1:
        Q = []
        R = []
        if return_probs:
            Q = (Q, R)
        for f in range(10):
            validate_fold(f, Q)
        return Q
    Q = Queue()
    R = None
    guess_numbers = Q
    if return_probs:
        R = Queue()
        guess_numbers = (Q, R)
    f0 = Process(target=validate_folds, args=(range(0, 3), guess_numbers))
    f1 = Process(target=validate_folds, args=(range(3, 6), guess_numbers))
    f2 = Process(target=validate_folds, args=(range(6, 8), guess_numbers))
    f3 = Process(target=validate_folds, args=(range(8, 10), guess_numbers))
    f0.start()
    f1.start()
    f2.start()
    f3.start()
    q = []
    r = []
    q_done = 0
    r_done = 0
    while q_done < 4 or (return_probs and r_done < 4):
        while not Q.empty():
            _q = Q.get()
            if _q is not None: q += [_q]
            else: q_done += 1
        if return_probs:
            while not R.empty():
                _r = R.get()
                if _r is not None: r += [_r]
                else: r_done += 1
        sleep(0.01)
    f0.join()
    f1.join()
    f2.join()
    f3.join()
    if return_probs:
        return q, r
    return q
            

