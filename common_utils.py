#! /usr/bin/env python

from json import load as json_load
from collections import namedtuple
from types import SimpleNamespace
from math import cos, sin, pi, sqrt
from os.path import join, isdir, isfile
from os import listdir
import numpy as np
from gc import collect as __gc_collect

def get_jsons():
    base_dir = '/home/zarandy/Documents/sound-samples/sync/'
    jsons = []
    for f in listdir(base_dir):
        d = join(base_dir, f)
        if (isdir(d)):
            json_file = join(d, "syncd.json")
            if (isfile(json_file)):
                jsons += [open_json(json_file)]
    return jsons

def open_json(filename):
    """
    Opens a JSON file
    """
    with open(filename) as f:
        def objhook(d):
            try:
                x = SimpleNamespace()
                for k, v in d.items():
                    setattr(x, k, v)
                return x
            except:
                return d
        return json_load(f, object_hook=objhook)

radius_cm = 4.63
speed_of_sound_cm_per_sec = 34029.0
r06 = range(0, 6)
mic_coordinates = np.array([ [cos((2.-k)*pi/3), sin((2.-k)*pi/3), 0] for k in r06]) * radius_cm

def crossval_lambda(traintest_fn):
    """
    Performs 10-fold cross-evaluation of Convolutional neural network classifier that discriminates between real taps and false positives.
    """
    tp = 0
    total = 0
    for i in range(0, 10):
        pred, act = traintest_fn(i)
        n = len(pred)
        ctp = (pred == act).sum()
        tp += ctp
        total += n
        __gc_collect()
    print('Crossval Accuracy:', tp/total)

def gen_ngrams(n, alphabet=None, init=0):
    ngram = {}
    ngram['__len_ngrams__'] = n
    with open('/usr/share/dict/words') as f:
        for w in f:
            s = w.rstrip('\n').lower()
            p = '^' * n
            for c in s + ('$'*n):
                p = p[1:]
                if p not in ngram.keys():
                    ngram[p] = {}
                    if alphabet is not None:
                        ngram[p][''] = init
                        ngram[p]['FALSE_POSITIVE'] = init
                        for a in alphabet:
                            ngram[p][a] = init
                m = ngram[p]
                if c not in m.keys():
                    m[c] = init
                m[c] += 1
                p += c
    for k in ngram:
        if k != '__len_ngrams__':
            t = 0
            m = ngram[k]
            for l in m:
                t += m[l]
            for l in m:
                m[l] /= t
    return ngram


