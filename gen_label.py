#! /usr/bin/env python

import sys
import json
from collections import namedtuple
import numpy as np
from scipy.io import wavfile
from math import cos, pi
from scipy.signal import correlate, stft


json_filename = sys.argv[1]
delay = 0
if len(sys.argv) >= 3:
    delay = float(sys.argv[2])

with open(json_filename) as f:
    def objhook(d):
        try:
            return namedtuple('X', d.keys())(*d.values())
        except:
            return d
    recording = json.load(f, object_hook=objhook)

with open(json_filename.rpartition('.')[0] + '.txt', mode='w') as f:
    for t in recording.taps:
        ch = ''
        if hasattr(t, 'ch'):
            ch = t.ch
        f.write('%f\t%f\t%s(%d,%d)\n' % ( t.timestamp / 1000 + delay, (t.timestamp + t.duration) / 1000 + delay, ch, t.x, t.y))


