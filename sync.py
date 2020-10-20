#! /usr/bin/env python

import sys
import json
from collections import namedtuple
import numpy as np
from scipy.io import wavfile
from math import cos, pi
from scipy.signal import correlate, stft
from types import SimpleNamespace

from common_utils import *
from sync_utils import *

json_filename = sys.argv[1]

recording = open_json(json_filename)

json_dir = json_filename.rpartition('/')[0]
wav_filename = recording.recordingRemoteFileName.rpartition('/')[-1]
fs, wav = wavfile.read(json_dir + '/' + wav_filename)
pcm_filename = recording.recordingLocalFilename.rpartition('/')[-1]
pcm = np.fromfile(json_dir + '/' + pcm_filename, dtype=(np.int16, recording.recordingNumChannels))

recording.wav = json_dir + '/' + wav_filename
recording.original_json = json_filename
recording.dir = json_dir




print(recording.recordingId)
print(np.shape(wav), fs)
offset_wav = find_sync_sequence(wav[:, 0], fs)
offset_pcm = find_sync_sequence(pcm[:, 0], recording.recordingSamplerate)
offset_diff = int((offset_pcm / recording.recordingSamplerate - offset_wav / fs) * recording.recordingSamplerate)
print(offset_wav)
print(offset_pcm)
print(offset_diff)
print(recording.recordingNumChannels)
if (offset_diff > 0):
    pcm_delayed = pcm[offset_diff:, :]
else:
    delay = np.zeros([-offset_diff, recording.recordingNumChannels], dtype=np.int16)
    pcm_delayed = np.concatenate((delay, pcm))

wavfile.write(json_dir + '/' + pcm_filename.rpartition('.')[0] + '.wav', recording.recordingSamplerate, pcm_delayed)
recording.self_recording_wav = json_dir + '/' + pcm_filename.rpartition('.')[0] + '.wav'
recording.offset_diff = offset_diff


f, t, Zxx = stft(pcm_delayed[:, 1] + pcm_delayed[:, 0], recording.recordingSamplerate, nperseg=128, noverlap=96)
fc, l = Zxx.shape
Zxxa = np.absolute(Zxx)
tap_cands = find_tap_cands(Zxxa, t, l, 50, verbose=False)
candidate_label_pairs, __ = sync_recording_to_labels(tap_cands, recording.taps, time_scale = t[1] * 1000)
for i in candidate_label_pairs:
    print(t[tap_cands[i][0]])

recording.taps_syncd_to_self_recording = []
for j, i in enumerate(candidate_label_pairs):
    tap = recording.taps[j]
    time_samples = tap_cands[i][0] * 32
    time = t[tap_cands[i][0]]
    ch = ''
    if hasattr(tap, 'ch'):
        ch = tap.ch
    recording.taps_syncd_to_self_recording += [{ 'time_seconds': time, 'time_samples': time_samples, 'x': tap.x, 'y': tap.y, 'ch': ch }]

lbl_filename = json_dir + '/' + pcm_filename.rpartition('.')[0] + '.txt'
recording.self_recording_label_filename = lbl_filename
with open(lbl_filename, mode='w') as f:
    for j, i in enumerate(candidate_label_pairs):
        if i != -1:
            tap = recording.taps[j]
            time = t[tap_cands[i][0]]
            ch = ''
            if hasattr(tap, 'ch'):
                ch = tap.ch
            f.write('%f\t%f\t%s(%d,%d)\n' % ( time, time + 0.01, ch, tap.x, tap.y))


print('-----------------')


f, t, Zxx = stft(wav, fs, nperseg=128, noverlap=96, axis=0)
fc, nc, l = Zxx.shape
assert(nc == 6)
Zxxa = np.absolute(Zxx)
tap_cands_in_pi_recording = None
candidate_label_pairs = None
score_best = 0
channel = 0
for i in [4]:
    tap_cands_in_pi_recording_current = find_tap_cands(Zxxa[:, i, :], t, l, 30, verbose=False)
    candidate_label_pairs_current, score = sync_recording_to_labels(tap_cands_in_pi_recording_current, recording.taps, time_scale = t[1] * 1000)
    if score > score_best:
        score_best = score
        tap_cands_in_pi_recording = tap_cands_in_pi_recording_current
        candidate_label_pairs = candidate_label_pairs_current
        channel = 4
    print('score on channel', i, ':', score)
    found = 0
    for tap in candidate_label_pairs_current:
        for s in recording.taps_syncd_to_self_recording:
            if abs(s['time_samples'] - tap_cands_in_pi_recording_current[tap][0] * 32) < 512:
                found += 1
                break
    if found * 2 <= len(candidate_label_pairs_current) + 1:
        print(i, ': \033[31;1mmisaligned\033[0m (', found, ')')
    for j in candidate_label_pairs_current:
        print(t[tap_cands_in_pi_recording_current[j][0]])

recording.taps_syncd = []
recording.channel_used_to_sync = channel
recording.sync_score = score_best
for j, i in enumerate(candidate_label_pairs):
    tap = recording.taps[j]
    time_samples = tap_cands_in_pi_recording[i][0] * 32
    time = t[tap_cands_in_pi_recording[i][0]]
    ch = ''
    if hasattr(tap, 'ch'):
        ch = tap.ch
    recording.taps_syncd += [{ 'time_seconds': time, 'time_samples': time_samples, 'x': tap.x, 'y': tap.y, 'ch': ch }]

lbl_filename = json_dir + '/' + wav_filename.rpartition('.')[0] + '.txt'
recording.label_filename = lbl_filename
with open(lbl_filename, mode='w') as f:
    for j, i in enumerate(candidate_label_pairs):
        if i != -1:
            tap = recording.taps[j]
            time = t[tap_cands_in_pi_recording[i][0]]
            ch = ''
            if hasattr(tap, 'ch'):
                ch = tap.ch
            f.write('%f\t%f\t%s(%d,%d)\n' % ( time, time + 0.01, ch, tap.x, tap.y))


with open(json_dir+'/syncd.json', 'w') as f:
    json.dump(recording, f, default=lambda x: x.__dict__)
        
        

        
