#! /usr/bin/env python

from utils import *
import sys
from os.path import join
from scipy.io import wavfile
import matplotlib
matplotlib.use('pgf')
from matplotlib import pyplot as plt


base_dir = '/home/zarandy/Documents/sound-samples/sync/'

out_filename = sys.argv[1]
args = out_filename.split('-')
fig_type = args[0]
recordingId = args[1]

json = open_json(join(base_dir, recordingId, 'syncd.json'))
out_file = join('AlmosPartIII', 'figs-gen', out_filename)

if fig_type == 'waveform':
    fs, x = wavfile.read(json.wav)
    fss, y = wavfile.read(json.self_recording_wav)
    assert fs == fss
    offset = int(args[2])
    length = int(args[3])

    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axs[0].set(ylim=(-200, 200))
    axs[0].plot(x[offset:(offset+length), 4], c='red', label='External microphone')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[0].legend()
    axs[1].set(ylim=(-500, 500))
    axs[1].plot(y[offset:(offset+length), 0], c='green', label='Internal microphone')
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[1].legend()
    plt.savefig(out_file)

if fig_type == 'specgram':
    fs, x = wavfile.read(json.wav)
    fss, y = wavfile.read(json.self_recording_wav)
    offset = int(args[2])
    length = int(args[3])

    fig, axs = plt.subplots(2, 1, figsize=(1.5, 5), sharex=True)
    plt.subplots_adjust(left=0.3)
    axs[0].specgram(x[offset:(offset+length), 4], NFFT=128, Fs=fs, noverlap=96)
    axs[0].set_xticklabels([])
    axs[0].set(ylim=(0, 8000))
    axs[1].specgram(y[offset:(offset+length), 0], NFFT=128, Fs=fss, noverlap=96)
    axs[1].set_xticklabels([])
    axs[1].set(ylim=(0, 8000))
    plt.savefig(out_file)

if fig_type == 'specgramforcepgf':
    fs, x = wavfile.read(json.wav)
    fss, y = wavfile.read(json.self_recording_wav)
    offset = int(args[2])
    length = int(args[3])

    fig, axs = plt.subplots(2, 1, figsize=(1.5, 5), sharex=True)
    plt.subplots_adjust(left=0.3)
    axs[0].specgram(x[offset:(offset+length), 4], NFFT=128, Fs=fs, noverlap=96)
    axs[0].set_xticklabels([])
    axs[0].set(ylim=(0, 8000))
    axs[1].specgram(y[offset:(offset+length), 0], NFFT=128, Fs=fss, noverlap=96)
    axs[1].set_xticklabels([])
    axs[1].set(ylim=(0, 8000))
    plt.savefig(out_file)

