#! /usr/bin/env python

import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import correlate
from scipy.optimize import least_squares, fmin
from itertools import product
from common_utils import *
from numpy.linalg import inv

def ir(x):
    """
    Round to nearest int
    """
    return int(round(x))

def abs_corr(a, b):
    """
    Returns the absolute value of cross-correlation
    """
    return np.abs(correlate(a, b))

def hpass2000(Y1, Y2, fs):
    """
    Frequency weighting function of an ideal high-pass filter with threshold frequency 2000Hz
    """
    n = len(Y1)
    return np.array([ 1 if i>2000.0/fs*n else 0 for i in range(0, n) ])

def hpass(Y1, Y2, fs):
    """
    Frequency weighting function of an ideal high-pass filter with threshold frequency 2500Hz
    """
    n = len(Y1)
    return np.array([ 1 if i>2500.0/fs*n else 0 for i in range(0, n) ])

def hpass3500(Y1, Y2, fs):
    """
    Frequency weighting function of an ideal high-pass filter with threshold frequency 3500Hz
    """
    n = len(Y1)
    return np.array([ 1 if i>3500.0/fs*n else 0 for i in range(0, n) ])

def hpass4500(Y1, Y2, fs):
    """
    Frequency weighting function of an ideal high-pass filter with threshold frequency 4500Hz
    """
    n = len(Y1)
    return np.array([ 1 if i>4500.0/fs*n else 0 for i in range(0, n) ])

def hpassf(f):
    def _hpass(Y1, Y2, fs):
        n = len(Y1)
        return np.array([ 1 if i>f/fs*n else 0 for i in range(0, n) ])
    return _hpass

def sig_coherence(Y1, Y2):
    """
    Returns the coherence function of two signals given their spectrum
    """
    nominator = Y1 * np.conj(Y2)
    denominator = np.conj(Y1) * Y2
    denominator[denominator == 0] = 1e-16
    return nominator / denominator


def eckart_hpass(Y1, Y2, fs):
    """
    ECKART frequency weighting function with high-pass filter
    """
    n = len(Y1)
    G_y1y2_a = np.abs(Y1 * np.conj(Y2))
    denominator = (Y1 * np.conj(Y1) - G_y1y2_a) * (Y2 * np.conj(Y2) - G_y1y2_a)
    denominator[denominator == 0] = 1e-16
    Ψ = G_y1y2_a / denominator
    return np.array([ Ψ[k] if k>2500.0/fs*n else 0 for k in range(0, n) ])

def roth_hpass(Y1, Y2, fs):
    """
    ROTH frequency weighting function with high-pass filter
    """
    n = len(Y1)
    denominator = Y1 * np.conj(Y1)
    denominator[denominator == 0] = 1e-16
    result = 1. / denominator
    result[:int(2500./fs*n)] = 0
    return result

def phat_hpass(Y1, Y2, fs):
    """
    PHAT frequency weighting function with high-pass filter
    """
    n = len(Y1)
    phat = Y1 * np.conj(Y2)
    phat[phat==0] = 1e-16
    phat = 1 / phat
    phat[:int(2500./fs*n)] = 0
    return phat

def gcc(Ψ, fs):
    """
    generalised cross-correlation
    """
    def gcc_fn(y1, y2):
        Y1 = rfft(np.pad(y1, (0, len(y1))))
        Y2 = rfft(np.pad(y2, (len(y2), 0)))
        G = Y1 * np.conj(Y2) * Ψ(Y1, Y2, fs)
        return irfft(G)[1:]
    return gcc_fn

def compute_TDoA(x, start, end, ref_ci, corr_fn, max_diff=30):
    """
    Computes Time Difference of Arrival between one microphone and all other microphones
    """
    n = end - start
    d = min(max_diff, n)
    correlations = [ corr_fn(x[start:end, i], x[start:end, ref_ci])[(n-d):(n+d)] for i in r06 ]
    return np.argmax(correlations, axis=1) - d + 1

def compute_TDoA_longer_filter(x, start, end, ref_ci, corr_fn, fs, filt_start, filt_end, threshold_freq, max_diff=30):
    """
    Computes Time Difference of Arrival between one microphone and all other microphones
    High-pass filters a longer sequence of the signal
    """
    X = rfft(x[filt_start:filt_end, :], axis=0)
    f = int(threshold_freq / fs * (filt_end - filt_start) / 2)
    X[:f, :] = 0
    y = irfft(X, axis=0)
    return compute_TDoA(y, start - filt_start, end - filt_start, ref_ci, corr_fn, max_diff)

def compute_TDoA_cancel_noise(x, start, end, ref_ci, corr_fn, no_signal_start, no_signal_end, max_diff=30):
    n = end - start
    d = min(max_diff, n)
    correlations = [ corr_fn(x[start:end, i], x[start:end, ref_ci])[(n-d):(n+d)] for i in r06 ]
    nn = no_signal_end - no_signal_start
    dn = min(max_diff, nn)
    ncorrelations = [ corr_fn(x[no_signal_start:no_signal_end, i], x[no_signal_start:no_signal_end, ref_ci])[(nn-dn):(nn+dn)] for i in r06 ]
    cdiff = np.array(correlations) - np.array(ncorrelations)
    return np.argmax(cdiff, axis=1) - d + 1


def compute_corr_pw(x, start, end, corr_fn, max_diff=30):
    """
    Computes correlation between all pairs of microphones
    """
    n = end - start
    d = min(max_diff, n)
    correlations = [[ corr_fn(x[start:end, i], x[start:end, j])[(n-d):(n+d)] for i in r06 ] for j in r06 ]
    return np.array(correlations)

def compute_TDoA_pw(x, start, end, corr_fn, max_diff=30):
    """
    Computes correlation between all pairs of microphones
    """
    n = end - start
    d = min(max_diff, n)
    correlations = [[ corr_fn(x[start:end, i], x[start:end, j])[(n-d):(n+d)] for i in r06 ] for j in r06 ]
    return np.argmax(correlations, axis=2) - d + 1

def compute_position_lse(arrivals, fs):
    """
    Compute position using least-squares error in time differences of arrival
    """
    diffs = (arrivals - np.average(arrivals)) * speed_of_sound_cm_per_sec / fs
    x0 = np.array([0, -8, .1])
    def residuals(u):
        n = np.linalg.norm(u - mic_coordinates, axis=1)
        return n - np.average(n) - diffs
    def jacobian(u):
        jd = np.array([ (u - s_i) / np.linalg.norm(u - s_i) for s_i in mic_coordinates ])
        return jd - np.average(jd, axis=0)
    p = least_squares(residuals, x0, jac=jacobian)
    return p

def compute_position_pw(crosscorr, u0):
    """
    Compute position by maximising pairwise cross-correlation
    """
    __, ___, m = np.shape(crosscorr)
    mid = int(- (m+1) / 2)
    def score_pw(u):
        n = np.linalg.norm(u - mic_coordinates, axis=1)
        r = 0.0
        for i, j in product(r06, r06):
            d = int(n[i] - n[j])
            r += crosscorr[i, j, mid+d]
        return r
    p = fmin(lambda u: -score_pw(u), u0, disp=False)
    return p

def compute_position_subsample(x, start, u0, length):
    """
    Compute position by maximising delayed sum of square of sums of signal with sum-sample accuracy.
    (Carter 1987, fig 21)
    """
    def score(u):
        dists = np.linalg.norm(u - mic_coordinates, axis=1) * fs / speed_of_sound_cm_per_sec
        delays = dists - np.average(dists)
        end = start + length
        #translated = np.array([ x[(start+ir(dists[i])):(end+ir(dists[i])), i] for i in r06 ])
        translated = np.array([ delayseq(x[(start+int(dists[i])):(end+int(dists[i])), i], float(-dists[i]) % 1, 1) for i in r06 ])
        total = np.sum(translated, axis = 0)
        autocorr = corr_fn(total, total)
        return autocorr[int(len(autocorr) / 2)]
    r = fmin(lambda u: -score(u), u0, disp=False)
    return r
   
def compute_position_sa_sim(D, ref_ci):
    """
    Compute position of source using Smith and Abel's spherical interpolation method
    """
    N = len(D)
    R = (mic_coordinates - mic_coordinates[ref_ci])[:, :2]
    S = np.concatenate((R[:ref_ci], R[(ref_ci+1):]))
    dd = np.concatenate((D[:ref_ci], D[(ref_ci+1):]))[:, np.newaxis] - D[ref_ci]
    δ = np.linalg.norm(S, keepdims=True, axis=1)**2 - dd**2
    W = np.identity(N-1)
    V = np.identity(N-1)
    if np.all(W == V):
        Pd_bot = np.identity(N-1) - (dd @ dd.T) / (dd.T @ dd)
        result = 0.5 * inv(S.T @ Pd_bot @ W @ Pd_bot @ S) @ S.T @ Pd_bot @ W @ Pd_bot @ δ
    else:
        Sw = inv(S.T @ W @ S) @ S.T @ W
        Ps = S @ Sw
        Ps_bot = np.identity(N-1) - Ps
        Rs = (dd.t @ Ps_bot @ V @ Ps_bot @ δ) / (2 * dd.T @ Ps_bot @ V @ Ps_bot @ dd)
        result = 0.5 * Sw @ (δ - 2 * Rs * dd)
    return result + mic_coordinates[ref_ci, :2, np.newaxis]

def compute_DOA_azimuth(dd):
    def err(φ):
        return (np.array([[
            cos(φ) * (mic_coordinates[i, 0] - mic_coordinates[j, 0]) +
            sin(φ) * (mic_coordinates[i, 1] - mic_coordinates[j, 1])
            for j in r06] for i in r06]) - dd).flatten()
    return least_squares(err, -pi/2).x

def compute_TDoA_valid_ranges_only(x, fs, ts):
    fgcc = gcc(hpassf(3000), fs)
    score = 0
    start_ts = 0
    for tsd in range(ts-128, min(ts+128, len(x)-64), 16):
        dcd = compute_corr_pw(x, tsd, tsd+64, fgcc, max_diff=12)
        if dcd[4, 4, 11] > score:
            score = dcd[4, 4, 11]
            start_ts = tsd
    dc = compute_corr_pw(x, start_ts, start_ts+64, fgcc, max_diff=12)
    dd = np.array([
        [
            0,
            np.argmax(dc[0, 1, 8:15]) - 3,
            np.argmax(dc[0, 2, 5:14]) - 6,
            np.argmax(dc[0, 3, 0:12]) - 11,
            np.argmax(dc[0, 4, 0:12]) - 11,
            np.argmax(dc[0, 5, 5:12]) - 6
            ],
        [
            np.argmax(dc[1, 0, 8:15]) - 3,
            0,
            np.argmax(dc[1, 2, 5:12]) - 6,
            np.argmax(dc[1, 3, 0:12]) - 11,
            np.argmax(dc[1, 4, 0:12]) - 11,
            np.argmax(dc[1, 5, 5:14]) - 6,
            ],
        [
            np.argmax(dc[2, 0, 9:19]) - 2,
            np.argmax(dc[2, 1, 11:18]),
            0,
            np.argmax(dc[2, 3, 5:12]) - 6,
            np.argmax(dc[2, 4, 2:12]) - 9,
            np.argmax(dc[2, 5, 3:20]) - 8
            ],
        [
            np.argmax(dc[3, 0, 11:23]),
            np.argmax(dc[3, 1, 11:23]),
            np.argmax(dc[3, 2, 11:18]),
            0,
            np.argmax(dc[3, 4, 5:18]) - 6,
            np.argmax(dc[3, 5, 11:21])
            ],
        [
            np.argmax(dc[4, 0, 11:23]),
            np.argmax(dc[4, 1, 11:23]),
            np.argmax(dc[4, 2, 11:21]),
            np.argmax(dc[4, 3, 5:18]) - 6,
            0,
            np.argmax(dc[4, 5, 11:18])
            ],
        [
            np.argmax(dc[5, 0, 11:18]),
            np.argmax(dc[5, 1, 9:19]) - 2,
            np.argmax(dc[5, 2, 3:20]) - 8,
            np.argmax(dc[5, 3, 2:12]) - 9,
            np.argmax(dc[5, 4, 5:12]) - 6,
            0
            ]
        ])
    return dd
