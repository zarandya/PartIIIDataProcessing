#! /usr/bin/env python

"""
Jobs using LDA to find false positives and classify keystrokes on
"""
#experimentName = 'atnPin'

from types import SimpleNamespace
from librosa.feature import mfcc
#import utils as U

jsons = []
audio_in = []
tap_cands_in_jsons = []
U = None
batch = None

tap_cand_waveforms = []
tap_cand_mfcc_features = []
tap_cand_mfcc_6mic_all_features = []
tap_cand_mfcc_6mic_coarse_features = []
tap_cand_mfcc_6mic_sanefakefs_features = []
tap_cand_mfcc_6mic_quarterfs_features = []
tap_cand_mfcc_6mic_sanefakelongseg_features = []
tap_cand_mfcc_6mic_sanefakeshortseg_features = []
tap_cand_mfcc_6mic_02fs_features = []
tap_cand_mfcc_6mic_03fs_features = []
tap_cand_mfcc_6mic_04fs_features = []
tap_cand_mfcc_6mic_05fs_features = []
tap_cand_mfcc_6mic_06fs_features = []
tap_cand_mfcc_6mic_07fs_features = []
tap_cand_mfcc_6mic_08fs_features = []
tap_cand_mfcc_6mic_09fs_features = []
tap_cand_mfcc_6mic_10fs_features = []
tap_cand_mfcc_6mic_11fs_features = []
tap_cand_mfcc_6mic_12fs_features = []
tap_cand_mfcc_6mic_13fs_features = []
tap_cand_mfcc_6mic_14fs_features = []
tap_cand_mfcc_6mic_1024nf_features = []
tap_cand_mfcc_6mic_512nf_features = []
tap_cand_mfcc_6mic_256nf_features = []
tap_cand_mfcc_6mic_1024nf78fs_features = []
tap_realtap_mfcc_6mic_1024nf78fs_features = []
tap_realtap_mfcc_6mic_1024nf78fs_16000_features = []
tap_realtap_mfcc_6mic_1024nf78fs_8000_features = []
tap_realtap_mfcc_6mic_1024nf78fs_16000l_features = []
tap_realtap_mfcc_6mic_1024nf78fs_8000l_features = []
tap_realtap_direction_estimate = []
tap_realtap_tdoa_estimate = []
tap_cand_json_name = []
lang_model_3gram_0 = []
lang_model_3gram_1 = []
lang_model_8gram_0 = []
lang_model_8gram_1 = []
dictionary=[]
tap_cand_labels = []
tap_cand_ch = []

ch_class_names = None
positives = None
tap_mfcc_features = None
tap_mfcc_6mic_all_features = None
tap_mfcc_6mic_coarse_features = None
tap_ch = None
tap_labels = None

def dependencies(symbols):
    deps = ['tap_cand_waveforms']
    if 'tap_mfcc_features' in symbols:
        deps += ['tap_cand_mfcc_features']
    if 'tap_mfcc_6mic_all_features' in symbols:
        deps += ['tap_cand_mfcc_6mic_all_features']
    if 'tap_mfcc_6mic_coarse_features' in symbols:
        deps += ['tap_cand_mfcc_6mic_coarse_features']
    if 'tap_realtap_direction_estimate' in symbols:
        deps += ['tap_realtap_tdoa_estimate']
    return deps

def init(symbols = None):
    global tap_cand_ch
    global ch_class_names
    global positives
    global tap_mfcc_features
    global tap_mfcc_6mic_all_features
    global tap_mfcc_6mic_coarse_features
    global tap_ch
    global tap_labels
    global tap_realtap_direction_estimate
    ch_class_names_arr = sorted(U.np.unique(tap_cand_ch))
    ch_class_names = U.np.array(ch_class_names_arr)
    tap_cand_ch = U.np.array([ ch_class_names_arr.index(ch) for ch in tap_cand_ch ])
    positives = U.np.array(tap_cand_labels) == 1
    if symbols is None or 'tap_mfcc_features' in symbols:
        tap_mfcc_features = tap_cand_mfcc_features[positives]
    if symbols is None or 'tap_mfcc_6mic_all_features' in symbols:
        tap_mfcc_6mic_all_features = tap_cand_mfcc_6mic_all_features[positives]
    if symbols is None or 'tap_mfcc_6mic_coarse_features' in symbols:
        tap_mfcc_6mic_coarse_features = tap_cand_mfcc_6mic_coarse_features[positives]
    if 'tap_realtap_direction_estimate' in symbols:
        tap_realtap_direction_estimate = U.np.array([U.compute_DOA_azimuth(t) for t in tap_realtap_tdoa_estimate])
    tap_ch = U.np.array(tap_cand_ch)[positives]
    tap_labels = U.np.ones(dtype=int, shape=len(tap_ch))

    tap_waveforms = U.np.array(tap_cand_waveforms)[positives]

#def crossval_tpfp_1f_0():
#    """
#    def crossval_tpfp_1f_0():
#        U.crossval(tap_cand_mfcc_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_all_features[:, 4], tap_cand_labels, verbose=True)
#
#def crossval_tpfp_2af_0():
#    """
#    def crossval_tpfp_2af_0():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_all_features)
#        features = tap_cand_mfcc_6mic_all_features[:, g]
#        U.crossval(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_all_features)
#    features = tap_cand_mfcc_6mic_all_features[:, g]
#    U.crossval(features, tap_cand_labels, verbose=True)
#
#def crossval_tpfp_2bf_0():
#    """
#    def crossval_tpfp_2bf_0():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_all_features)
#        features = tap_cand_mfcc_6mic_all_features[:, g]
#        U.crossval(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_all_features)
#    features = tap_cand_mfcc_6mic_all_features[:, g]
#    U.crossval(features, tap_cand_labels, verbose=True)
#
#def crossval_tpfp_2cf_0():
#    """
#    def crossval_tpfp_2cf_0():
#        g = U.np.array([3, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_all_features)
#        features = tap_cand_mfcc_6mic_all_features[:, g]
#        U.crossval(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([3, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_all_features)
#    features = tap_cand_mfcc_6mic_all_features[:, g]
#    U.crossval(features, tap_cand_labels, verbose=True)
#
#def crossval_tpfp_6a_0():
#    """
#    def crossval_tpfp_6DTF_0():
#        U.crossval(tap_cand_mfcc_6mic_all_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_all_features, tap_cand_labels, verbose=True)

#def crossval_tpfp_1f_0():
#    """
#    def crossval_tpfp_1f_0():
#        U.crossval(tap_cand_mfcc_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_all_features[:, 4], tap_cand_labels, verbose=True)
#
#def crossval_tpfp_2afc_0():
#    """
#    def crossval_tpfp_2afc_0():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_coarse_features)
#        features = tap_cand_mfcc_6mic_coarse_features[:, g]
#        U.crossval(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_coarse_features)
#    features = tap_cand_mfcc_6mic_coarse_features[:, g]
#    U.crossval(features, tap_cand_labels, verbose=True)
#
#def crossval_tpfp_2bfc_0():
#    """
#    def crossval_tpfp_2bfc_0():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_coarse_features)
#        features = tap_cand_mfcc_6mic_coarse_features[:, g]
#        U.crossval(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_coarse_features)
#    features = tap_cand_mfcc_6mic_coarse_features[:, g]
#    U.crossval(features, tap_cand_labels, verbose=True)
#
#def crossval_tpfp_2cfc_0():
#    """
#    def crossval_tpfp_2cfc_0():
#        g = U.np.array([3, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_coarse_features)
#        features = tap_cand_mfcc_6mic_coarse_features[:, g]
#        U.crossval(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([3, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_mfcc_6mic_coarse_features)
#    features = tap_cand_mfcc_6mic_coarse_features[:, g]
#    U.crossval(features, tap_cand_labels, verbose=True)
#
# 
#def crossval_tpfp_6c_0():
#   """
#    def crossval_tpfp_6FTD_0():
#        U.crossval(tap_cand_mfcc_6mic_coarse_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_coarse_features, tap_cand_labels, verbose=True)

#def crossval_ch_1f_0():
#    """
#    def crossval_ch_1f_0():
#        f = lambda fold: U.traintest_ch(tap_mfcc_6mic_all_features[:, 4], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_mfcc_6mic_all_features[:, 4]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_2af_0():
#    """
#    def crossval_ch_2af_0():
#        g = U.np.array([1, 4])
#        features = tap_mfcc_6mic_all_features[:, g]
#        f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([1, 4])
#    features = tap_mfcc_6mic_all_features[:, g]
#    print(U.np.shape(features))
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_2bf_0():
#    """
#    def crossval_ch_2bf_0():
#        g = U.np.array([0, 3])
#        features = tap_mfcc_6mic_all_features[:, g]
#        f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([0, 3])
#    features = tap_mfcc_6mic_all_features[:, g]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_2cf_0():
#    """
#    def crossval_ch_2cf_0():
#        g = U.np.array([3, 4])
#        l, nd, nt, nf = U.np.shape(tap_mfcc_6mic_all_features)
#        features = tap_mfcc_6mic_all_features[:, g]
#        f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([3, 4])
#    l, nd, nt, nf = U.np.shape(tap_mfcc_6mic_all_features)
#    features = tap_mfcc_6mic_all_features[:, g]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_6a_0():
#    """
#    def crossval_ch_6a_0():
#        f = lambda fold: U.traintest_ch(tap_mfcc_6mic_all_features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_mfcc_6mic_all_features
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_6c_0():
#    """
#    def crossval_ch_6c_0():
#        f = lambda fold: U.traintest_ch(tap_mfcc_6mic_coarse_features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_mfcc_6mic_coarse_features
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_1f_0():
#    """
#    def crossval_ch_1f_0():
#        f = lambda fold: U.traintest_ch(tap_mfcc_6mic_coarse_features[:, 4], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_mfcc_6mic_coarse_features[:, 4]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_2afc_0():
#    """
#    def crossval_ch_2afc_0():
#        g = U.np.array([1, 4])
#        features = tap_mfcc_6mic_coarse_features[:, g]
#        f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([1, 4])
#    features = tap_mfcc_6mic_coarse_features[:, g]
#    print(U.np.shape(features))
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_2bfc_0():
#    """
#    def crossval_ch_2bfc_0():
#        g = U.np.array([0, 3])
#        features = tap_mfcc_6mic_coarse_features[:, g]
#        f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([0, 3])
#    features = tap_mfcc_6mic_coarse_features[:, g]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_2cfc_0():
#    """
#    def crossval_ch_2cfc_0():
#        g = U.np.array([3, 4])
#        l, nd, nt, nf = U.np.shape(tap_mfcc_6mic_coarse_features)
#        features = tap_mfcc_6mic_coarse_features[:, g]
#        f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([3, 4])
#    l, nd, nt, nf = U.np.shape(tap_mfcc_6mic_coarse_features)
#    features = tap_mfcc_6mic_coarse_features[:, g]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
 
#def crossval_tpfp_6_sanefakefs_0():
#    """
#    def crossval_tpfp_6_sanefakefs_0():
#        U.crossval(tap_cand_mfcc_6mic_sanefakefs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_sanefakefs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_sanefakefs_0():
#    """
#    def crossval_ch_6c_sanefakefs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_sanefakefs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_sanefakefs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_quarterfs_0():
#    """
#    def crossval_tpfp_6_quarterfs_0():
#        U.crossval(tap_cand_mfcc_6mic_quarterfs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_quarterfs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_quarterfs_0():
#    """
#    def crossval_ch_6c_quarterfs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_quarterfs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_quarterfs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_sanefakelongseg_0():
#    """
#    def crossval_tpfp_6_sanefakelongseg_0():
#        U.crossval(tap_cand_mfcc_6mic_sanefakelongseg_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_sanefakelongseg_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_sanefakelongseg_0():
#    """
#    def crossval_ch_6c_sanefakelongseg_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_sanefakelongseg_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_sanefakelongseg_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_sanefakeshortseg_0():
#    """
#    def crossval_tpfp_6_sanefakeshortseg_0():
#        U.crossval(tap_cand_mfcc_6mic_sanefakeshortseg_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_sanefakeshortseg_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_sanefakeshortseg_0():
#    """
#    def crossval_ch_6c_sanefakeshortseg_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_sanefakeshortseg_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_sanefakeshortseg_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)

#def crossval_tpfp_6_02fs_0():
#    """
#    def crossval_tpfp_6_02fs_0():
#        U.crossval(tap_cand_mfcc_6mic_02fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_02fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_02fs_0():
#    """
#    def crossval_ch_6c_02fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_02fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_02fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_03fs_0():
#    """
#    def crossval_tpfp_6_03fs_0():
#        U.crossval(tap_cand_mfcc_6mic_03fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_03fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_03fs_0():
#    """
#    def crossval_ch_6c_03fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_03fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_03fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_04fs_0():
#    """
#    def crossval_tpfp_6_04fs_0():
#        U.crossval(tap_cand_mfcc_6mic_04fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_04fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_04fs_0():
#    """
#    def crossval_ch_6c_04fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_04fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_04fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_05fs_0():
#    """
#    def crossval_tpfp_6_05fs_0():
#        U.crossval(tap_cand_mfcc_6mic_05fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_05fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_05fs_0():
#    """
#    def crossval_ch_6c_05fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_05fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_05fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_06fs_0():
#    """
#    def crossval_tpfp_6_06fs_0():
#        U.crossval(tap_cand_mfcc_6mic_06fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_06fs_features, tap_cand_labels, verbose=True)
#
def crossval_ch_6_06fs_0():
    """
    def crossval_ch_6c_06fs_0():
        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_06fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
        U.crossval_lambda(f)
    """
    features = tap_cand_mfcc_6mic_06fs_features[positives]
    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)
#
#def crossval_tpfp_6_07fs_0():
#    """
#    def crossval_tpfp_6_07fs_0():
#        U.crossval(tap_cand_mfcc_6mic_07fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_07fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_07fs_0():
#    """
#    def crossval_ch_6c_07fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_07fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_07fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_08fs_0():
#    """
#    def crossval_tpfp_6_08fs_0():
#        U.crossval(tap_cand_mfcc_6mic_08fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_08fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_08fs_0():
#    """
#    def crossval_ch_6c_08fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_08fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_08fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_09fs_0():
#    """
#    def crossval_tpfp_6_09fs_0():
#        U.crossval(tap_cand_mfcc_6mic_09fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_09fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_09fs_0():
#    """
#    def crossval_ch_6c_09fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_09fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_09fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_10fs_0():
#    """
#    def crossval_tpfp_6_10fs_0():
#        U.crossval(tap_cand_mfcc_6mic_10fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_10fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_10fs_0():
#    """
#    def crossval_ch_6c_10fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_10fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_10fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_11fs_0():
#    """
#    def crossval_tpfp_6_11fs_0():
#        U.crossval(tap_cand_mfcc_6mic_11fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_11fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_11fs_0():
#    """
#    def crossval_ch_6c_11fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_11fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_11fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_12fs_0():
#    """
#    def crossval_tpfp_6_12fs_0():
#        U.crossval(tap_cand_mfcc_6mic_12fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_12fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_12fs_0():
#    """
#    def crossval_ch_6c_12fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_12fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_12fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_13fs_0():
#    """
#    def crossval_tpfp_6_13fs_0():
#        U.crossval(tap_cand_mfcc_6mic_13fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_13fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_13fs_0():
#    """
#    def crossval_ch_6c_13fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_13fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_13fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_14fs_0():
#    """
#    def crossval_tpfp_6_14fs_0():
#        U.crossval(tap_cand_mfcc_6mic_14fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_14fs_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_14fs_0():
#    """
#    def crossval_ch_6c_14fs_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_14fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_14fs_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_1024nf_0():
#    """
#    def crossval_tpfp_6_1024nf_0():
#        U.crossval(tap_cand_mfcc_6mic_1024nf_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_1024nf_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_1024nf_0():
#    """
#    def crossval_ch_6c_1024nf_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_1024nf_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_1024nf_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_512nf_0():
#    """
#    def crossval_tpfp_6_512nf_0():
#        U.crossval(tap_cand_mfcc_6mic_512nf_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_512nf_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_512nf_0():
#    """
#    def crossval_ch_6c_512nf_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_512nf_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_512nf_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_256nf_0():
#    """
#    def crossval_tpfp_6_256nf_0():
#        U.crossval(tap_cand_mfcc_6mic_256nf_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_256nf_features, tap_cand_labels, verbose=True)
#
#def crossval_ch_6_256nf_0():
#    """
#    def crossval_ch_6c_256nf_0():
#        f = lambda fold: U.traintest_ch(tap_cand_mfcc_6mic_256nf_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    features = tap_cand_mfcc_6mic_256nf_features[positives]
#    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6_1024nf78fs_0():
#    """
#    def crossval_tpfp_6_1024nf78fs_0():
#        U.crossval(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, verbose=True)

def crossval_ch_6_1024nf78fs_0():
    """
    def crossval_ch_6c_1024nf78fs_0():
        f = lambda fold: U.traintest_ch(tap_realtap_mfcc_6mic_1024nf78fs_features[positives], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
        U.crossval_lambda(f)
    """
    features = tap_realtap_mfcc_6mic_1024nf78fs_features[positives]
    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)

def crossval_ch_6TDoA_1024nf78fs_0():
    features = tap_realtap_mfcc_6mic_1024nf78fs_features[positives]
    feats_shape = U.np.shape(features)
    features = U.np.reshape(features, (feats_shape[0], int(U.np.product(feats_shape) / feats_shape[0])))
    tdoa = tap_realtap_tdoa_estimate[positives]
    dirs_shape = U.np.shape(tdoa)
    tdoa = U.np.reshape(tdoa, (dirs_shape[0], int(U.np.product(dirs_shape) / dirs_shape[0])))
    data = U.np.concatenate((features, tdoa), axis=1)
    f = lambda fold: U.traintest_ch(data, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)

def crossval_ch_6DoA_1024nf78fs_0():
    features = tap_realtap_mfcc_6mic_1024nf78fs_features[positives]
    feats_shape = U.np.shape(features)
    features = U.np.reshape(features, (feats_shape[0], int(U.np.product(feats_shape) / feats_shape[0])))
    directions = tap_realtap_direction_estimate[positives]
    dirs_shape = U.np.shape(directions)
    directions = U.np.reshape(directions, (dirs_shape[0], int(U.np.product(dirs_shape) / dirs_shape[0])))
    data = U.np.concatenate((features, directions), axis=1)
    f = lambda fold: U.traintest_ch(data, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)

def crossval_ch_1_1024nf78fs_0():
    """
    def crossval_ch_1_1024nf78fs_0():
        f = lambda fold: U.traintest_ch(tap_realtap_mfcc_6mic_1024nf78fs_features[positives, 4], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
        U.crossval_lambda(f)
    """
    features = tap_realtap_mfcc_6mic_1024nf78fs_features[positives, 4]
    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)

def crossval_ch_2a_1024nf78fs_0():
    """
    def crossval_ch_2a_1024nf78fs_0():
        f = lambda fold: U.traintest_ch(tap_realtap_mfcc_6mic_1024nf78fs_features[positives, :][:, [0,3]], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
        U.crossval_lambda(f)
    """
    features = tap_realtap_mfcc_6mic_1024nf78fs_features[positives, :][:, [0,3]]
    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)

def crossval_ch_2b_1024nf78fs_0():
    """
    def crossval_ch_2b_1024nf78fs_0():
        f = lambda fold: U.traintest_ch(tap_realtap_mfcc_6mic_1024nf78fs_features[positives, :][:, [1,4]], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
        U.crossval_lambda(f)
    """
    features = tap_realtap_mfcc_6mic_1024nf78fs_features[positives, :][:, [1,4]]
    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)

def crossval_ch_2c_1024nf78fs_0():
    """
    def crossval_ch_2c_1024nf78fs_0():
        f = lambda fold: U.traintest_ch(tap_realtap_mfcc_6mic_1024nf78fs_features[positives, :][:, [3,4]], tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
        U.crossval_lambda(f)
    """
    features = tap_realtap_mfcc_6mic_1024nf78fs_features[positives, :][:, [3,4]]
    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)

#def guesseval_lda_8():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=8, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_10():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=10, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_15():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=15, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_20():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=20, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_25():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=25, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_30():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=30, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_35():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=35, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_40():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=40, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_9():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=9, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_7():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=7, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_6():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=6, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_5():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=5, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_4():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=4, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_3():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=3, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_2():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=2, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_1():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1, num_digits=5, penalise_non_chosen=False, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_3p():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=3, num_digits=5, penalise_non_chosen=True, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_2p():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=2, num_digits=5, penalise_non_chosen=True, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_1p():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1, num_digits=5, penalise_non_chosen=True, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_8p():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=8, num_digits=5, penalise_non_chosen=True, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_i3p():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1/3, num_digits=5, penalise_non_chosen=True, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_i2p():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1/2, num_digits=5, penalise_non_chosen=True, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_i1p():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1/1, num_digits=5, penalise_non_chosen=True, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_i8p():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1/8, num_digits=5, penalise_non_chosen=True, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
def guesseval_lda_1p_skipdetect():
    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features[positives], tap_cand_labels[positives], tap_cand_ch[positives], jsons, tap_cand_json_name[positives], ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features[positives], max_guess=20000, tp_prob_power=1, num_digits=5, penalise_non_chosen=True, num_threads=1, verbose=True)
    gu = U.np.array(sorted(gu))
    print(gu)
    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))

def guesseval_lda_pin4_1_skipdetect():
    gu = U.find_pin_lda(tap_realtap_mfcc_6mic_1024nf78fs_features[positives], tap_cand_labels[positives], tap_cand_ch[positives], jsons, tap_cand_json_name[positives], ch_class_names, ch_features=tap_realtap_mfcc_6mic_1024nf78fs_features[positives], max_guess=20000, tp_prob_power=1, num_digits=4, penalise_non_chosen=False, num_threads=1, verbose=True)
    gu = U.np.array(sorted(gu))
    print(gu)
    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))

#def guesseval_lda_1p_l30():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_realtap_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1, penalise_non_chosen=True, num_threads=1, ngram_model = lang_model_3gram_0, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_1p_l31():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_realtap_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1, penalise_non_chosen=True, num_threads=1, ngram_model = lang_model_3gram_1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_1p_l80():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_realtap_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1, penalise_non_chosen=True, num_threads=1, ngram_model = lang_model_8gram_0, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_1p_l81():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_realtap_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1, penalise_non_chosen=True, num_threads=1, ngram_model = lang_model_8gram_1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))

#def guesseval_lda_1p_l30_skipdetect():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features[positives], tap_cand_labels[positives], tap_cand_ch[positives], jsons, tap_cand_json_name[positives], ch_class_names, ch_features=tap_realtap_mfcc_6mic_1024nf78fs_features[positives], max_guess=20000, tp_prob_power=1, penalise_non_chosen=True, num_threads=1, ngram_model = lang_model_3gram_0, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_1p_l80_skipdetect():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features[positives], tap_cand_labels[positives], tap_cand_ch[positives], jsons, tap_cand_json_name[positives], ch_class_names, ch_features=tap_realtap_mfcc_6mic_1024nf78fs_features[positives], max_guess=20000, tp_prob_power=1, penalise_non_chosen=True, num_threads=1, ngram_model = lang_model_8gram_0, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_1p_l31_skipdetect():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features[positives], tap_cand_labels[positives], tap_cand_ch[positives], jsons, tap_cand_json_name[positives], ch_class_names, ch_features=tap_realtap_mfcc_6mic_1024nf78fs_features[positives], max_guess=20000, tp_prob_power=1, penalise_non_chosen=True, num_threads=1, ngram_model = lang_model_3gram_1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_1p_l81_skipdetect():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features[positives], tap_cand_labels[positives], tap_cand_ch[positives], jsons, tap_cand_json_name[positives], ch_class_names, ch_features=tap_realtap_mfcc_6mic_1024nf78fs_features[positives], max_guess=20000, tp_prob_power=1, penalise_non_chosen=True, num_threads=1, ngram_model = lang_model_8gram_1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_1p_dict():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, tp_prob_power=1, penalise_non_chosen=True, dictionary=dictionary, num_threads=1, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))
#
#def guesseval_lda_1p_dict_skipdetect():
#    gu = U.find_pin_lda(tap_cand_mfcc_6mic_all_features[positives], tap_cand_labels[positives], tap_cand_ch[positives], jsons, tap_cand_json_name[positives], ch_class_names, ch_features=tap_realtap_mfcc_6mic_1024nf78fs_features[positives], max_guess=20000, tp_prob_power=1, penalise_non_chosen=True, num_threads=1, dictionary=dictionary, verbose=True)
#    gu = U.np.array(sorted(gu))
#    print(gu)
#    print('Guessed in under 50:', U.np.sum(gu<50), U.np.sum(gu<50) / len(gu))
#    print('Guessed in under 100:', U.np.sum(gu<100), U.np.sum(gu<100) / len(gu))
#    print('Guessed in under 150:', U.np.sum(gu<150), U.np.sum(gu<150) / len(gu))
#    print('Guessed in under 300:', U.np.sum(gu<300), U.np.sum(gu<300) / len(gu))
#    print('Guessed in under 500:', U.np.sum(gu<500), U.np.sum(gu<500) / len(gu))
#    print('Guessed in under 1000:', U.np.sum(gu<1000), U.np.sum(gu<1000) / len(gu))
#    print('Guessed in under 1500:', U.np.sum(gu<1500), U.np.sum(gu<1500) / len(gu))
#    print('Guessed in under 3000:', U.np.sum(gu<3000), U.np.sum(gu<3000) / len(gu))

def crossval_ch_6_1024nf78fs_16000_0():
    features = tap_realtap_mfcc_6mic_1024nf78fs_16000_features[positives]
    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)

def crossval_ch_6_1024nf78fs_8000_0():
    features = tap_realtap_mfcc_6mic_1024nf78fs_8000_features[positives]
    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)

def crossval_ch_6_1024nf78fs_16000l_0():
    features = tap_realtap_mfcc_6mic_1024nf78fs_16000l_features[positives]
    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)

def crossval_ch_6_1024nf78fs_8000l_0():
    features = tap_realtap_mfcc_6mic_1024nf78fs_8000l_features[positives]
    f = lambda fold: U.traintest_ch(features, tap_labels, tap_ch, ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)




runs = []
for r in dir():
    if r.startswith('crossval') or r.startswith('guesseval'):
        runs += [r]
