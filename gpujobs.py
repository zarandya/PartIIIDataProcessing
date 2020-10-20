#! /usr/bin/env python

"""
Jobs using CNN to find false positives and classify keystrokes on:
"""
experimentName = 'atPin'

from types import SimpleNamespace
#import utils as U

jsons = []
audio_in = []
tap_cands_in_jsons = []
U = None
batch = None

tap_cand_waveforms = []
tap_cand_fourier_features = []
tap_cand_fourier_6mic_all_features = []
tap_cand_fourier_6mic_wideband_features = []
tap_cand_fourier_6mic_all_16000_features = []
tap_cand_fourier_6mic_all_8000_features = []
tap_cand_fourier_6mic_all_16000l_features = []
tap_cand_fourier_6mic_all_8000l_features = []
tap_cand_labels = []
tap_cand_ch = []
tap_cand_json_name = []
lang_model_3gram_0 = []
lang_model_3gram_1 = []
lang_model_8gram_0 = []
lang_model_8gram_1 = []
dictionary = []
M = SimpleNamespace()

def dependencies(symbols):
    deps = []
    if 'selection_features' in symbols or 'ch_features' in symbols or 'tap_fourier_6mic_all_features' in symbols:
        deps += ['tap_cand_fourier_6mic_all_features']
    if 'tap_fourier_features' in symbols:
        deps += ['tap_cand_fourier_features']
    if 'tap_fourier_6mic_wideband_features' in symbols:
        symbols += ['tap_cand_fourier_6mic_wideband_features']
    return deps

def init(symbols):
    global tap_cand_ch
    ch_class_names_arr = sorted(U.np.unique(tap_cand_ch))
    M.ch_class_names = U.np.array(ch_class_names_arr)
    tap_cand_ch = U.np.array([ ch_class_names_arr.index(ch) for ch in tap_cand_ch ])
    M.positives = U.np.array(tap_cand_labels) == 1
    if 'tap_fourier_features' in symbols:
        M.tap_fourier_features = tap_cand_fourier_features[M.positives]
    if 'tap_fourier_6mic_all_features' in symbols:
        M.tap_fourier_6mic_all_features = tap_cand_fourier_6mic_all_features[M.positives]
    M.tap_ch = U.np.array(tap_cand_ch)[M.positives]
    M.tap_labels = U.np.ones(dtype=int, shape=len(M.tap_ch))
    if 'selection_features' in symbols:
        M.selection_features = U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3)
    if 'ch_features' in symbols:
        l, nd,nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
        M.ch_features = U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1))
    if 'tap_fourier_6mic_wideband_features' in symbols:
        M.tap_fourier_6mic_wideband_features = tap_cand_fourier_6mic_wideband_features[M.positives]

def crossval_tpfp_2cf_0():
    """
    def crossval_tpfp_2cf_0():
        g = U.np.array([3, 4])
        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        U.crossval_neural(features, tap_cand_labels, verbose=True)
    """
    g = U.np.array([3, 4])
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    U.crossval_neural(features, tap_cand_labels, verbose=True)

#def crossval_tpfp_2cf_13():
#    """
#    def crossval_tpfp_2cf_0():
#        g = U.np.array([3, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([3, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, model=U.gen_model_13, verbose=True)
#
#def crossval_tpfp_2cf_14():
#    """
#    def crossval_tpfp_2cf_0():
#        g = U.np.array([3, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([3, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, model=U.gen_model_14, verbose=True)
#
#def crossval_tpfp_2cf_15():
#    """
#    def crossval_tpfp_2cf_0():
#        g = U.np.array([3, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([3, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, model=U.gen_model_15, verbose=True)
#
#def crossval_tpfp_2cf_33():
#    """
#    def crossval_tpfp_2cf_0():
#        g = U.np.array([3, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([3, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, model=U.gen_model_33, verbose=True)
#
#def crossval_tpfp_2cf_44():
#    """
#    def crossval_tpfp_2cf_0():
#        g = U.np.array([3, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([3, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, model=U.gen_model_44, verbose=True)
#
#def crossval_tpfp_2cf_55():
#    """
#    def crossval_tpfp_2cf_0():
#        g = U.np.array([3, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True)
#    """
#    g = U.np.array([3, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, model=U.gen_model_55, verbose=True)

def crossval_tpfp_1f_0():
    """
    def crossval_tpfp_1f_0():
        U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True)
    """
    U.crossval_neural(tap_cand_fourier_6mic_all_features[:, 4, :, :, U.np.newaxis], tap_cand_labels, verbose=True)
#
def crossval_tpfp_2af_0():
    """
    def crossval_tpfp_2af_0():
        g = U.np.array([1, 4])
        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        U.crossval_neural(features, tap_cand_labels, verbose=True)
    """
    g = U.np.array([1, 4])
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    U.crossval_neural(features, tap_cand_labels, verbose=True)

def crossval_tpfp_2bf_0():
    """
    def crossval_tpfp_2bf_0():
        g = U.np.array([0, 3])
        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        U.crossval_neural(features, tap_cand_labels, verbose=True)
    """
    g = U.np.array([0, 3])
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    U.crossval_neural(features, tap_cand_labels, verbose=True)

#def crossval_tpfp_6DTF_0():
#    """
#    def crossval_tpfp_6DTF_0():
#        U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True)
#    """
#    U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True)
#    
def crossval_tpfp_6FTD_0():
    """
    def crossval_tpfp_6FTD_0():
        U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True)
    """
    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True)

#def crossval_tpfp_6DtF_0():
#    """
#    def crossval_tpfp_6DtF_0():
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True)
#    """
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True)
#
#def crossval_tpfp_6TFd_0():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1)), tap_cand_labels, verbose=True)
#
#def crossval_tpfp_6TdF_0():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1)), tap_cand_labels, verbose=True)
#
#def crossval_tpfp_6TDf_0():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1)), tap_cand_labels, verbose=True)
#
#def crossval_ch_1f_0():
#    """
#    def crossval_ch_1f_0():
#        f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_2af_0():
#    """
#    def crossval_ch_2af_0():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_2bf_0():
#    """
#    def crossval_ch_2bf_0():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_6DTF_0():
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_6mic_all_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_6FTD_0():
#    features = U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3)
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
def crossval_ch_6DtF_0():
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features, (l, nd*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
    U.crossval_lambda(f)
#
#def crossval_ch_6TFd_0():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TdF_0():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TDf_0():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True)
#    U.crossval_lambda(f)
#
## model _33
#
#def crossval_tpfp_1f_33():
#    """
#    def crossval_tpfp_1f_33():
#        U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_33)
#    """
#    U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_33)
#
def crossval_tpfp_2af_33():
    """
    def crossval_tpfp_2af_33():
        g = U.np.array([1, 4])
        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_33)
    """
    g = U.np.array([1, 4])
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_33)

def crossval_tpfp_2bf_33():
    """
    def crossval_tpfp_2bf_33():
        g = U.np.array([0, 3])
        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_33)
    """
    g = U.np.array([0, 3])
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_33)

#def crossval_tpfp_6DTF_33():
#    """
#    def crossval_tpfp_6DTF_33():
#        U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_33)
#    """
#    U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_33)
#    
#def crossval_tpfp_6FTD_33():
#    """
#    def crossval_tpfp_6FTD_33():
#        U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_33)
#    """
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_33)
#
#def crossval_tpfp_6DtF_33():
#    """
#    def crossval_tpfp_6DtF_33():
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_33)
#    """
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_33)
#
#def crossval_tpfp_6TFd_33():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1)), tap_cand_labels, verbose=True, model=U.gen_model_33)
#
#def crossval_tpfp_6TdF_33():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_33)
#
#def crossval_tpfp_6TDf_33():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_33)
#
#def crossval_ch_1f_33():
#    """
#    def crossval_ch_1f_33():
#        f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
#        U.crossval_lambda(f)
#    """
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
#    U.crossval_lambda(f)
#
def crossval_ch_2af_33():
    """
    def crossval_ch_2af_33():
        g = U.np.array([1, 4])
        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
        U.crossval_lambda(f)
    """
    g = U.np.array([1, 4])
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
    U.crossval_lambda(f)

def crossval_ch_2bf_33():
    """
    def crossval_ch_2bf_33():
        g = U.np.array([0, 3])
        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
        U.crossval_lambda(f)
    """
    g = U.np.array([0, 3])
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
    U.crossval_lambda(f)

def crossval_ch_2cf_33():
    """
    def crossval_ch_2cf_33():
        g = U.np.array([3, 4])
        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
        U.crossval_lambda(f)
    """
    g = U.np.array([3, 4])
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
    U.crossval_lambda(f)

#def crossval_ch_6DTF_33():
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_6mic_all_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
#    U.crossval_lambda(f)
#
#def crossval_ch_6FTD_33():
#    features = U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3)
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
#    U.crossval_lambda(f)
#
def crossval_ch_6DtF_33():
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features, (l, nd*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
    U.crossval_lambda(f)

#def crossval_ch_6TFd_33():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TdF_33():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TDf_33():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_33)
#    U.crossval_lambda(f)
#
## model _44
#
def crossval_tpfp_1f_44():
    """
    def crossval_tpfp_1f_44():
        U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_44)
    """
    U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_44)

def crossval_tpfp_2af_44():
    """
    def crossval_tpfp_2af_44():
        g = U.np.array([1, 4])
        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_44)
    """
    g = U.np.array([1, 4])
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_44)

def crossval_tpfp_2bf_44():
    """
    def crossval_tpfp_2bf_44():
        g = U.np.array([0, 3])
        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_44)
    """
    g = U.np.array([0, 3])
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_44)

def crossval_tpfp_6DTF_44():
    """
    def crossval_tpfp_6DTF_44():
        U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_44)
    """
    U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_44)
    
def crossval_tpfp_6FTD_44():
    """
    def crossval_tpfp_6FTD_44():
        U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_44)
    """
    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_44)

def crossval_tpfp_6DtF_44():
    """
    def crossval_tpfp_6DtF_44():
        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
        U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_44)
    """
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_44)

def crossval_tpfp_6TFd_44():
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1)), tap_cand_labels, verbose=True, model=U.gen_model_44)

def crossval_tpfp_6TdF_44():
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_44)

def crossval_tpfp_6TDf_44():
    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_44)

def crossval_ch_1f_44():
    """
    def crossval_ch_1f_44():
        f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
        U.crossval_lambda(f)
    """
    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_2af_44():
    """
    def crossval_ch_2af_44():
        g = U.np.array([1, 4])
        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
        U.crossval_lambda(f)
    """
    g = U.np.array([1, 4])
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_2bf_44():
    """
    def crossval_ch_2bf_44():
        g = U.np.array([0, 3])
        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
        U.crossval_lambda(f)
    """
    g = U.np.array([0, 3])
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_2cf_44():
    """
    def crossval_ch_2cf_44():
        g = U.np.array([3, 4])
        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
        U.crossval_lambda(f)
    """
    g = U.np.array([3, 4])
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_6DTF_44():
    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_6mic_all_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_6FTD_44():
    features = U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3)
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_6DtF_44():
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features, (l, nd*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_6TFd_44():
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_6TdF_44():
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

def crossval_ch_6TDf_44():
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
    U.crossval_lambda(f)

## model _55
#
#def crossval_tpfp_1f_55():
#    """
#    def crossval_tpfp_1f_55():
#        U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_55)
#    """
#    U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_55)
#
#def crossval_tpfp_2af_55():
#    """
#    def crossval_tpfp_2af_55():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_55)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_55)
#
#def crossval_tpfp_2bf_55():
#    """
#    def crossval_tpfp_2bf_55():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_55)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_55)

#def crossval_tpfp_6DTF_55():
#    """
#    def crossval_tpfp_6DTF_55():
#        U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_55)
#    """
#    U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_55)
#    
#def crossval_tpfp_6FTD_55():
#    """
#    def crossval_tpfp_6FTD_55():
#        U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_55)
#    """
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_55)
#
#def crossval_tpfp_6DtF_55():
#    """
#    def crossval_tpfp_6DtF_55():
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_55)
#    """
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_55)
#
#def crossval_tpfp_6TFd_55():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1)), tap_cand_labels, verbose=True, model=U.gen_model_55)
#
#def crossval_tpfp_6TdF_55():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_55)
#
#def crossval_tpfp_6TDf_55():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_55)
#
#def crossval_ch_1f_55():
#    """
#    def crossval_ch_1f_55():
#        f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#        U.crossval_lambda(f)
#    """
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#    U.crossval_lambda(f)
#
#def crossval_ch_2af_55():
#    """
#    def crossval_ch_2af_55():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#    U.crossval_lambda(f)
#
#def crossval_ch_2bf_55():
#    """
#    def crossval_ch_2bf_55():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#    U.crossval_lambda(f)
#
#def crossval_ch_6DTF_55():
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_6mic_all_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#    U.crossval_lambda(f)
#
#def crossval_ch_6FTD_55():
#    features = U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3)
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#    U.crossval_lambda(f)
#
def crossval_ch_6DtF_55():
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features, (l, nd*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
    U.crossval_lambda(f)

#def crossval_ch_6TFd_55():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TdF_55():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TDf_55():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_55)
#    U.crossval_lambda(f)
#
## model _13
#
#def crossval_tpfp_1f_13():
#    """
#    def crossval_tpfp_1f_13():
#        U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_13)
#    """
#    U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_13)
#
#def crossval_tpfp_2af_13():
#    """
#    def crossval_tpfp_2af_13():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_13)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_13)
#
#def crossval_tpfp_2bf_13():
#    """
#    def crossval_tpfp_2bf_13():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_13)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_13)

#def crossval_tpfp_6DTF_13():
#    """
#    def crossval_tpfp_6DTF_13():
#        U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_13)
#    """
#    U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_13)
#    
#def crossval_tpfp_6FTD_13():
#    """
#    def crossval_tpfp_6FTD_13():
#        U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_13)
#    """
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_13)
#
#def crossval_tpfp_6DtF_13():
#    """
#    def crossval_tpfp_6DtF_13():
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_13)
#    """
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_13)
#
#def crossval_tpfp_6TFd_13():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1)), tap_cand_labels, verbose=True, model=U.gen_model_13)
#
#def crossval_tpfp_6TdF_13():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_13)
#
#def crossval_tpfp_6TDf_13():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_13)
#
#def crossval_ch_1f_13():
#    """
#    def crossval_ch_1f_13():
#        f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#        U.crossval_lambda(f)
#    """
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#    U.crossval_lambda(f)
#
#def crossval_ch_2af_13():
#    """
#    def crossval_ch_2af_13():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#    U.crossval_lambda(f)
#
#def crossval_ch_2bf_13():
#    """
#    def crossval_ch_2bf_13():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#    U.crossval_lambda(f)
#
#def crossval_ch_6DTF_13():
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_6mic_all_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#    U.crossval_lambda(f)
#
#def crossval_ch_6FTD_13():
#    features = U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3)
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#    U.crossval_lambda(f)
#
def crossval_ch_6DtF_13():
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features, (l, nd*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
    U.crossval_lambda(f)

#def crossval_ch_6TFd_13():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TdF_13():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TDf_13():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_13)
#    U.crossval_lambda(f)
#
## model _14
#
#def crossval_tpfp_1f_14():
#    """
#    def crossval_tpfp_1f_14():
#        U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_14)
#    """
#    U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_14)
#
#def crossval_tpfp_2af_14():
#    """
#    def crossval_tpfp_2af_14():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_14)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_14)
#
#def crossval_tpfp_2bf_14():
#    """
#    def crossval_tpfp_2bf_14():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_14)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_14)

#def crossval_tpfp_6DTF_14():
#    """
#    def crossval_tpfp_6DTF_14():
#        U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_14)
#    """
#    U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_14)
#    
#def crossval_tpfp_6FTD_14():
#    """
#    def crossval_tpfp_6FTD_14():
#        U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_14)
#    """
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_14)
#
#def crossval_tpfp_6DtF_14():
#    """
#    def crossval_tpfp_6DtF_14():
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_14)
#    """
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_14)
#
#def crossval_tpfp_6TFd_14():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1)), tap_cand_labels, verbose=True, model=U.gen_model_14)
#
#def crossval_tpfp_6TdF_14():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_14)
#
#def crossval_tpfp_6TDf_14():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_14)
#
#def crossval_ch_1f_14():
#    """
#    def crossval_ch_1f_14():
#        f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#        U.crossval_lambda(f)
#    """
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#    U.crossval_lambda(f)
#
#def crossval_ch_2af_14():
#    """
#    def crossval_ch_2af_14():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#    U.crossval_lambda(f)
#
#def crossval_ch_2bf_14():
#    """
#    def crossval_ch_2bf_14():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#    U.crossval_lambda(f)
#
#def crossval_ch_6DTF_14():
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_6mic_all_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#    U.crossval_lambda(f)
#
#def crossval_ch_6FTD_14():
#    features = U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3)
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#    U.crossval_lambda(f)
#
def crossval_ch_6DtF_14():
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features, (l, nd*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
    U.crossval_lambda(f)

#def crossval_ch_6TFd_14():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TdF_14():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TDf_14():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_14)
#    U.crossval_lambda(f)
#
## model _15
#
#def crossval_tpfp_1f_15():
#    """
#    def crossval_tpfp_1f_15():
#        U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_15)
#    """
#    U.crossval_neural(tap_cand_fourier_features, tap_cand_labels, verbose=True, model=U.gen_model_15)
#
#def crossval_tpfp_2af_15():
#    """
#    def crossval_tpfp_2af_15():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_15)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_15)
#
#def crossval_tpfp_2bf_15():
#    """
#    def crossval_tpfp_2bf_15():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_15)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    features = U.np.reshape(tap_cand_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    U.crossval_neural(features, tap_cand_labels, verbose=True, model=U.gen_model_15)

#def crossval_tpfp_6DTF_15():
#    """
#    def crossval_tpfp_6DTF_15():
#        U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_15)
#    """
#    U.crossval_neural(tap_cand_fourier_6mic_all_features, tap_cand_labels, verbose=True, model=U.gen_model_15)
#    
#def crossval_tpfp_6FTD_15():
#    """
#    def crossval_tpfp_6FTD_15():
#        U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_15)
#    """
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_15)
#
#def crossval_tpfp_6DtF_15():
#    """
#    def crossval_tpfp_6DtF_15():
#        l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#        U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_15)
#    """
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_15)
#
#def crossval_tpfp_6TFd_15():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1)), tap_cand_labels, verbose=True, model=U.gen_model_15)
#
#def crossval_tpfp_6TdF_15():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_15)
#
#def crossval_tpfp_6TDf_15():
#    l, nd, nt, nf = U.np.shape(tap_cand_fourier_6mic_all_features)
#    U.crossval_neural(U.np.reshape(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1)), tap_cand_labels, verbose=True, model=U.gen_model_15)
#
#def crossval_ch_1f_15():
#    """
#    def crossval_ch_1f_15():
#        f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#        U.crossval_lambda(f)
#    """
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#    U.crossval_lambda(f)
#
#def crossval_ch_2af_15():
#    """
#    def crossval_ch_2af_15():
#        g = U.np.array([1, 4])
#        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([1, 4])
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#    U.crossval_lambda(f)
#
#def crossval_ch_2bf_15():
#    """
#    def crossval_ch_2bf_15():
#        g = U.np.array([0, 3])
#        l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#        features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#        f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#        U.crossval_lambda(f)
#    """
#    g = U.np.array([0, 3])
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features[:, g], (l, 2*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#    U.crossval_lambda(f)
#
#def crossval_ch_6DTF_15():
#    f = lambda fold: U.traintest_ch_cnn(M.tap_fourier_6mic_all_features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#    U.crossval_lambda(f)
#
#def crossval_ch_6FTD_15():
#    features = U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3)
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#    U.crossval_lambda(f)
#
def crossval_ch_6DtF_15():
    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
    features = U.np.reshape(M.tap_fourier_6mic_all_features, (l, nd*nt, nf, 1))
    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
    U.crossval_lambda(f)

#def crossval_ch_6TFd_15():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 3), (l, nt, nf*nd, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TdF_15():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt*nd, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#    U.crossval_lambda(f)
#
#def crossval_ch_6TDf_15():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(U.np.swapaxes(M.tap_fourier_6mic_all_features, 1, 2), (l, nt, nd*nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_15)
#    U.crossval_lambda(f)
#
#def guesseval_cnn_1():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, verbose=True)
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
#def guesseval_cnn_2():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, tp_prob_power=2, verbose=True)
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
#def guesseval_cnn_3():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, tp_prob_power=3, verbose=True)
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
#def guesseval_cnn_5():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, tp_prob_power=5, verbose=True)
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
#def guesseval_cnn_8():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, tp_prob_power=8, verbose=True)
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
#def guesseval_cnn_10():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, tp_prob_power=10, verbose=True)
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
#
#def guesseval_cnn_log_1():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, verbose=True)
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
#def guesseval_cnn_log_2():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, tp_prob_power=2, verbose=True)
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
#def guesseval_cnn_log_3():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, tp_prob_power=3, verbose=True)
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
#def guesseval_cnn_log_5():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, tp_prob_power=5, verbose=True)
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
#def guesseval_cnn_log_8():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, tp_prob_power=8, verbose=True)
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
#def guesseval_cnn_log_9():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, tp_prob_power=9, verbose=True)
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
#def guesseval_cnn_log_10():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, tp_prob_power=10, verbose=True)
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
#def guesseval_cnn_log_12():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, tp_prob_power=12, verbose=True)
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
#def guesseval_cnn_log_15():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, tp_prob_power=15, verbose=True)
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
#def guesseval_cnn_log_18():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, tp_prob_power=18, verbose=True)
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
#def guesseval_cnn_log_20():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, tp_prob_power=20, verbose=True)
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
#def guesseval_cnn_log_8p():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=8, verbose=True)
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
#def guesseval_cnn_log_6p():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=6, verbose=True)
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
#def guesseval_cnn_log_4p():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=4, verbose=True)
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
#def guesseval_cnn_log_2p():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=2, verbose=True)
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
def guesseval_cnn_log_1p():
    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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
    
def guesseval_cnn_log_1p_skipdetect():
    gu = U.find_pin_cnn(M.selection_features[M.positives], tap_cand_labels[M.positives], tap_cand_ch[M.positives], jsons, tap_cand_json_name[M.positives], M.ch_class_names, ch_features=M.ch_features[M.positives], ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=5, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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

def guesseval_cnn_pin4_log_1_skipdetect():
    gu = U.find_pin_cnn(M.selection_features[M.positives], tap_cand_labels[M.positives], tap_cand_ch[M.positives], jsons, tap_cand_json_name[M.positives], M.ch_class_names, ch_features=M.ch_features[M.positives], ch_model_gen = U.gen_model_44, max_guess=20000, num_digits=4, use_log_probs=True, penalise_non_chosen=False, tp_prob_power=1, verbose=True)
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

#def guesseval_cnn_log_1p_l30():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, ngram_model=lang_model_3gram_0, verbose=True)
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
#def guesseval_cnn_log_1p_l31():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, ngram_model=lang_model_3gram_1, verbose=True)
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
#def guesseval_cnn_log_1p_l80():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, ngram_model=lang_model_8gram_0, verbose=True)
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
#def guesseval_cnn_log_1p_l81():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, ngram_model=lang_model_8gram_1, verbose=True)
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
    


#def crossval_ch_6DtF_deep445():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features, (l, nd*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_deep445)
#    U.crossval_lambda(f)
#
#def crossval_ch_6DtF_deep447():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features, (l, nd*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_deep447)
#    U.crossval_lambda(f)
#
#def crossval_ch_6DtF_deep449():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_all_features)
#    features = U.np.reshape(M.tap_fourier_6mic_all_features, (l, nd*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_deep449)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6FTD_deep445():
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_deep445)
#
#def crossval_tpfp_6FTD_deep447():
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_deep447)
#
#def crossval_tpfp_6FTD_deep449():
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3), tap_cand_labels, verbose=True, model=U.gen_model_deep449)
#
#def guesseval_cnn_log_1p_deep445():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, model_gen = U.gen_model_deep445, max_guess=20000, num_digits=5, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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
#def guesseval_cnn_log_1p_deep447():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, model_gen = U.gen_model_deep447, max_guess=20000, num_digits=5, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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
#def guesseval_cnn_log_1p_deep449():
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, model_gen = U.gen_model_deep449, max_guess=20000, num_digits=5, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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

#def guesseval_cnn_log_1p_l30_skipdetect():
#    if len(M.ch_class_names) < 20:
#        print("This is not a words experiment, skipping test")
#        return
#    gu = U.find_pin_cnn(M.selection_features[M.positives], tap_cand_labels[M.positives], tap_cand_ch[M.positives], jsons, tap_cand_json_name[M.positives], M.ch_class_names, ch_features=M.ch_features[M.positives], ch_model_gen = U.gen_model_44, max_guess=20000, ngram_model=lang_model_3gram_0, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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
#def guesseval_cnn_log_1p_l80_skipdetect():
#    if len(M.ch_class_names) < 20:
#        print("This is not a words experiment, skipping test")
#        return
#    gu = U.find_pin_cnn(M.selection_features[M.positives], tap_cand_labels[M.positives], tap_cand_ch[M.positives], jsons, tap_cand_json_name[M.positives], M.ch_class_names, ch_features=M.ch_features[M.positives], ch_model_gen = U.gen_model_44, max_guess=20000, ngram_model=lang_model_8gram_0, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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
#def guesseval_cnn_log_1p_l31_skipdetect():
#    if len(M.ch_class_names) < 20:
#        print("This is not a words experiment, skipping test")
#        return
#    gu = U.find_pin_cnn(M.selection_features[M.positives], tap_cand_labels[M.positives], tap_cand_ch[M.positives], jsons, tap_cand_json_name[M.positives], M.ch_class_names, ch_features=M.ch_features[M.positives], ch_model_gen = U.gen_model_44, max_guess=20000, ngram_model=lang_model_3gram_1, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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
#def guesseval_cnn_log_1p_l81_skipdetect():
#    if len(M.ch_class_names) < 20:
#        print("This is not a words experiment, skipping test")
#        return
#    gu = U.find_pin_cnn(M.selection_features[M.positives], tap_cand_labels[M.positives], tap_cand_ch[M.positives], jsons, tap_cand_json_name[M.positives], M.ch_class_names, ch_features=M.ch_features[M.positives], ch_model_gen = U.gen_model_44, max_guess=20000, ngram_model=lang_model_8gram_1, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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
#def guesseval_cnn_log_1p_dict():
#    if len(M.ch_class_names) < 20:
#        print("This is not a words experiment, skipping test")
#        return
#    gu = U.find_pin_cnn(M.selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, M.ch_class_names, ch_features=M.ch_features, ch_model_gen = U.gen_model_44, max_guess=20000, dictionary=dictionary, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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
#def guesseval_cnn_log_1p_dict_skipdetect():
#    if len(M.ch_class_names) < 20:
#        print("This is not a words experiment, skipping test")
#        return
#    gu = U.find_pin_cnn(M.selection_features[M.positives], tap_cand_labels[M.positives], tap_cand_ch[M.positives], jsons, tap_cand_json_name[M.positives], M.ch_class_names, ch_features=M.ch_features[M.positives], ch_model_gen = U.gen_model_44, max_guess=20000, dictionary=dictionary, use_log_probs=True, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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
#def crossval_tpfp_wb_6FTD_0():
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_wideband_features, 1, 3), tap_cand_labels, verbose=True)
#
#def crossval_ch_wb_6DtF_44():
#    l, nd, nt, nf = U.np.shape(M.tap_fourier_6mic_wideband_features)
#    features = U.np.reshape(M.tap_fourier_6mic_wideband_features, (l, nd*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_44)
#    U.crossval_lambda(f)


#def crossval_ch_6DtF_16000_0():
#    features = tap_cand_fourier_6mic_all_16000_features[M.positives]
#    l, nd, nt, nf = U.np.shape(features)
#    features = U.np.reshape(features, (l, nd*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_0)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6FTD_16000_0():
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_16000_features, 1, 3), tap_cand_labels, verbose=True)
#
#def crossval_ch_6DtF_8000_0():
#    features = tap_cand_fourier_6mic_all_8000_features[M.positives]
#    l, nd, nt, nf = U.np.shape(features)
#    features = U.np.reshape(features, (l, nd*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_0)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6FTD_8000_0():
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_8000_features, 1, 3), tap_cand_labels, verbose=True)
#
#
#def crossval_ch_6DtF_16000l_0():
#    features = tap_cand_fourier_6mic_all_16000l_features[M.positives]
#    l, nd, nt, nf = U.np.shape(features)
#    features = U.np.reshape(features, (l, nd*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_0)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6FTD_16000l_0():
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_16000l_features, 1, 3), tap_cand_labels, verbose=True)
#
#def crossval_ch_6DtF_8000l_0():
#    features = tap_cand_fourier_6mic_all_8000l_features[M.positives]
#    l, nd, nt, nf = U.np.shape(features)
#    features = U.np.reshape(features, (l, nd*nt, nf, 1))
#    f = lambda fold: U.traintest_ch_cnn(features, M.tap_labels, M.tap_ch, M.ch_class_names, fold=fold, verbose=True, model=U.gen_model_0)
#    U.crossval_lambda(f)
#
#def crossval_tpfp_6FTD_8000l_0():
#    U.crossval_neural(U.np.swapaxes(tap_cand_fourier_6mic_all_8000l_features, 1, 3), tap_cand_labels, verbose=True)
#








runs = []
for r in dir():
    if r.startswith('crossval') or r.startswith('guesseval'):
        runs += [r]
