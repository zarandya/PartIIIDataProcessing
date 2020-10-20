#! /usr/bin/env python

"""
Guess counting using CNN for tap detection and LDA for classification
"""


jsons = []
audio_in = []
tap_cands_in_jsons = []
U = None
batch = None

tap_cand_mfcc_6mic_1024nf78fs_features = []
tap_cand_fourier_6mic_all_features = []
tap_cand_labels = []
tap_cand_ch = []
tap_cand_json_name = []

ch_features = None

def dependencies(symbols):
    deps = []
    if 'ch_features' in symbols:
        deps += ['tap_cand_mfcc_6mic_all_features']
    return deps

def init(symbols = None):
    global tap_cand_ch
    global ch_class_names
    global selection_features
    ch_class_names_arr = sorted(U.np.unique(tap_cand_ch))
    ch_class_names = U.np.array(ch_class_names_arr)
    for i in range(len(tap_cand_ch)):
        tap_cand_ch[i] = ch_class_names_arr.index(tap_cand_ch[i])
    if symbols is not None and 'selection_features' in symbols:
        ch_features = U.np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3)

def guesseval_hybrid_log_8p():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=8, verbose=True)
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
    
def guesseval_hybrid_log_7p():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=7, verbose=True)
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
    
def guesseval_hybrid_log_6p():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=6, verbose=True)
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
    
def guesseval_hybrid_log_5p():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=5, verbose=True)
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
    
def guesseval_hybrid_log_4p():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=4, verbose=True)
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
    
def guesseval_hybrid_log_3p():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=3, verbose=True)
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
    
def guesseval_hybrid_log_2p():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=2, verbose=True)
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
    
def guesseval_hybrid_log_1p():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=1, verbose=True)
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

def guesseval_hybrid_log_8ip():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=1/8, verbose=True)
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
    
def guesseval_hybrid_log_7ip():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=1/7, verbose=True)
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
    
def guesseval_hybrid_log_6ip():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=1/6, verbose=True)
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
    
def guesseval_hybrid_log_5ip():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=1/5, verbose=True)
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
    
def guesseval_hybrid_log_4ip():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=1/4, verbose=True)
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
    
def guesseval_hybrid_log_3ip():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=1/3, verbose=True)
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
    
def guesseval_hybrid_log_2ip():
    gu = U.find_pin_hybrid(tap_cand_mfcc_6mic_1024nf78fs_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=ch_features, max_guess=20000, num_digits=5, penalise_non_chosen=True, tp_prob_power=1/2, verbose=True)
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

runs = []
for r in dir():
    if r.startswith('crossval') or r.startswith('guesseval'):
        runs += [r]
