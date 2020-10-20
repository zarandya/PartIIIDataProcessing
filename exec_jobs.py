#! /usr/bin/env python

import utils as __utils
from utils import *
from os import listdir
from os.path import join, isdir, isfile
from gc import collect
from importlib.util import spec_from_file_location, module_from_spec
from traceback import print_exc
from tensorflow.config import list_physical_devices as __tensorflow_list_physical_devices
from tensorflow.config.experimental import set_device_policy as __tensorflow_set_device_policy
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from acoustics.cepstrum import real_cepstrum
from librosa.feature import mfcc
import sys

if __tensorflow_list_physical_devices('GPU') == []:
	print("no GPU found")
	#quit()
__tensorflow_set_device_policy('warn')
disable_eager_execution()

stdout_orig = sys.stdout
stderr_orig = sys.stderr

def get_jsons():
    base_dir = '/home/zarandy/Documents/sound-samples/sync/'
    #base_dir = '/mnt/sdc1/almos/sound-samples/sync'
    jsons = []
    for f in listdir(base_dir):
        d = join(base_dir, f)
        if (isdir(d)):
            json_file = join(d, "syncd.json")
            if (isfile(json_file)):
                jsons += [open_json(json_file)]
    return jsons

def run_experiment(batch, log, donefile):
    done_runs = []
    if isfile(donefile):
        with open(donefile) as done:
            done_runs = [ line.rstrip('\n') for line in done ]
    not_all_done = False
    runs_to_do = []
    symbols = []
    for ru in batch.runs:
        if not ru in done_runs:
            runs_to_do += [ru]
            symbols += getattr(batch, ru).__code__.co_names
    if len(runs_to_do) == 0:
        return
    if hasattr(batch, 'dependencies'):
        symbols += batch.dependencies(symbols)
    symbols = np.unique(symbols)
    batch.U = __utils
    batch.batch = batch
    log.write(batch.__doc__)
    jsons = get_jsons()
    experimentName = batch.experimentName
    print(experimentName)
    print('runs to do:', runs_to_do)
    print('symbols:', symbols)
    if not hasattr(batch, 'fallback_mode'):
        setattr(batch, 'fallback_mode', 'ignore')
    audio_in, tap_cands_in_jsons, num_found, num_correctly_syncd = open_audio_files(jsons, experimentName, batch.fallback_mode)
    log.write('found %d\n' % num_found)
    log.write('correctly sync\'d %d\n' % num_correctly_syncd)
    log.flush()
    batch.jsons = jsons
    get_ngram_model_for_batch(batch, symbols)
    extract_features(batch, jsons, tap_cands_in_jsons, audio_in, symbols)
    log.write('done extracting features\n')
    batch.audio_in = audio_in
    batch.tap_cands_in_jsons = tap_cands_in_jsons
    log.write('initialising batch\n')
    if batch.init.__code__.co_argcount == 1:
        batch.init(symbols)
    else:
        batch.init()
    log.write('running runs\n')
    log.flush()
    with open(donefile, mode='a') as done:
        for ru in runs_to_do:
            r = getattr(batch, ru)
            try:
                log.write('\n--------\n')
                log.write(r.__name__)
                log.write('\n')
                if (r.__doc__) is not None:
                    log.write(r.__doc__)
                log.flush()
                r()
                log.write('\n')
            except BaseException as e:
                log.write('run failed\n')
                print_exc(file=log)
                if type(e) is ResourceExhaustedError:
                    collect()
                    raise e
            done.write(r.__name__)
            done.write('\n')
            done.flush()
            log.flush()
            collect()
    log.write('\ndone with batch\n')
    log.flush()
            
def extract_features(batch, jsons, tap_cands_in_jsons, audio_in, symbols):
    experimentName = batch.experimentName
    feats = []
    feats_realtap_only = []
    fallback_mode = 'ignore'
    if hasattr(batch, 'fallback_mode'):
        fallback_mode = batch.fallback_mode
    for k in feature_types.keys():
        if (hasattr(batch, k)) and k in symbols:
            setattr(batch, k, [])
            feats += [k]
        if (hasattr(batch, k.replace('_cand_', '_realtap_', 1))):
            feats_realtap_only += [k]
    for jid, (json, tap_cands, fsx) in enumerate(zip(jsons, tap_cands_in_jsons, audio_in)):
        if (json.experimentName == experimentName or (type(experimentName) is re.Pattern and experimentName.fullmatch(json.experimentName) is not None)):
            taps_syncd = json.taps_syncd
            if hasattr(json, 'taps_syncd_orig'):
                if fallback_mode == 'ignore':
                    continue
                if fallback_mode == 'use_syncd_to_self':
                    pass
                if fallback_mode == 'use_syncd':
                    taps_syncd = json.taps_syncd_orig
            fs, x = fsx
            print('extracting features from', json.wav)
            actual_taps = [ int(i.time_samples) for i in taps_syncd ]
            i = 0
            run = 0
            tap_cands = np.array(tap_cands).astype(int)
            while i < len(tap_cands):
                if run == 0:
                    selected = -1
                    while i+run+1 < len(tap_cands) and tap_cands[i+run][0] + 1 == tap_cands[i+run+1][0]:
                        if tap_cands[i+run][0] * 32 in actual_taps:
                            selected = run
                        run += 1
                    if tap_cands[i+run][0] * 32 in actual_taps:
                        selected = run
                    run += 1
                    if selected == -1:
                        probs = np.array([ c[0] for c in tap_cands[i:(i+run)] ])
                        probs = probs / np.sum(probs)
                        selected = np.random.choice(run, p=probs)
                if selected == 0:
                    idx = tap_cands[i][0] * 32
                    seg = x[(idx-64):(idx+448), :]
                    longseg = x[(idx-128):(idx+896)]
                    C = SimpleNamespace(idx=idx, seg=seg, json=json, fs=fs, json_idx = jid, longseg=longseg)
                    if len(seg) == 512 and len(longseg) == 1024 and (idx < len(x) - 9600 or idx <= actual_taps[-1]):
                        finite = True
                        features_arr = []
                        features_arr_realtap_only = []
                        for feat in feats:
                            new_features = feature_types[feat](C)
                            features_arr += [new_features]
                            if feat in features_requiring_finite_check and not np.isfinite(new_features).all():
                                finite = False
                                break
                        labels = 0
                        ch = 'FALSE_POSITIVE'
                        for tap in taps_syncd:
                            if idx == int(tap.time_samples):
                                labels = 1
                                if hasattr(tap, "ch"):
                                    ch = tap.ch
                                break
                        for feat in feats_realtap_only:
                            fattr = feat.replace('_cand_', '_realtap_')
                            new_features = None
                            if labels == 0:
                                arr = getattr(batch, fattr)
                                if len(arr) > 0:
                                    new_features = np.zeros(np.shape(arr[0]))
                            if new_features is None:
                                new_features = feature_types[feat](C)
                            features_arr_realtap_only += [new_features]
                            if fattr in features_requiring_finite_check and not np.isfinite(new_features).all():
                                finite = False
                                break
                        if finite:
                            for n, f in zip(feats, features_arr):
                                attr = getattr(batch, n)
                                setattr(batch, n, attr + [f])
                            for n, f in zip(feats_realtap_only, features_arr_realtap_only):
                                fattr = n.replace('_cand_', '_realtap_')
                                attr = getattr(batch, fattr)
                                setattr(batch, fattr, attr + [f])
                            batch.tap_cand_labels += [labels]
                            batch.tap_cand_ch += [ch]
                run -= 1
                selected -= 1
                i += 1
            print('tap cands found in total:', len(batch.tap_cand_labels))
        #this indentation level might have been continued over
    for k in feats:
        print('packing feature', k)
        arr = getattr(batch, k)
        print(len(arr), np.shape(arr[0]))
        setattr(batch, k, np.array(arr))
    for l in feats_realtap_only:
        k = l.replace('_cand_', '_realtap_')
        print('packing feature', k)
        arr = getattr(batch, k)
        print(len(arr), np.shape(arr[0]))
        setattr(batch, k, np.array(arr))
    batch.tap_cand_labels = np.array(batch.tap_cand_labels)
    batch.tap_cand_ch = np.array(batch.tap_cand_ch)
        
def get_ngram_model_for_batch(batch, symbols):
    if hasattr(batch, 'dictionary'):
        with open('/usr/share/dict/words') as f:
            batch.dictionary = [s.rstrip('\n').lower() for s in f]
#    jj = None
#    experimentName = batch.experimentName
#    for json in batch.jsons:
#        if (json.experimentName == experimentName or (type(experimentName) is re.Pattern and experimentName.fullmatch(json.experimentName) is not None)):
#            jj = json
#            break
#    alphabet = jj.keys.__dict__
#    for i in range(32):
#        for j in range(2):
#            attr = "lang_model_%dgram_%d" % (i, j)
#            if hasattr(batch, attr) and attr in symbols:
#                setattr(batch, attr, gen_ngrams(i, alphabet, j))
#
    

                        

feature_types = {
        'tap_cand_waveforms': (lambda C: C.seg),
        'tap_cand_mfcc_features': (lambda C: mfcc(C.seg[:, 4].astype(float), sr=C.fs, hop_length=32, n_fft=128, n_mfcc=20)),
        'tap_cand_ceps_features': (lambda C: real_cepstrum(C.seg[:, 4])),
        'tap_cand_fourier_features': (lambda C: stft(C.seg[:, 4], fs=C.fs, nperseg=128, noverlap=96)[2][np.newaxis, 0:16].swapaxes(1, 2)),
        'tap_cand_mfcc_6mic_coarse_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=C.fs) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_all_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=C.fs, hop_length=32, n_fft=128) for j in range(0, 6)])),
        'tap_cand_fourier_6mic_all_features': (lambda C: np.moveaxis(stft(C.seg, fs=C.fs, nperseg=128, noverlap=96, axis=0)[2][0:16], 0, 2)),
        'tap_cand_fourier_6mic_wideband_features': (lambda C: np.moveaxis(stft(C.seg, fs=C.fs, nperseg=128, noverlap=96, axis=0)[2], 0, 2)),
        'tap_cand_json_name': (lambda C: C.json_idx),
        'tap_cand_mfcc_6mic_sanefakefs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs/2), hop_length=512, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_quarterfs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs/4), hop_length=512, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_sanefakelongseg_features': (lambda C: np.array([mfcc(C.longseg[:, j].astype(float), sr=int(C.fs/2), hop_length=512, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_sanefakeshortseg_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs/2), hop_length=128, n_fft=128) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_02fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 0.2), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_03fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 0.3), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_04fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 0.4), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_05fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 0.5), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_06fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 0.6), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_07fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 0.7), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_08fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 0.8), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_09fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 0.9), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_10fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 1.0), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_11fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 1.1), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_12fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 1.2), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_13fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 1.3), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_14fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 1.0), hop_length=2048, n_fft=2048) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_1024nf_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 1.0), hop_length=2048, n_fft=1024) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_512nf_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 1.0), hop_length=512, n_fft=512) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_256nf_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 1.0), hop_length=512, n_fft=256) for j in range(0, 6)])),
        'tap_cand_mfcc_6mic_1024nf78fs_features': (lambda C: np.array([mfcc(C.seg[:, j].astype(float), sr=int(C.fs * 7 / 8), hop_length=2048, n_fft=1024) for j in range(0, 6)])),
        'tap_cand_fourier_6mic_all_16000_features': (lambda C: np.moveaxis(stft(C.seg[::4], fs=C.fs/4, nperseg=32, noverlap=24, axis=0)[2][0:16], 0, 2)),
        'tap_cand_mfcc_6mic_1024nf78fs_16000_features': (lambda C: np.array([mfcc(C.seg[::4, j].astype(float), sr=int(C.fs * 7 / 32), hop_length=512, n_fft=256) for j in range(0, 6)])),
        'tap_cand_fourier_6mic_all_8000_features': (lambda C: np.moveaxis(stft(C.seg[::8], fs=C.fs/4, nperseg=16, noverlap=12, axis=0)[2][0:16], 0, 2)),
        'tap_cand_mfcc_6mic_1024nf78fs_8000_features': (lambda C: np.array([mfcc(C.seg[::8, j].astype(float), sr=int(C.fs * 7 / 64), hop_length=256, n_fft=128) for j in range(0, 6)])),
        'tap_cand_fourier_6mic_all_16000l_features': (lambda C: np.moveaxis(stft(C.longseg[::4], fs=C.fs/4, nperseg=64, noverlap=48, axis=0)[2][0:16], 0, 2)),
        'tap_cand_mfcc_6mic_1024nf78fs_16000l_features': (lambda C: np.array([mfcc(C.longseg[::4, j].astype(float), sr=int(C.fs * 7 / 32), hop_length=1024, n_fft=512) for j in range(0, 6)])),
        'tap_cand_fourier_6mic_all_8000l_features': (lambda C: np.moveaxis(stft(C.longseg[::8], fs=C.fs/4, nperseg=32, noverlap=24, axis=0)[2][0:16], 0, 2)),
        'tap_cand_mfcc_6mic_1024nf78fs_8000l_features': (lambda C: np.array([mfcc(C.longseg[::8, j].astype(float), sr=int(C.fs * 7 / 64), hop_length=512, n_fft=256) for j in range(0, 6)])),
        'tap_cand_tdoa_estimate': (lambda C: compute_TDoA_valid_ranges_only(C.longseg, C.fs, 128)),
        'tap_cand_timestamp_sample': (lambda C: C.idx),
}

features_requiring_finite_check = [
        'tap_cand_mfcc_features',
        'tap_cand_ceps_features',
        'tap_cand_fourier_features',
        'tap_cand_mfcc_6mic_coarse_features',
        'tap_cand_mfcc_6mic_all_features',
        'tap_cand_fourier_6mic_all_features',
        'tap_cand_mfcc_6mic_sanefakefs_features',
        'tap_cand_mfcc_6mic_quarterfs_features',
        'tap_cand_mfcc_6mic_sanefakelongseg_features',
        'tap_cand_mfcc_6mic_sanefakeshortseg_features',
        ]




jobs_dir = '/home/zarandy/Documents/jobs/'
for dn in sorted(listdir(jobs_dir)):
    d = join(jobs_dir, dn)
    if isdir(d):
        fn = join(d, 'batch.py')
        fnc = join(d, 'batch_completed')
        if isfile(fn):
            spec = spec_from_file_location('module', fn)
            batch = module_from_spec(spec)
            spec.loader.exec_module(batch)
            done_filename = join(d, 'runs_done')
            if isfile(join(d, 'experimentName')):
                with open(join(d, 'experimentName')) as expn:
                    batch.experimentName = expn.read().rstrip('\n')
            print(d)
            with open(join(d, 'log.log'), mode='a', buffering=1) as logfile:
                sys.stdout = logfile
                sys.stderr = logfile
                success=1
                try:
                    run_experiment(batch, logfile, done_filename)
                    print_exc(file=logfile)
                except BaseException as e:
                    logfile.write('run failed\n')
                    print_exc(file=logfile)
                    success=0
                    if type(e) is ResourceExhaustedError:
                        sys.stdout = stdout_orig
                        sys.stderr = stderr_orig
                        raise e
                finally:
                    sys.stdout = stdout_orig
                    sys.stderr = stderr_orig
                with open(fnc, 'w') as fc:
                    fc.write('%d\n' % (success,))





                

