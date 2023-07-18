import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import librosa as lbr
# import librosa.display as lid
import warnings
# from tqdm import tqdm
from termcolor import colored
from scipy.signal import fftconvolve, medfilt2d
from sklearn.model_selection import StratifiedKFold

import os
from PIL import Image

# print options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
np.set_printoptions(linewidth=1000)
warnings.filterwarnings('ignore')

# config
class CFG:
    # global settings
    input_root = "../"
    output_melspec = False  # output
    seed = 2023
    rng = np.random.default_rng(seed)  # random number generator
    eps = 1e-10  # very small value to prevent division by zero
    debug_mode = True
    num_fold = 5  # for stratified K-Fold

    # audio specifics
    target_sr = 44100           # sampling rate
    fmax = target_sr // 2       # maximum frequency; fix at half the sampling rate
    fmin = target_sr // 40      # minimum frequency
    n_mels = 128                # height of spectrogram
    spec_width = n_mels * 4     # width of spectrogram
    time_long = 30              # long time window, taken from original signal
    time_focus = 3              # focus point within long time window
    time_step = (time_focus * time_long ** 2) ** (1 / 3)  # time offset step forward for splitting up longer recordings
    clip_audio = time_long      # clip noise
    base_snr = 'std_rms'       # base for signal-noise-ratio: abs_mean, abs_median, std_rms

    # augmentation parameters
    min_sample = 50             # upsampling threshold
    # time domain
    prob_background = .7        # probability of mixing background noise to the signal
    mix_background_snr = .7     # maximal strength of background noise, minimal is half
    prob_ir = .7                # probability of adding impulse response
    prob_white_noise = .7       # probability of mixing white noise to the signal
    mix_white_noise_snr = .7    # maximal strength of white noise, minimal is half
    prob_jitter = .3            # probability of adding sampling jitter
    time_stretch = 1.05         # maximal time stretching
    prob_timestretch = .5       # probability of time stretching
    # frequency domain
    kernel_size = 7             # for mask that is used to filter out noise from spectrogram
    snr_quantile = .25          # determine base value for determining signal vs. noise
    snr_threshold = 1.5         # factor of base value, above that is considered signal
    prob_time_mask = .3         # probability of applying time mask
    width_time_mask = .15       # width of time mask
    prob_freq_mask = .3         # probability of applying frequency mask
    width_freq_mask = .15       # width of frequency mask

    # white noise and specifics
    wn = rng.random(time_long * target_sr) * 2 - 1
    wn_abs_mean = np.mean(np.abs(wn))
    wn_abs_median = np.median(np.abs(wn))
    wn_std_rms = np.std(wn)


# functions and classes

def augment_freq_domain(spec, aug_plan, focus=False):
    tm, wtm, fm, wfm = aug_plan.tm, aug_plan.wtm, aug_plan.fm, aug_plan.wfm  # time and frequency masking
    width, height = np.shape(spec)
    # time masking
    if tm == 1 and not focus:  # no time masking for focus window
        tm_start = int(width * ((wtm * 100) % 1))
        tm_end = (tm_start + int(width * wtm)) % width
        spec[tm_start:tm_end,:] = 0
        summary_stats('after time masking', spec)
    # frequency masking
    if fm == 1:
        fm_start = int(height * ((wfm * 100) % 1))
        fm_end = (fm_start + int(height * wfm)) % height
        spec[:, fm_start:fm_end] = 0
        summary_stats('after frequency masking', spec)
    return spec

def augment_time_domain(audio, aug_plan, audio_specs, noise_data=pd.DataFrame(), ir_data=pd.DataFrame()):
    # get specifics for augmentations from augmentation plan
    bg, bg_snr, ir = aug_plan.bg, aug_plan.bg_snr, aug_plan.ir  # background noise, impulse response
    wn, wn_snr, jit, ts = aug_plan.wn, aug_plan.wn_snr, aug_plan.jit, aug_plan.ts  # white noise, jitter, time stretch
    # get audio specifics
    abs_mean, abs_median, std_rms = audio_specs.abs_mean, audio_specs.abs_median, audio_specs.std_rms
    # augmentation part 1: mix background noise
    if bg != -1:
        # get noise audio and specifics
        noise_abs_mean = noise_data.at[bg, 'abs_mean']
        noise_abs_median = noise_data.at[bg, 'abs_median']
        noise_std_rms = noise_data.at[bg, 'std_rms']
        noise_audio = noise_data.at[bg, 'audio']  # dictionary of noise d
        noise_audio = noise_audio['audio']  # get actual d out of dictionary
        # cut noise audio to match with audio chunk
        if len(noise_audio) < CFG.time_long * CFG.target_sr:
            noise_audio = np.concatenate([noise_audio] * (CFG.time_long * CFG.target_sr // len(noise_audio) + 1))
        if len(noise_audio) > CFG.time_long * CFG.target_sr:
            noise_audio = noise_audio[:CFG.time_long * CFG.target_sr]
        # determine factor of signal noise ratio according to base snr
        if CFG.base_snr == 'abs_mean':
            snr = abs_mean / noise_abs_mean * bg_snr
        elif CFG.base_snr == 'abs_median':
            snr = abs_median / noise_abs_median * bg_snr
        else:
            snr = std_rms / noise_std_rms * bg_snr
        summary_stats('background noise original', noise_audio)
        noise_audio *= snr # add adapted noise to signal
        summary_stats('background noise adapted', noise_audio)
        audio += noise_audio  # add adapted noise to signal
        summary_stats('after mixing background noise', audio)
    # augmentation part 2: mix white noise
    if wn == 1:
        # determine factor of signal noise ratio according to base snr
        if CFG.base_snr == 'abs_mean':
            snr = abs_mean / CFG.wn_abs_mean * wn_snr
        elif CFG.base_snr == 'abs_median':
            snr = abs_median / CFG.wn_abs_median * wn_snr
        else:
            snr = std_rms / CFG.wn_std_rms * wn_snr
        summary_stats('white noise original', CFG.wn)
        wn_adapted = CFG.wn * snr  # adapt white noise
        summary_stats('white noise adapted', wn_adapted)
        audio += wn_adapted # add adapted noise to signal
        summary_stats('after mixing white noise', audio)
    # augmentation part 3: time stretch
    if ts != 1:
        audio = lbr.effects.time_stretch(audio, rate=ts)
        summary_stats('after time stretch before length adaption', audio)
        # revert length to original audio
        if len(audio) < CFG.time_long * CFG.target_sr:
            audio = np.concatenate([audio] * 2)
        if len(audio) > CFG.time_long * CFG.target_sr:
            audio = audio[:CFG.time_long * CFG.target_sr]
        summary_stats('after time stretch', audio)
    # augmentation part 4: impulse response
    if ir != -1:
        # get impulse response audio
        ir_audio = ir_data.at[ir, 'audio']  # dictionary of impulse response d
        ir_audio = ir_audio['audio']  # get actual d out of dictionary
        audio = fftconvolve(audio, ir_audio, mode='same')
        summary_stats('after convolving impulse response', audio)
    # sampling jitter
    if jit == 1:
        audio = lbr.resample(audio, orig_sr=CFG.target_sr, target_sr=48000)
        audio = lbr.resample(audio, orig_sr=48000, target_sr=45000)
        audio = lbr.resample(audio, orig_sr=45000, target_sr=48600)
        audio = lbr.resample(audio, orig_sr=48600, target_sr=CFG.target_sr)
        summary_stats('after adding jitter', audio)
    return audio

def display_spec(spec):
    img = Image.fromarray(np.uint8(255.9999 - spec * 255.9999))
    img.show()

def get_audio():
    f_metadata = f'{CFG.input_root}data/metadata.csv'
    df_audio = pd.read_csv(f_metadata)[['path', 'label']]
    df_audio['ext'] = df_audio.path.apply(lambda x: x.split('.')[-1])
    df_audio = df_audio[df_audio['ext'].isin(['wav', 'mp3', 'ogg'])].drop(columns=['ext'])
    if CFG.debug_mode:
        df_audio = df_audio[df_audio['label'] < 5]
    for i, r in df_audio.iterrows():
        ser_audio_meta = get_audio_metadata(CFG.input_root + r['path'])
        df_audio.loc[i, ser_audio_meta.index] = ser_audio_meta
    return df_audio


def get_audio_metadata(path, clip=False):
    try:
        audio, sr = lbr.load(path, sr=CFG.target_sr, mono=True)
        if clip:
            audio = audio[:CFG.clip_audio * CFG.target_sr]
        audio = audio / np.max(np.abs(audio))  # standardize
        return pd.Series(
            {'frames': len(audio), 'length': len(audio) / CFG.target_sr, 'max': np.max(audio),
             'min': np.min(audio), 'abs_mean': np.mean(np.abs(audio)), 'abs_median': np.median(np.abs(audio)),
             'std_rms': np.std(audio), 'audio': {'audio': audio}, 'err': 0, 'errmsg': ''})
    except Exception as e:
        print(colored(f'Error on file: {path}, error: {e}', 'red'))
        return pd.Series(
            {'sr': 0, 'frames': 0, 'length': 0, 'max': 0,
             'min': 0, 'abs_mean': 0, 'abs_median': 0,
             'std_rms': 0, 'audio': 0, 'err': 1, 'errmsg': e})

def get_focus_point(audio):  # find rolling average the size of the small focus window within big window
    if len(audio) < CFG.time_focus * CFG.target_sr:  # for chunks smaller than the focus window
        return 0.0
    return np.argmax(np.cumsum((np.roll(audio, - int(CFG.time_focus * CFG.target_sr)) - audio)[
                               :int((CFG.time_long - CFG.time_focus) * CFG.target_sr)])) / CFG.target_sr


# get focus point for all audio chunks: will be removed later, since we need to do augmentations individually first
def get_focus_point_temp(df, dfa):
    for i, r in df.iterrows():
        audio = dfa.at[r['idx'], 'audio']
        audio = audio['audio']  # extract audio from dictionary that is stored in d frame
        audio = audio[int(r['offset']*CFG.target_sr) : int((r['offset']+CFG.time_long)*CFG.target_sr)]  # extract chunk
        df.loc[i, ['focus']] = get_focus_point(audio)
    return df


def get_impulse_response():
    df_ir = pd.DataFrame(os.listdir(f'{CFG.input_root}data/impulse_response/'), columns=['path'])
    df_ir.path = df_ir.path.apply(lambda x: CFG.input_root + 'data/impulse_response/' + x)
    df_ir['ext'] = df_ir.path.apply(lambda x: x.split('.')[-1])
    df_ir = df_ir[df_ir['ext'].isin(['wav', 'mp3', 'ogg'])].drop(columns=['ext'])

    for i, r in df_ir.iterrows():
        ser_audio_meta = get_audio_metadata(r['path'])
        df_ir.loc[i, ser_audio_meta.index] = ser_audio_meta
    return df_ir

def get_noise():
    df_noise = pd.DataFrame(os.listdir(f'{CFG.input_root}data/noise/'), columns=['path'])
    df_noise.path = df_noise.path.apply(lambda x: CFG.input_root + 'data/noise/' + x)
    df_noise['ext'] = df_noise.path.apply(lambda x: x.split('.')[-1])
    df_noise = df_noise[df_noise['ext'].isin(['wav', 'mp3', 'ogg'])].drop(columns=['ext'])
    for i, r in df_noise.iterrows():
        ser_audio_meta = get_audio_metadata(r['path'], clip=True)
        df_noise.loc[i, ser_audio_meta.index] = ser_audio_meta
    return df_noise

def get_spectrogram(audio):
    hop_length = int(len(audio) / CFG.spec_width)
    n_fft = hop_length * 4
    spec = lbr.feature.melspectrogram(
        hop_length=hop_length, y=audio, sr=CFG.target_sr, n_mels=CFG.n_mels, n_fft=n_fft,
        fmax=CFG.fmax, fmin=CFG.fmin)
    amin = np.quantile(spec, 0.01) + CFG.eps  # reasonable value for lower bound of spectrogram db
    spec = lbr.power_to_db(spec, ref=np.max(spec), amin=amin)
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + CFG.eps)  # standardize spectrogram
    spec = spec[:, :CFG.spec_width]  # deal with rounding errors
    # display_spec(spec)
    return spec

def print_metadata(metadata):
    if len(metadata) > 100:
        print(metadata.sample(100))
    else:
        print(metadata)
    print(metadata.info())
    print(metadata.describe())

def separate_signal_noise(spec):
    display_spec(spec)
    mask = medfilt2d(spec, kernel_size=CFG.kernel_size)
    thres = np.quantile(mask, CFG.snr_quantile, axis=None)
    thres_0 = np.quantile(mask, CFG.snr_quantile, axis=0)
    thres_1 = np.quantile(mask, CFG.snr_quantile, axis=1)
    med = (np.zeros(spec.shape) + np.expand_dims(thres_0, 0) + np.expand_dims(thres_1, 1) + thres) / 3 * CFG.snr_threshold
    spec -= med
    spec[spec < 0.0] = 0.0
    spec /= np.max(spec)
    display_spec(spec)
    return spec

def set_augmentation_plan(df, num_bg_noise=1, num_ir=1):
    # time domain
    # background noise
    df['bg'] = CFG.rng.random(len(df))
    df.loc[df['bg'] > CFG.prob_background, ['bg']] = -1
    df.loc[df['bg'] != -1, ['bg']] = (df['bg'] / CFG.prob_background * num_bg_noise).astype('int')
    df['bg_snr'] = CFG.rng.random(len(df)) * CFG.mix_background_snr / 2 + CFG.mix_background_snr / 2
    # impulse response
    df['ir'] = CFG.rng.random(len(df))
    df.loc[df['ir'] > CFG.prob_ir, ['ir']] = -1
    df.loc[df['ir'] != -1, ['ir']] = (df['ir'] / CFG.prob_ir * num_ir).astype('int')
    # white noise
    df['wn'] = CFG.rng.random(len(df))
    df.loc[df['wn'] > CFG.prob_white_noise, ['wn']] = 0
    df.loc[df['wn'] != 0, ['wn']] = 1
    df['wn_snr'] = CFG.rng.random(len(df)) * CFG.mix_white_noise_snr / 2 + CFG.mix_white_noise_snr / 2
    # sampling jitter
    df['jit'] = CFG.rng.random(len(df))
    df.loc[df['jit'] > CFG.prob_jitter, ['jit']] = 0
    df.loc[df['jit'] != 0, ['jit']] = 1
    # time stretching
    df['ts'] = CFG.rng.random(len(df))
    df.loc[df['ts'] > CFG.prob_timestretch, ['ts']] = 1.0
    df.loc[df['ts'] != 1, ['ts']] = CFG.time_stretch ** (df['ts'] / CFG.prob_timestretch * 2 - 1)
    # frequency domain
    # time masking
    df['tm'] = CFG.rng.random(len(df))
    df.loc[df['tm'] > CFG.prob_time_mask, ['tm']] = 0
    df.loc[df['tm'] != 0, ['tm']] = 1
    df['wtm'] = CFG.rng.random(len(df)) * CFG.width_time_mask / 2 + CFG.width_time_mask / 2
    # frequency masking
    df['fm'] = CFG.rng.random(len(df))
    df.loc[df['fm'] > CFG.prob_freq_mask, ['fm']] = 0
    df.loc[df['fm'] != 0, ['fm']] = 1
    df['wfm'] = CFG.rng.random(len(df)) * CFG.width_freq_mask / 2 + CFG.width_freq_mask / 2
    df[['bg', 'ir', 'wn', 'jit', 'tm', 'fm']] = df[['bg', 'ir', 'wn', 'jit', 'tm', 'fm']].astype('int')
    return df

def split_audio(df_audio):   # split long audios into shorter chunks
    dfa = df_audio[df_audio['frames'] <= (CFG.time_long + CFG.time_step) * CFG.target_sr]  # short chunks, adopted 1:1
    df = pd.DataFrame(index=dfa.index).reset_index().rename(columns={'index': 'idx'})
    df['offset'] = 0

    print(df_audio.describe())
    print(df.describe())
    print((df_audio['frames'] <= (CFG.time_long + CFG.time_step) * CFG.target_sr).describe())



    for i, r in df_audio.iterrows():  # extract long audios and split into chunks
        if r['frames'] > (CFG.time_long + CFG.time_step) * CFG.target_sr:
            # calculate number of splits
            num_splits = int((r['frames'] - CFG.time_long * CFG.target_sr) // (CFG.time_step * CFG.target_sr)) + 1
            for j in range(num_splits):  # calculate chunks and concat to d frame
                df = pd.concat(
                    [df, pd.DataFrame({'idx': int(i), 'offset': j * CFG.time_step}, index=[0])],
                    ignore_index=[1], axis=0)
    df['idx'] = df['idx'].astype('int')
    return df

def summary_stats(comment, d):
    print(f'\n{comment}:')
    print(f'shape: {np.shape(d)}, mean: {np.mean(d)}, min/max: {np.min(d)}, {np.max(d)}')
    print(f'rms_std: {np.std(d)}, abs_mean: {np.mean(np.abs(d))}, abs_median: {np.median(np.abs(d))}')
    print(f'data:\n{d}')

# transform audio to spectrogram; only training d will get augmentations
def transform_audio(df, dfa, idx=0, train=True, noise_data=pd.DataFrame(), ir_data=pd.DataFrame()):
    print('transform')
    # get audio, multiply if too small, and select the right chunk
    idxa = df.at[idx, 'idx']  # get index of audio
    audio = dfa.at[idxa, 'audio']  # get audio dictionary
    audio = audio['audio']  # get audio d from dictionary

    # if length of audio is smaller than the time of the long window, multiply audio accordingly
    if len(audio) < CFG.time_long * CFG.target_sr:
        audio = np.concatenate([audio] * (CFG.time_long * CFG.target_sr // len(audio) + 1))
    # get chunk of audio, starting from offset and length of long time window
    audio = audio[int(df.at[idx, 'offset'] * CFG.target_sr) :
                  int((df.at[idx, 'offset'] + CFG.time_long) * CFG.target_sr)]
    summary_stats('audio signal chunk', audio)

    # augmentations time domain
    if train:
        print(f'idx: {idx}, idxa: {idxa}')
        audio = augment_time_domain(
            audio, aug_plan=df.loc[idx], audio_specs=dfa.loc[idxa], noise_data=noise_data, ir_data=ir_data)

    # determine focus point, get audio at focus point
    focus_point = get_focus_point(audio)
    audio_focus = audio[int(focus_point * CFG.target_sr) : int((focus_point + CFG.time_focus) * CFG.target_sr)]
    summary_stats('after getting audio focus', audio_focus)

    # get spectrogram of long window and small focus window
    spec, spec_focus = get_spectrogram(audio), get_spectrogram(audio_focus)
    summary_stats('after getting spectrogram', spec)
    summary_stats('and spectrogram for focus', spec_focus)

    # separate signal from noise using specified threshold

    spec, spec_focus = separate_signal_noise(spec), separate_signal_noise(spec_focus)

    # augmentations frequency domain
    if train:
        spec = augment_freq_domain(spec, aug_plan=df.loc[idx], focus=False)
        spec_focus = augment_freq_domain(spec_focus, aug_plan=df.loc[idx], focus=True)

    # combine spectrograms of focus and chunk
    spec = np.concatenate([spec, spec_focus], axis=0)
    summary_stats('after combining spectrograms', spec)

    return spec

def upsample(df):  # upsample metadata for labels below a specified minimum
    # get number of audio chunks per label
    df_label_stats = df.groupby(['label'], as_index=False).agg(cnt=('label', 'count'))
    df['upsampled'] = False
    # calc factor to multiply labels to reach at least the minimum number of chunks to be upsampled to
    df_label_stats['upsample_factor'] = CFG.min_sample // df_label_stats['cnt']
    # print(df_label_stats)
    for _, r in df_label_stats.iterrows():
        if r['upsample_factor'] > 0:
            df_tmp = (df[df['label'] == r['label']]).copy()
            # multiply audio chunks
            df_tmp = pd.concat([df_tmp] * r['upsample_factor'], ignore_index=True)
            # remove overhead to reduce to minimum sample size per label
            df_tmp = df_tmp.head(CFG.min_sample - r['cnt'])
            df_tmp['upsampled'] = True  # marker for upsampled chunks
            df = pd.concat([df, df_tmp], ignore_index=True)
    return df.reset_index(drop=True)

# main routine
def main():
    # read audio, noise and impulse response d
    print('get audio data...')
    df_audio = get_audio()  # train and validation d
    print('get noise data...')
    df_noise = get_noise()
    print('get impulse response data...')
    df_ir = get_impulse_response()

    # print_metadata(df_noise)
    # print_metadata(df_ir)
    # print_metadata(df_audio)

    # split long audios into chunks
    print('split audio into chunks...')
    df_train = split_audio(df_audio)
    df_train = df_train.merge(df_audio[['label']], right_index=True, left_on=['idx'])  # merge label info

    # find focus point within audio chunks
    # print('determine focus point within audio chunks')
    # df_train = get_focus_point_temp(df_train, df_audio)

    # upsample metadata for labels below a specified minimum
    print('upsampling...')
    df_train = upsample(df_train)

    # print('after upsampling:')
    # print(df_train.groupby(['label'], as_index=False).agg(cnt=('label', 'count')))
    #
    print('set augmentation plan...')
    df_train = set_augmentation_plan(df_train, num_bg_noise=len(df_noise), num_ir=len(df_ir))
    # print(df_train.head(100))
    # print(df_train.tail(100))
    print(df_train.sample(100))

    # output metadata
    # df_noise.loc[:, df_noise.columns != 'audio'].to_csv(f'{CFG.input_root}data/noise_metadata.csv')
    # df_ir.loc[:, df_ir.columns != 'audio'].to_csv(f'{CFG.input_root}data/impulse_response_metadata.csv')
    # df_audio.loc[:, df_audio.columns != 'audio'].to_csv(f'{CFG.input_root}data/audio_metadata.csv')

    print('transform audio for sample...')
    # idx: 345, 729, 512; debug mode: 1, 100
    spec = transform_audio(df_train, df_audio, idx=100, train=True, noise_data=df_noise, ir_data=df_ir)
    print(spec)
    display_spec(spec)

    # for later when creating datasets
    # np.tile(A, reps)  # repeat values of array

    # define folds for stratified K-fold
    skf = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['label'])):
        df_train.loc[val_idx, 'fold'] = fold


if __name__ == '__main__':
    main()

