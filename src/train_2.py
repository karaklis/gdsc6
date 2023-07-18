# https://towardsdatascience.com/pytorch-image-classification-tutorial-for-beginners-94ea13f56f2
# common data science packages
import numpy as np
import pandas as pd

# machine learning packages
import optuna  # hyperparameter tuning
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# audio and image packages
import librosa as lbr
from scipy.signal import fftconvolve
from PIL import Image

# various packages
import os
import warnings
from tqdm import tqdm
import joblib
import argparse

# print options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
np.set_printoptions(linewidth=1000)
warnings.filterwarnings('ignore')


# classes

# config
class CFG:
    # global parameters
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_channel", type=str, default=os.environ["SM_CHANNEL_DATA"])
    # parser.add_argument("--test_dir", type=str, default="test")
    # parser.add_argument("--output_dir", type=str, default=os.environ['SM_MODEL_DIR'])
    # args, _ = parser.parse_known_args()
    # output_dir = f"{args.output_dir}/"  # replaces output_dir in local implementation
    # input_root = f"{args.data_channel}/"  # replaces output_dir in local implementation
    input_root = "../"
    output_dir = input_root
    model_location = 'data/models/effnet_buzz_40_1.pkl'
    phase = 'predict'  # 'pretrain', 'train', 'predict'
    pretrain_mode = False
    pretrain_labels = 10
    verbose = 1
    output_stats = False  # output summary stats
    eps = 1e-10  # very small value to prevent division by zero
    save_spectrogram = False

    # training and evaluation
    device = 'cpu'  #  ('cuda' if torch.cuda.is_available() else 'cpu')  # doesn't work with cuda
    batch_size = 64
    epochs = 8 if phase == 'pretrain' else 12  # 20
    min_sample = 40             # upsampling threshold


    # optimizable parameters

    # initial and final learning rate
    model_name = 'efficientnet_b0'
    optimizer = 'Adam'
    learning_rate = 1.5e-3
    lr_min_factor = 1e-1        # for scheduler; 1 means constant LR, 1e-2 means final LR is 1/100 of initial LR

    # audio specifics
    target_sr = 44100           # sampling rate
    # n_mels = 112              # height of spectrogram; 64-128 chosen 112 to match input size for efficientnet b0
    time_long = 40.0            # long time window, taken from original signal
    time_focus = 2.0            # focus point within long time window
    time_step = 38.0            # time offset step forward for splitting up longer recordings
    base_snr = 'std_rms'        # base for signal-noise-ratio: abs_mean, abs_median, std_rms

    # augmentation parameters
    # time domain
    prob_background = .8        # probability of mixing background noise to the signal
    mix_background_snr = .4     # maximal strength of background noise, minimal is half
    prob_ir = .8                # probability of adding impulse response
    prob_white_noise = .5       # probability of mixing white noise to the signal
    mix_white_noise_snr = .2    # maximal strength of white noise, minimal is half
    prob_jitter = .3            # probability of adding sampling jitter
    time_stretch = 1.025        # maximal time stretching
    prob_timestretch = .7       # probability of time stretching
    pitch_shift = 0.5           # number of semitones
    prob_pitchshift = .7        # probability of pitch shifting
    # suspended
    # frequency domain
    # prob_time_mask = .3         # probability of applying time mask
    # width_time_mask = .15       # width of time mask
    # prob_freq_mask = .3         # probability of applying frequency mask
    # width_freq_mask = .15       # width of frequency mask

    # other adjustable parameters
    tmax = 0  # for scheduler; will be set based on number audio chunks that are used for training
    out_features = 0  # number of classes to predict; will be set based on datasets
    weights = [0]  # weights will be calculated from the occurrences of each label

    # random seed
    seed = 2023
    rng = np.random.default_rng(seed)  # random number generator

    # dependent parameters
    spec_width = 224            # width of spectrogram; 2 means square image, which is good for efficientnet b0
    # time_step = (time_focus * time_long ** 2) ** (1 / 3)  # previously
    clip_audio = time_long      # clip noise
    fmax = target_sr // 2       # maximum frequency; fix at half the sampling rate
    fmin = target_sr // 32      # minimum frequency

    # white noise and specifics
    # wn = rng.random(time_long * target_sr) * 2 - 1
    wn = rng.normal(0, 1, int(time_long * target_sr))
    wn_abs_mean = np.mean(np.abs(wn))
    wn_abs_median = np.median(np.abs(wn))
    wn_std_rms = np.std(wn)


# skeleton for loading dataset
class LoadDataset(Dataset):
    def __init__(self, df, train=True, df_audio=pd.DataFrame(), df_noise=pd.DataFrame(), df_ir=pd.DataFrame()):
        self.train = train
        self.df_audio = df_audio
        self.df_noise = df_noise
        self.df_ir = df_ir
        self.df = df
        self.X = df

        if 'label' not in self.df.columns:
            self.df['label'] = 0
        self.y = df['label'].astype('int64')


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spec = transform_audio(
            self.df, self.df_audio, idx=idx, train=self.train, noise_data=self.df_noise, ir_data=self.df_ir)
        mean, std = np.mean(spec), np.std(spec) + CFG.eps  # prevent division by zero
        spec = np.repeat(spec[np.newaxis, :, :], 3, axis=0)  # convert from grayscale to color by repeating 3 times
        if np.isnan(spec).any():  # detected nan
            return None

        if np.all(spec == 0.0) or np.all(spec == 1.0):
            return None

        spec_tensor = torch.from_numpy(spec)
        norm = transforms.Normalize((mean, mean, mean), (std, std, std))
        spec_tensor_norm = norm(spec_tensor)
        return [spec_tensor_norm, torch.from_numpy(self.y[idx].flatten())]

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

# functions

# suspended
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
    # background noise, impulse response, white noise, jitter, time stretch, pitch shift
    bg, bg_snr, ir = aug_plan.bg, aug_plan.bg_snr, aug_plan.ir
    wn, wn_snr, jit, ts, ps = aug_plan.wn, aug_plan.wn_snr, aug_plan.jit, aug_plan.ts, aug_plan.ps
    # get audio specifics
    abs_mean, abs_median, std_rms = audio_specs.abs_mean, audio_specs.abs_median, audio_specs.std_rms
    # augmentation part 1: mix background noise
    audio_c = np.copy(audio)
    if bg != -1:
        # get noise audio and specifics
        noise_abs_mean = noise_data.at[bg, 'abs_mean']
        noise_abs_median = noise_data.at[bg, 'abs_median']
        noise_std_rms = noise_data.at[bg, 'std_rms']
        noise_audio = noise_data.at[bg, 'audio']  # dictionary of noise d
        noise_audio = noise_audio['audio']  # get actual d out of dictionary
        # cut noise audio to match with audio chunk
        if len(noise_audio) < len(audio):  # this can happen due to rounding errors
            noise_audio = np.concatenate([noise_audio] * int(len(audio) // len(noise_audio) + 1))
        if len(noise_audio) > len(audio):
            noise_audio = noise_audio[:len(audio)]
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
    if np.isnan(audio).any() or np.all(audio == 0.0):  # if transform failed, use previous audio
        print('failure at background mix')
        audio = audio_c

    # augmentation part 2: mix white noise
    audio_c = np.copy(audio)
    if wn == 1:
        # determine factor of signal noise ratio according to base snr
        if CFG.base_snr == 'abs_mean':
            snr = abs_mean / CFG.wn_abs_mean * wn_snr
        elif CFG.base_snr == 'abs_median':
            snr = abs_median / CFG.wn_abs_median * wn_snr
        else:
            snr = std_rms / CFG.wn_std_rms * wn_snr
        summary_stats('white noise original', CFG.wn)
        wn_adapted = CFG.wn * snr * CFG.rng.random(1)[0]  # adapt white noise
        summary_stats('white noise adapted', wn_adapted)
        if len(wn_adapted) < len(audio):  # this can happen due to rounding errors
            wn_adapted = np.concatenate([wn_adapted] * int(len(audio) // len(wn_adapted) + 1))
        if len(wn_adapted) > len(audio):
            wn_adapted = wn_adapted[:len(audio)]

        audio += wn_adapted # add adapted noise to signal
        summary_stats('after mixing white noise', audio)
    if np.isnan(audio).any() or np.all(audio == 0.0):  # if transform failed, use previous audio
        print('failure at white noise mix')
        audio = audio_c

    # augmentation part 3: time stretch
    audio_c = np.copy(audio)
    if ts != 1:
        try:
            audio = lbr.effects.time_stretch(audio, rate=ts)
            summary_stats('after time stretch before length adaption', audio)
            # revert length to original audio
            if len(audio) < CFG.time_long * CFG.target_sr:
                audio = np.concatenate([audio] * 2)
            if len(audio) > CFG.time_long * CFG.target_sr:
                audio = audio[:int(CFG.time_long * CFG.target_sr)]
        except lbr.util.exceptions.ParameterError as e:
            print(f'librosa error {e}: audio len: {len(audio)}')
            print(f'isnan: {audio[np.isnan(audio)]}')
            audio = audio_c  # if transform failed, use previous audio
        summary_stats('after time stretch', audio)
    if np.isnan(audio).any() or np.all(audio == 0.0):  # if transform failed, use previous audio
        print('failure at time stretch')
        audio = audio_c

    # augmentation part 4: pitch shift
    audio_c = np.copy(audio)
    if ps != 0:
        try:
            audio = lbr.effects.pitch_shift(audio, sr=CFG.target_sr, n_steps=ps)
            summary_stats('after pitch shift', audio)
            # revert length to original audio
            if len(audio) < CFG.time_long * CFG.target_sr:
                audio = np.concatenate([audio] * 2)
            if len(audio) > CFG.time_long * CFG.target_sr:
                audio = audio[:int(CFG.time_long * CFG.target_sr)]
        except lbr.util.exceptions.ParameterError as e:
            print(f'librosa error {e}: audio len: {len(audio)}')
            print(f'isnan: {audio[np.isnan(audio)]}')
            audio = audio_c  # if transform failed, use previous audio
        summary_stats('after pitch shift', audio)
    if np.isnan(audio).any() or np.all(audio == 0.0):  # if transform failed, use previous audio
        print('failure at time stretch')
        audio = audio_c

    # augmentation part 5: impulse response
    audio_c = np.copy(audio)
    if ir != -1:
        # get impulse response audio
        ir_audio = ir_data.at[ir, 'audio']  # dictionary of impulse response d
        ir_audio = ir_audio['audio']  # get actual d out of dictionary
        audio = fftconvolve(audio, ir_audio, mode='same')
        summary_stats('after convolving impulse response', audio)
    if np.isnan(audio).any() or np.all(audio == 0.0):  # if transform failed, use previous audio
        print('failure at impulse response')
        audio = audio_c

    # augmentation part 6: sampling jitter
    audio_c = np.copy(audio)
    if jit == 1:
        try:
            audio2 = lbr.resample(audio, orig_sr=CFG.target_sr, target_sr=48000)
            audio3 = lbr.resample(audio2, orig_sr=48000, target_sr=45000)
            audio4 = lbr.resample(audio3, orig_sr=45000, target_sr=48600)
            audio = lbr.resample(audio4, orig_sr=48600, target_sr=CFG.target_sr)
            summary_stats('after adding jitter', audio)
        except lbr.util.exceptions.ParameterError as e:
            print(f'librosa error {e}: audio len: {len(audio)}')
            print(f'isnan: {audio[np.isnan(audio)]}')
            audio = audio_c  # if transform failed, use previous audio
    if np.isnan(audio).any() or np.all(audio == 0.0):  # if transform failed, use previous audio
        print('failure at jitter')
        audio = audio_c

    return audio

def calculate_metric(y, y_pred):
  return f1_score(y, y_pred, average='macro')

def calculate_weights(df):
    w = df.groupby(['label'], as_index=False).agg(cnt=('label', 'count'))
    w['label'] = w.label.astype('float32')
    w = w.sort_values(by=['label'])
    w['cnt'] = (1 / w['cnt']).astype('float32')
    return w.cnt.values.flatten()

def display_spec(spec):
    img = Image.fromarray(np.uint8(255.9999 - spec * 255.9999))
    img.show()

def fit(model, optimizer, scheduler, criterion, train_dataloader, valid_dataloader=None):
    f1_list, loss_list, val_f1_list, val_loss_list = [], [], [], []

    for epoch in range(CFG.epochs):
        if CFG.verbose == 1:
            print(f"Epoch {epoch + 1}/{CFG.epochs}")
        set_seed()

        f1, loss = train_one_epoch(train_dataloader, model, optimizer, scheduler, criterion)
        if CFG.verbose == 1:
            print(f'\nTrain Loss: {loss:.4f} F1: {f1:.4f}')
        f1_list.append(f1)
        loss_list.append(loss)

        if valid_dataloader:
            val_f1, val_loss = validate_one_epoch(valid_dataloader, model, criterion)
            if CFG.verbose == 1:
                print(f'\nVal Loss: {val_loss:.4f} Val F1: {val_f1:.4f}')
            val_f1_list.append(val_f1)
            val_loss_list.append(val_loss)

    return f1_list, loss_list, val_f1_list, val_loss_list, model

def get_audio():
    f_metadata = f'{CFG.input_root}data/metadata.csv'
    df_audio = pd.read_csv(f_metadata)[['path', 'label']]
    df_audio['ext'] = df_audio.path.apply(lambda x: x.split('.')[-1])
    df_audio = df_audio[df_audio['ext'].isin(['wav', 'mp3', 'ogg'])].drop(columns=['ext'])
    if CFG.pretrain_mode:
        print(f'full audio data length: {len(df_audio)}')
        df_audio = df_audio[df_audio['label'] < CFG.pretrain_labels]
        print(f'reduced to {len(df_audio)} due to pretrain mode')
    for i, r in df_audio.iterrows():
        ser_audio_meta = get_audio_metadata(CFG.input_root + r['path'])
        df_audio.loc[i, ser_audio_meta.index] = ser_audio_meta
    return df_audio


def get_audio_metadata(path, clip=False):
    try:
        audio, sr = lbr.load(path, sr=CFG.target_sr, mono=True)
        if clip and len(audio) > CFG.clip_audio * CFG.target_sr:
            audio = audio[:int(CFG.clip_audio * CFG.target_sr)]
        audio = audio / (np.max(np.abs(audio)) + CFG.eps)  # standardize
        return pd.Series(
            {'frames': len(audio), 'length': len(audio) / CFG.target_sr, 'max': np.max(audio),
             'min': np.min(audio), 'abs_mean': np.mean(np.abs(audio)), 'abs_median': np.median(np.abs(audio)),
             'std_rms': np.std(audio), 'audio': {'audio': audio}, 'err': 0, 'errmsg': ''})
    except Exception as e:
        print(f'Error on file: {path}, error: {e}')
        return pd.Series(
            {'sr': 0, 'frames': 0, 'length': 0, 'max': 0,
             'min': 0, 'abs_mean': 0, 'abs_median': 0,
             'std_rms': 0, 'audio': 0, 'err': 1, 'errmsg': e})

def get_audio_test():
    df_audio_test = pd.DataFrame(os.listdir(f'{CFG.input_root}data/test/'), columns=['path'])
    df_audio_test.path = df_audio_test.path.apply(lambda x: CFG.input_root + 'data/test/' + x)
    df_audio_test['ext'] = df_audio_test.path.apply(lambda x: x.split('.')[-1])
    df_audio_test = df_audio_test[df_audio_test['ext'].isin(['wav', 'mp3', 'ogg'])].drop(columns=['ext'])
    for i, r in df_audio_test.iterrows():
        ser_audio_meta = get_audio_metadata(r['path'], clip=False)
        df_audio_test.loc[i, ser_audio_meta.index] = ser_audio_meta
    return df_audio_test

def get_focus_point(audio):  # find rolling average the size of the small focus window within big window
    if len(audio) < CFG.time_focus * CFG.target_sr:  # for chunks smaller than the focus window
        return 0.0
    return np.argmax(np.cumsum((np.roll(np.abs(audio), - int(CFG.time_focus * CFG.target_sr)) - np.abs(audio))[
                               :int((CFG.time_long - CFG.time_focus) * CFG.target_sr)])) / CFG.target_sr

def get_impulse_response():
    df_ir = pd.DataFrame(os.listdir(f'{CFG.input_root}data/impulse_response/'), columns=['path'])
    df_ir.path = df_ir.path.apply(lambda x: CFG.input_root + 'data/impulse_response/' + x)
    df_ir['ext'] = df_ir.path.apply(lambda x: x.split('.')[-1])
    df_ir = df_ir[df_ir['ext'].isin(['wav', 'mp3', 'ogg'])].drop(columns=['ext'])

    for i, r in df_ir.iterrows():
        ser_audio_meta = get_audio_metadata(r['path'])
        df_ir.loc[i, ser_audio_meta.index] = ser_audio_meta
    return df_ir

def get_model(model_name='efficientnet_b0', opt='Adam'):
    model = models.efficientnet_b0(pretrained=True)
    for params in model.parameters():  # fine-tuning intermediate layers
        params.requires_grad = True
    model.classifier[1] = nn.Linear(in_features=1280, out_features=CFG.out_features)
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Linear(in_features, out_features=CFG.out_features)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features=CFG.out_features)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Linear(in_features, out_features=CFG.out_features)
    model = model.to(CFG.device)
    criterion = nn.CrossEntropyLoss(weight=CFG.weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=0)
    if opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=CFG.learning_rate, weight_decay=0)
    elif opt == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=CFG.learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.tmax, eta_min=CFG.learning_rate * CFG.lr_min_factor)

    return model, criterion, optimizer, scheduler

def get_noise():
    df_noise = pd.DataFrame(os.listdir(f'{CFG.input_root}data/noise/'), columns=['path'])
    df_noise.path = df_noise.path.apply(lambda x: CFG.input_root + 'data/noise/' + x)
    df_noise['ext'] = df_noise.path.apply(lambda x: x.split('.')[-1])
    df_noise = df_noise[df_noise['ext'].isin(['wav', 'mp3', 'ogg'])].drop(columns=['ext'])
    for i, r in df_noise.iterrows():
        ser_audio_meta = get_audio_metadata(r['path'], clip=True)
        df_noise.loc[i, ser_audio_meta.index] = ser_audio_meta
    return df_noise

def get_spectrogram(audio, height, width):
    audio = np.nan_to_num(audio, copy=True, posinf=0.0, neginf=0.0).astype('float32')  # prevent librosa animosities
    hop_length = int(len(audio) / width)  # determines the width of the spectrogram
    n_fft = hop_length * 4
    spec = lbr.feature.melspectrogram(
        hop_length=hop_length, y=audio, sr=CFG.target_sr, n_mels=height, n_fft=n_fft,
        fmax=CFG.fmax, fmin=CFG.fmin)
    # amin = np.quantile(spec, 0.01) + CFG.eps  # reasonable value for lower bound of spectrogram db
    spec = lbr.power_to_db(spec, ref=np.max(spec) + CFG.eps, amin=CFG.eps)
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + CFG.eps)  # standardize spectrogram
    spec = spec[:, :CFG.spec_width]  # deal with rounding errors
    if np.isnan(spec).any():
        spec = np.nan_to_num(spec)
    spec = spec[:height, :width]
    # display_spec(spec)
    return spec

def objective(trial, train_dataloader=None, valid_dataloader=None):
    params = {  # description of each parameter in class CFG
        # skipped: optimizing model and optimizer: selection efficientnet_b0 and Adam
        # 'model_name': trial.suggest_categorical('model_name', ['efficientnet_b0', 'resnet18', 'alexnet', 'vgg16']),
        # 'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD']),
        # step 1: optimizing learning rates: set to 0.0015 and 0.01
        # 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        # 'lr_min_factor': trial.suggest_loguniform('lr_min_factor', 1e-2, 1.0),
        # skipped: target_sr and n_mels
        # 'target_sr': trial.suggest_loguniform('target_sr', 11025, 44100),
        # 'n_mels': trial.suggest_int('n_mels', 64, 192),
        # step 2: optimizing time windows: optimum: 40 and 1
        'time_long': trial.suggest_loguniform('time_long', 20.0, 60.0),
        'time_focus': trial.suggest_loguniform('time_focus', 1.0, 6.0),
        # skipped: using std_rms as base_snr
        # 'base_snr': trial.suggest_categorical('base_snr', ['abs_mean', 'abs_median', 'std_rms']),
        # skipped: optimizing augmentation parameters
        # 'prob_background': trial.suggest_uniform('prob_background', 0.0, 1.0),
        # 'mix_background_snr': trial.suggest_uniform('mix_background_snr', 0.0, 1.0),
        # 'prob_ir': trial.suggest_uniform('prob_ir', 0.0, 1.0),
        # 'prob_white_noise': trial.suggest_uniform('prob_white_noise', 0.0, 1.0),
        # 'mix_white_noise_snr': trial.suggest_uniform('mix_white_noise_snr', 0.0, 0.5),
        # 'prob_jitter': trial.suggest_uniform('prob_jitter', 0.0, 1.0),
        # 'time_stretch': trial.suggest_uniform('time_stretch', 1.0, 1.059463),  # 2^(1/12), i.e. a half tone
        # 'prob_timestretch': trial.suggest_uniform('prob_timestretch', 0.0, 1.0),
        # skipped: frequency domain parameters
        # 'prob_time_mask': trial.suggest_uniform('prob_time_mask', 0.0, 1.0),
        # 'width_time_mask': trial.suggest_uniform('width_time_mask', 0.0, 0.5),
        # 'prob_freq_mask': trial.suggest_uniform('prob_freq_mask', 0.0, 1.0),
        # 'width_freq_mask': trial.suggest_uniform('width_freq_mask', 0.0, 0.5)
    }
    update_cfg(params)  # update parameters

    # build model from pretrained model
    model, criterion, optimizer, scheduler = get_model(
        model_name=CFG.model_name, opt=CFG.optimizer)
    if CFG.verbose:
        print(f'trial with parms: {output_cfg()}')

    # train and validate
    f1, loss, val_f1, val_loss, model = fit(
        model, optimizer, scheduler, criterion, train_dataloader, valid_dataloader)
    # visualize_history(f1, loss, val_f1, val_loss)
    # CFG.save_spectrogram = False
    return np.min(val_loss)

def output_cfg():
    # return current CFG-parameters as string
    cfg_out = ''
    # cfg_out += f'model_name: {CFG.model_name}, '
    # cfg_out += f'optimizer: {CFG.optimizer}, '
    cfg_out += f'learning_rate: {CFG.learning_rate}, '
    cfg_out += f'lr_min_factor: {CFG.lr_min_factor}, '
    # cfg_out += f'target_sr: {CFG.target_sr}, '
    # cfg_out += f'n_mels: {CFG.n_mels}, '
    cfg_out += f'time_long: {CFG.time_long}, '
    cfg_out += f'time_focus: {CFG.time_focus}, '
    # cfg_out += f'base_snr: {CFG.base_snr}, '
    cfg_out += f'prob_background: {CFG.prob_background}, '
    cfg_out += f'mix_background_snr: {CFG.mix_background_snr}, '
    cfg_out += f'prob_ir: {CFG.prob_ir}, '
    cfg_out += f'prob_white_noise: {CFG.prob_white_noise}, '
    cfg_out += f'mix_white_noise_snr: {CFG.mix_white_noise_snr}, '
    cfg_out += f'prob_jitter: {CFG.prob_jitter}, '
    cfg_out += f'time_stretch: {CFG.time_stretch}, '
    cfg_out += f'prob_timestretch: {CFG.prob_timestretch}, '
    # cfg_out += f'prob_time_mask: {CFG.prob_time_mask}, '
    # cfg_out += f'width_time_mask: {CFG.width_time_mask}, '
    # cfg_out += f'prob_freq_mask: {CFG.prob_freq_mask}, '
    # cfg_out += f'width_freq_mask: {CFG.width_freq_mask}, '
    # cfg_out += f'spec_width: {CFG.spec_width}, '
    cfg_out += f'time_step: {CFG.time_step}\n'
    return cfg_out

def prediction():
    df, df_audio = preprocess_audio_test()
    _, dataloader = prepare_dataset_test(df, df_audio)  # test dataset not needed here

    print('load saved model')
    model = torch.load(f'{CFG.input_root}{CFG.model_location}')

    # Validation mode
    model.eval()

    # Init lists to store y and y-prediction
    final_y, final_y_pred, final_loss = [], [], []

    # Iterate over data
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(CFG.device)
        y = batch[1].to(CFG.device)

        with torch.no_grad():
            # Forward: Get model outputs and calculate loss
            y_pred = model(X)
            prob = torch.nn.functional.softmax(y_pred, dim=1)

            # Convert y and y-prediction to lists
            y = y.detach().cpu().numpy().tolist()
            y_pred = prob.detach().cpu().numpy().tolist()

            # Extend original list
            final_y.extend(y)
            final_y_pred.extend(y_pred)

    rejoin_chunks(df, df_audio, final_y_pred)


def prepare_dataset(df, df_audio, df_noise, df_ir):
    CFG.out_features = df['label'].nunique()  # number of labels to train/validate/predict
    print('split data into train/validation set')

    train_df = df[df.tts == 0].reset_index(drop=True)
    valid_df = df[df.tts == 1].reset_index(drop=True)


    # no augmentations for validation set, so drop all duplicates that were created due to augmentation/upsampling
    valid_df = valid_df.drop_duplicates(subset=['idx', 'offset']).reset_index(drop=True)
    print('build train dataset')
    train_dataset = LoadDataset(train_df, train=True, df_audio=df_audio, df_noise=df_noise, df_ir=df_ir)
    print('build validation dataset')
    valid_dataset = LoadDataset(valid_df, train=False, df_audio=df_audio, df_noise=df_noise, df_ir=df_ir)


    print('load train data into NN')
    train_dataloader = DataLoader(
        train_dataset, batch_size=CFG.batch_size, shuffle=True, collate_fn=LoadDataset.collate_fn)
    print('load val data into NN')
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False)
    CFG.tmax = np.ceil(len(train_dataloader.dataset) / CFG.batch_size) * CFG.epochs  # needed for scheduler

    return train_df, valid_df, train_dataset, valid_dataset, train_dataloader, valid_dataloader, valid_df

def prepare_dataset_test(df, df_audio):
    print('build test dataset')
    test_dataset = LoadDataset(df, train=False, df_audio=df_audio, df_noise=None, df_ir=None)

    print('load train data into NN')
    test_dataloader = DataLoader(
        test_dataset, batch_size=CFG.batch_size, shuffle=False, collate_fn=LoadDataset.collate_fn)
    return test_dataset, test_dataloader

def preprocess_audio():
    # read audio, noise and impulse response data
    print('get audio data...')
    df_audio = get_audio()  # train and validation data
    print('get noise data...')
    df_noise = get_noise()
    print('get impulse response data...')
    df_ir = get_impulse_response()

    # train-test split before splitting audio into chunks and before upsampling to prevent from overfitting
    _, test_idx = train_test_split(df_audio.index.values, test_size=0.225, stratify=df_audio['label'].values)
    df_audio['tts'] = 0
    df_audio.loc[test_idx, ['tts']] = 1

    # split long audios into chunks
    print('split audio into chunks...')
    df = split_audio(df_audio)
    df = df.merge(df_audio[['label', 'tts']], right_index=True, left_on=['idx'])  # merge label and train-test split


    # upsample metadata for labels below a specified minimum
    print('upsampling...')
    df = upsample(df)
    CFG.weights = torch.from_numpy(calculate_weights(df))
    print(f'total set of upsampled chunks: {len(df)}')
    print(f'thereof validation: {len(df[df.tts==1])}')
    print('set augmentation plan...')
    df = set_augmentation_plan(df, num_bg_noise=len(df_noise), num_ir=len(df_ir))

    # output metadata
    # df_noise.loc[:, df_noise.columns != 'audio'].to_csv(f'{CFG.input_root}data/noise_metadata.csv')
    # df_ir.loc[:, df_ir.columns != 'audio'].to_csv(f'{CFG.input_root}data/impulse_response_metadata.csv')
    # df_audio.loc[:, df_audio.columns != 'audio'].to_csv(f'{CFG.input_root}data/audio_metadata.csv')

    return df, df_audio, df_noise, df_ir

# prepare audio data for test dataset
def preprocess_audio_test():
    print('get audio data of test dataset...')
    df_audio_test = get_audio_test()  # test data for prediction
    df_test = split_audio(df_audio_test)
    return df_test, df_audio_test

# hyperparameter tuning with optuna
def pretraining():
    # preprocess training data
    # get audio data, noise and impulse response
    # split audio into equally sized chunks, upsample underrepresented labels
    # finally create augmentation plan for chunks
    df, df_audio, df_noise, df_ir = preprocess_audio()
    # split train/test, prepare datasets and dataloaders
    train_df, valid_df, train_dataset, valid_dataset, train_dataloader, valid_dataloader, valid_df = \
        prepare_dataset(df, df_audio, df_noise, df_ir)

    # call objective
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=CFG.seed),
        pruner=optuna.pruners.HyperbandPruner, study_name='buzz hyperparameter tuning')
    study.optimize(lambda trial: objective(
        trial, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader), n_trials=40, timeout=3*24*3600)
    joblib.dump(study, f'{CFG.input_root}data/models/optuna_study.pkl')
    print(f'best study:\n, {study.best_trial}')

def rejoin_chunks(df, df_audio, pred):
    p = np.array(pred)
    df_audio['file_name'] = df_audio.path.apply(lambda x: x.split('/')[-1])
    cols = [str(x) for x in list(range(p.shape[1]))]
    df_pred = pd.DataFrame(p, columns=cols)
    df = pd.merge(df, df_pred, left_index=True, right_index=True, how='inner')
    df_details_before_rejoin = pd.merge(df_audio[['file_name']], df, left_index=True, right_on='idx')
    df_details_before_rejoin.to_csv(f'{CFG.output_dir}pred_details_40_1_before_rj.csv', index=False)

    # a simple mean calculation for the combined chunks gives poor results, so we will transform and retransform
    # the probabilities to put more weight on chunks that have good detection probabilities
    df[cols] = 1 / (1 - df[cols]) - 1  # transform to put more weight on good detection results
    df = df.groupby(['idx'], as_index=False)[cols].agg('mean')
    df[cols] = 1 - 1 / (df[cols] + 1)  # re-transform to get back to probabilities
    df['predicted_class_id'] = df[cols].idxmax(axis='columns')
    df_details = pd.merge(df_audio[['file_name']], df, left_index=True, right_on='idx')
    df = pd.merge(df_audio[['file_name']], df[['idx', 'predicted_class_id']], left_index=True, right_on='idx')
    print('save predictions')
    df_details.to_csv(f'{CFG.output_dir}pred_details_40_1_rj.csv', index=False)
    df[['file_name', 'predicted_class_id']].to_csv(
        f'{CFG.output_dir}prediction_40_1_rj.csv', index=False)
    print('done')

def save_spec(spec, idx, idxa, offset):
    img = Image.fromarray(np.uint8(255.9999 - spec * 255.9999))
    img.save(f'{CFG.input_root}data/specs/spec_{CFG.time_long}_{CFG.time_focus}_{idxa}_{idx}_{offset}.png')

def save_stats(df, fn):
    df.loc[:, df.columns != 'audio'].to_csv(f'{CFG.input_root}data/stats/{fn}.csv', index=False)

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
    # pitch shifting
    df['ps'] = CFG.rng.random(len(df))
    df.loc[df['ps'] > CFG.prob_pitchshift, ['ps']] = 0
    df.loc[df['ps'] != 0, ['ps']] = (df['ps'] / CFG.prob_pitchshift * 2 - 1) * CFG.pitch_shift

    # suspended
    # frequency domain
    # time masking
    # df['tm'] = CFG.rng.random(len(df))
    # df.loc[df['tm'] > CFG.prob_time_mask, ['tm']] = 0
    # df.loc[df['tm'] != 0, ['tm']] = 1
    # df['wtm'] = CFG.rng.random(len(df)) * CFG.width_time_mask / 2 + CFG.width_time_mask / 2
    # # frequency masking
    # df['fm'] = CFG.rng.random(len(df))
    # df.loc[df['fm'] > CFG.prob_freq_mask, ['fm']] = 0
    # df.loc[df['fm'] != 0, ['fm']] = 1
    # df['wfm'] = CFG.rng.random(len(df)) * CFG.width_freq_mask / 2 + CFG.width_freq_mask / 2
    # df[['bg', 'ir', 'wn', 'jit', 'tm', 'fm']] = df[['bg', 'ir', 'wn', 'jit', 'tm', 'fm']].astype('int')
    df[['bg', 'ir', 'wn', 'jit']] = df[['bg', 'ir', 'wn', 'jit']].astype('int')
    return df

def set_seed():
    CFG.seed += 1
    CFG.rng = np.random.default_rng(CFG.seed)  # random number generator
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)  # In general seed PyTorch operations
    torch.cuda.manual_seed(CFG.seed)  # If you are using CUDA on 1 GPU, seed it
    torch.cuda.manual_seed_all(CFG.seed)  # If you are using CUDA on more than 1 GPU, seed them all
    return CFG.seed

def split_audio(df_audio):   # split long audios into shorter chunks
    dfa = df_audio[df_audio['frames'] <= (CFG.time_long + CFG.time_step) * CFG.target_sr]  # short chunks, adopted 1:1
    df = pd.DataFrame(index=dfa.index).reset_index().rename(columns={'index': 'idx'})
    df['offset'] = 0
    for i, r in df_audio.iterrows():  # extract long audios and split into chunks
        if r['frames'] > (CFG.time_long + CFG.time_step) * CFG.target_sr:
            # calculate number of splits
            num_splits = int((r['frames'] - CFG.time_long * CFG.target_sr) // (CFG.time_step * CFG.target_sr)) + 1
            for j in range(num_splits):  # calculate chunks and concat to d frame
                df = pd.concat(
                    [df, pd.DataFrame({'idx': int(i), 'offset': j * CFG.time_step}, index=[0])],
                    ignore_index=[1], axis=0)
    df['idx'] = df['idx'].astype('int')
    return df.reset_index(drop=True)

def summary_stats(comment, d):
    if CFG.output_stats:
        print(f'\n{comment}:')
        print(f'shape: {np.shape(d)}, mean: {np.mean(d)}, min/max: {np.min(d)}, {np.max(d)}')
        print(f'rms_std: {np.std(d)}, abs_mean: {np.mean(np.abs(d))}, abs_median: {np.median(np.abs(d))}')
        print(f'data:\n{d}')


def training():
    # preprocess training data
    # get audio data, noise and impulse response
    # split audio into equally sized chunks, upsample underrepresented labels
    # finally create augmentation plan for chunks
    df, df_audio, df_noise, df_ir = preprocess_audio()
    save_stats(df, 'augm_plan')
    save_stats(df_audio, 'audio_stats')
    save_stats(df_noise, 'noise_stats')
    save_stats(df_ir, 'ir_stats')
    # split train/test, prepare datasets and dataloaders
    train_df, valid_df, train_dataset, valid_dataset, train_dataloader, valid_dataloader, valid_df = \
        prepare_dataset(df, df_audio, df_noise, df_ir)

    # build model from pretrained model
    model, criterion, optimizer, scheduler = get_model(
        model_name=CFG.model_name, opt=CFG.optimizer)

    # train and validate
    f1, loss, val_f1, val_loss, model = fit(
        model, optimizer, scheduler, criterion, train_dataloader, valid_dataloader)
    print(f'metrics for saved model: loss: {loss}, f1 score: {f1}')
    torch.save(model, f'{CFG.input_root}{CFG.model_location}')


def train_one_epoch(dataloader, model, optimizer, scheduler, criterion):
    # Training mode
    model.train()

    # Init lists to store y and y-prediction
    final_y, final_y_pred, final_loss = [], [], []

    # Iterate over data
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(CFG.device)
        y = batch[1].to(CFG.device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # Forward: Get model outputs and calculate loss
            y_pred = model(X)
            prob = torch.nn.functional.softmax(y_pred, dim=1)
            # print(f'vor dem abkack: y:\n{y.detach().cpu().numpy()}\nypred:\n{y_pred.detach().cpu().numpy()}probabilities:\n{prob.detach().cpu().numpy()}')
            loss = criterion(y_pred, torch.squeeze(y))

            # Convert y and y-prediction to lists
            y = y.detach().cpu().numpy().tolist()
            # y_pred = y_pred.detach().cpu().numpy().tolist()

            # Extend original list
            final_y.extend(y)
            # final_y_pred.extend(y_pred)
            final_y_pred.extend(prob.detach().cpu().numpy().tolist())
            final_loss.append(loss.item())

            # Backward: Optimize
            loss.backward()
            optimizer.step()

        scheduler.step()

    # Calculate statistics
    loss = np.mean(final_loss)
    final_y_pred = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred)

    return metric, loss

# transform audio to spectrogram; only training d will get augmentations
def transform_audio(df, dfa, idx=0, train=True, noise_data=pd.DataFrame(), ir_data=pd.DataFrame()):
    idxa = df.at[idx, 'idx']  # get index of audio
    audio = dfa.at[idxa, 'audio']  # get audio dictionary
    audio = audio['audio']  # get audio data from dictionary
    offset = df.at[idx, 'offset']
    # print(f'plan at idx {idx}:\n{df.loc[[idx]]}\nwith audio at {idxa}:\n{dfa.loc[[idxa]]}')
    if np.isnan(audio).any() or np.all(audio == 0.0):  # if transform failed, use previous audio
        print(f'original audio has nan or is all zero at index {idxa}: {dfa.loc[[idxa]].T}')

    # if length of audio is smaller than the time of the long window, multiply audio accordingly
    if len(audio) < CFG.time_long * CFG.target_sr:
        audio = np.concatenate([audio] * int(CFG.time_long * CFG.target_sr // len(audio) + 1))
    # get chunk of audio, starting from offset and length of long time window
    audio = audio[int(offset * CFG.target_sr) :
                  int((offset + CFG.time_long) * CFG.target_sr)]
    if np.isnan(audio).any() or np.all(audio == 0.0):  # if transform failed, use previous audio
        print(f'audio chunk has nan or is all zero at {idxa}: {dfa.loc[[idxa]].to_frame().T} and offset: {offset}')

    summary_stats('audio signal chunk', audio)
    audio = np.nan_to_num(audio, posinf=0.0, neginf=0.0).astype('float32')

    # augmentations time domain
    if train:
        audio = augment_time_domain(
            audio, aug_plan=df.loc[idx], audio_specs=dfa.loc[idxa], noise_data=noise_data, ir_data=ir_data)
        if np.isnan(audio).any():
            print(f'augmented audio has nan at index {idxa}: {dfa.loc[[idxa]].to_frame().T}')

    # determine focus point, get audio at four focus points
    # split audio into four parts
    l = len(audio) // 4
    audio1, audio2, audio3, audio4 = audio[:l], audio[l:l*2], audio[l*2:l*3], audio[l*3:]

    fp1, fp2, fp3, fp4 = \
        get_focus_point(audio1), get_focus_point(audio2), get_focus_point(audio3), get_focus_point(audio4)
    audio_focus1 = np.tile(audio, 2)[int(fp1 * CFG.target_sr) : int((fp1 + CFG.time_focus) * CFG.target_sr)]
    audio_focus2 = np.tile(audio, 2)[int(fp2 * CFG.target_sr) : int((fp2 + CFG.time_focus) * CFG.target_sr)]
    audio_focus3 = np.tile(audio, 2)[int(fp3 * CFG.target_sr) : int((fp3 + CFG.time_focus) * CFG.target_sr)]
    audio_focus4 = np.tile(audio, 2)[int(fp4 * CFG.target_sr) : int((fp4 + CFG.time_focus) * CFG.target_sr)]
    # summary_stats('after getting audio focus', audio_focus)
    # if np.isnan(audio_focus).any():
    #     print(f'focus audio has nan at index {idxa}: {dfa.loc[idxa]}')


    # get spectrogram of long window and small focus window
    spec = get_spectrogram(audio, width=224, height=32)
    spec1 = get_spectrogram(audio_focus1, width=112, height=96)
    spec2 = get_spectrogram(audio_focus2, width=112, height=96)
    spec3 = get_spectrogram(audio_focus3, width=112, height=96)
    spec4 = get_spectrogram(audio_focus4, width=112, height=96)
    # if np.isnan(spec).any():
    #     print(f'original spectrogram has nan at index {idxa}: {dfa.loc[idxa]}')
    # if np.isnan(spec_focus).any():
    #     print(f'original spectrogram focus has nan at index {idxa}: {dfa.loc[idxa]}')
    # summary_stats('after getting spectrogram', spec)
    # summary_stats('and spectrogram for focus', spec_focus)

    # suspended
    # augmentations frequency domain
    # if train:
    #     spec = augment_freq_domain(spec, aug_plan=df.loc[idx], focus=False)
    #     spec_focus = augment_freq_domain(spec_focus, aug_plan=df.loc[idx], focus=True)
    #     if np.isnan(spec).any():
    #         print(f'spectrogram after augmentation has nan at index {idxa}: {dfa.loc[idxa]}')
    #     if np.isnan(spec_focus).any():
    #         print(f'spectrogram focus after augmentation has nan at index {idxa}: {dfa.loc[idxa]}')

    # combine spectrograms of focus and chunk

    spec12 = np.concatenate([spec1, spec2], axis=1)
    spec34 = np.concatenate([spec3, spec4], axis=1)
    # print(spec.shape, spec12.shape, spec34.shape)
    spec = np.concatenate([spec, spec12, spec34], axis=0)

    summary_stats('after combining spectrograms', spec)
    if np.isnan(spec).any():
        print(f'spectrogram after combining has nan at index {idxa}: {dfa.loc[idxa]}')

    if CFG.save_spectrogram:
        save_spec(spec, idx, idxa, offset)

    return spec.astype('float32')

def update_cfg(params):
    # update from optimization parameters
    # CFG.model_name = params['model_name']
    # CFG.optimizer = params['optimizer']
    # CFG.learning_rate = params['learning_rate']
    # CFG.lr_min_factor = params['lr_min_factor']
    # CFG.target_sr = params['target_sr']
    # CFG.n_mels = params['n_mels']
    CFG.time_long = int(params['time_long'])
    CFG.time_focus = int(params['time_focus'])
    # CFG.base_snr = params['base_snr']
    # CFG.prob_background = params['prob_background']
    # CFG.mix_background_snr = params['mix_background_snr']
    # CFG.prob_ir = params['prob_ir']
    # CFG.prob_white_noise = params['prob_white_noise']
    # CFG.mix_white_noise_snr = params['mix_white_noise_snr']
    # CFG.prob_jitter = params['prob_jitter']
    # CFG.time_stretch = params['time_stretch']
    # CFG.prob_timestretch = params['prob_timestretch']
    # CFG.prob_time_mask = params['prob_time_mask']
    # CFG.width_time_mask = params['width_time_mask']
    # CFG.prob_freq_mask = params['prob_freq_mask']
    # CFG.width_freq_mask = params['width_freq_mask']

    # update dependent parameters
    # CFG.spec_width = CFG.n_mels * 2     # width of spectrogram
    # CFG.time_step = (CFG.time_focus * CFG.time_long ** 2) ** (1 / 3)  # previously
    CFG.time_step = CFG.time_long / 2 + CFG.time_focus  # time offset step forward
    # CFG.clip_audio = CFG.time_long      # clip noise
    # CFG.fmax = CFG.target_sr // 2       # maximum frequency; fix at half the sampling rate
    # CFG.fmin = CFG.target_sr // 40      # minimum frequency
    # CFG.wn = CFG.rng.random(int(CFG.time_long * CFG.target_sr)) * 2 - 1
    # CFG.wn_abs_mean = np.mean(np.abs(CFG.wn))
    # CFG.wn_abs_median = np.median(np.abs(CFG.wn))
    # CFG.wn_std_rms = np.std(CFG.wn)

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


def validate_one_epoch(dataloader, model, criterion):
    # Validation mode
    model.eval()

    # Init lists to store y and y-prediction
    final_y, final_y_pred, final_loss = [], [], []

    # Iterate over data
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(CFG.device)
        y = batch[1].to(CFG.device)

        with torch.no_grad():
            # Forward: Get model outputs and calculate loss
            y_pred = model(X)
            prob = torch.nn.functional.softmax(y_pred, dim=1)
            loss = criterion(y_pred, torch.squeeze(y))

            # Convert y and y-prediction to lists
            y = y.detach().cpu().numpy().tolist()
            # y_pred = y_pred.detach().cpu().numpy().tolist()

            # Extend original list
            final_y.extend(y)
            final_y_pred.extend(prob.detach().cpu().numpy().tolist())
            final_loss.append(loss.item())

    # Calculate statistics
    loss = np.mean(final_loss)
    final_y_pred = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred)

    return metric, loss

# main routine
def main():
    if CFG.phase == 'pretrain':
        pretraining()
    elif CFG.phase == 'train':
        training()
    else:
        prediction()

if __name__ == '__main__':
    main()

