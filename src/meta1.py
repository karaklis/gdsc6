import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa as lbr
import librosa.display as lid
import warnings
from tqdm import tqdm
from termcolor import colored

# print options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(linewidth=1000)
warnings.filterwarnings('ignore')

# config
class CFG:
    # files and paths
    input_root = "../"

    # audio specifics
    target_sr = 44100
    fmax = target_sr // 2
    fmin = target_sr // 40
    n_mels = 200  # 128
    n_fft = 2028
    hop_length = 384

    output_melspec = False



def main():
    # main routine
    f_metadata = f'{CFG.input_root}data/metadata.csv'
    df_audio = pd.read_csv(f_metadata)
    df_audio['ext'] = df_audio.path.apply(lambda x: x.split('.')[-1])
    df_audio[['sr', 'err', 'audio']] = 0
    df_audio[['len', 'max', 'min', 'mean', 'median', 'abs_mean', 'abs_median', 'std_rms', 'shape', 'specshape']] = 0
    df_audio['title'] = ''
    df_audio['species'] = df_audio.path.apply(lambda x: x.split('.')[-2].split(' - ')[-1])
    df_audio = df_audio[df_audio['ext'].isin(['wav', 'mp3', 'ogg'])]
    print(df_audio)
    for i, r in tqdm(df_audio.iterrows()):
        try:
            audio, sr = lbr.load(CFG.input_root + r['path'], sr=None)
            if sr != CFG.target_sr:
                audio = lbr.resample(audio, orig_sr=sr, target_sr=CFG.target_sr)
            a = f'len {len(audio)}, max {np.max(audio)}, min {np.min(audio)}, avg {np.mean(audio)}, median {np.median(audio)}, std {np.std(audio)}'

            spec = lbr.feature.melspectrogram(
                hop_length=CFG.hop_length, y=audio, sr=sr, n_mels=CFG.n_mels, n_fft=CFG.n_fft,
                fmax=CFG.fmax, fmin=CFG.fmin)
            spec = lbr.power_to_db(spec, ref=1.0)
            print(i, r['path'], f'melspec shape {spec.shape}, {a}')

            # df_audio.at[i, ['audio']] = a
            df_audio.at[i, ['sr']] = sr
            df_audio.at[i, ['len']] = len(audio)
            df_audio.at[i, ['shape']] = np.shape(audio)
            df_audio.at[i, ['specshape']] = f'{spec.shape}'
            df_audio.at[i, ['max']] = np.max(audio)
            df_audio.at[i, ['min']] = np.min(audio)
            df_audio.at[i, ['mean']] = np.mean(audio)
            df_audio.at[i, ['median']] = np.median(audio)
            df_audio.at[i, ['abs_mean']] = np.mean(np.abs(audio))
            df_audio.at[i, ['abs_median']] = np.median(np.abs(audio))
            df_audio.at[i, ['std_rms']] = np.std(audio.flatten())
            df_audio.at[i, ['title']] = r['path'].split('/')[-1]
            # df_meta.at[i, ['audio']] = audio
            if CFG.output_melspec:
                fig = plt.figure(figsize=(12, 6))
                lid.specshow(
                    spec, sr=sr, n_fft=2028, fmin=CFG.fmin, fmax=CFG.fmax,
                    x_axis='time', y_axis='mel', cmap='coolwarm')
                plt.title(r['path'].split('/')[-1])
                plt.show()
        except Exception as e:
            print(colored(f'Error on file: {r.path}, error: {e}', 'red'))
            df_audio.loc[i, ['err']] = 1

    print('summary: potential errors:\n', df_audio[df_audio['err'] != 0])
    print('deviating sample rates:\n', df_audio[df_audio['sr'] != df_audio['sample_rate']])
    print('head 100:\n', df_audio.head(100))
    print('sampling rates:\n', pd.value_counts(df_audio.sr))
    # print('species:\n', pd.value_counts(df_audio.species))
    df_audio.to_csv(f'{CFG.input_root}data/real_metadata.csv')

if __name__ == '__main__':
    main()

