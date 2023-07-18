# experimenting with extracting signal from noise
import numpy as np
# import pandas as pd
from PIL import Image
from scipy.signal import medfilt2d

kernel_size = 7  # for mask that is used to filter out noise from spectrogram
snr_quantile = .25  # determine base value for determining signal vs. noise
snr_threshold = 1.5  # factor of base value, above that is considered signal

img = Image.open("../data/specs/729_2.png")
spec = 1 - np.asarray(img).astype(np.float32) / 255.0
print(spec)
print(spec.shape)

mask = medfilt2d(spec, kernel_size=kernel_size)
thres = np.quantile(mask, snr_quantile, axis=None)
thres_0 = np.quantile(mask, snr_quantile, axis=0)
thres_1 = np.quantile(mask, snr_quantile, axis=1)
med = (np.zeros(spec.shape) + np.expand_dims(thres_0, 0) + np.expand_dims(thres_1, 1) + thres) / 3 * snr_threshold
print(med)
print(med.shape)

spec -= med
spec[spec < 0.0] = 0.0
spec /= np.max(spec)

img = Image.fromarray(np.uint8(255.9999 - spec * 255.9999))
img.show()

