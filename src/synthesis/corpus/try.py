from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import librosa

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def process_wave(w, sr):
    w[0] = butter_highpass_filter(w[0], 10, sr)
    w[1] = butter_highpass_filter(w[1], 10, sr)

    #w[0] -= np.mean(w[0])
    #w[1] -= np.mean(w[1])
    
    _, index = librosa.effects.trim(w, top_db=30, frame_length=512, hop_length=128)
    index[1] = (w.shape[1] + index[1] * 2) / 3
    w = w[:, index[0]:index[1]]

    return w

def get_volume(w, sr):
    return np.sqrt(np.mean(w[:, :sr//2] ** 2))


def rmse(arr, axis=None):
    return np.sqrt(np.mean(arr ** 2, axis=axis))

def mae(arr, axis=None):
    return np.mean(np.abs(arr), axis=axis)


def trim_start(w, sr):
    n = 64

    start = 0
    end = (4 * sr) // n * n
    step = (end - start) // n

    while (step > 64):
        splitted = np.split(w[:, start:end], n, axis=1)
        s = np.stack(splitted, axis=0)
        vol = rmse(s, (1, 2))

        vol_diff = np.diff(vol)
        
        index = np.argmax(vol_diff)
        start = start + index * step
        end = start + 2 * step//n*n
        
        print(f'step: {step}, section: {index}-{index+1}')

        if n > 4:
            n //= 2

        step = (end - start) // n

    start -= int(sr * 0.01)

    return start


def trim_end(w, sr, threshold=1e-3):
    n = 100
    start = 5 * sr // n * n
    end = w.shape[1] // n * n
    step = (end - start) // n

    splitted = np.split(w[:, start:end], n, axis=1)
    s = np.stack(splitted, axis=0)
    vol = rmse(s, (1, 2))
    print(vol)
    
    index = np.argmax(vol < threshold)

    print(start, end, step, index)

    return start + index * step


w1, sr1 = librosa.load(r'C:\Users\mrshu\reps\music-style-performer\sounds\converted\70_2.wav', mono=False, sr=None)

w1 = w1[:, trim_start(w1, sr1):]

#w1 = process_wave(w1, sr1)

plt.plot(w1[0], alpha=0.5)
plt.axvline(x=trim_start(w1, sr1))
plt.axvline(x=trim_end(w1, sr1))
plt.show()

w1 = (w1 * 32767).astype(np.int16)


wavfile.write(r'C:\Users\mrshu\reps\music-style-performer\test\synth\test2.wav', sr1, np.swapaxes(w1, 0, 1))

