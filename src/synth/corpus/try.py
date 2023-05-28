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


w1, sr1 = librosa.load(r'C:\Users\mrshu\reps\music-style-performer\sounds\converted\31_0.wav', mono=False, sr=None)
w2, sr2 = librosa.load(r'C:\Users\mrshu\reps\music-style-performer\sounds\converted\31_1.wav', mono=False, sr=None)

w1 = process_wave(w1, sr1)
w2 = process_wave(w2, sr2)

plt.plot(w1[0], alpha=0.5, label='22')
plt.plot(w2[0], alpha=0.5, label='70')
plt.legend()
plt.show()

vol1 = get_volume(w1, sr1)
vol2 = get_volume(w2, sr2)
print(vol1, vol2)

vol_mean = (vol1 + vol2) / 2
w1 *= vol_mean / vol1
w2 *= vol_mean / vol2
print(get_volume(w1, sr1), get_volume(w2, sr2))


# wavfile.write(r'C:\Users\mrshu\reps\music-style-performer\sounds\processed\22_2.wav', sr1, np.swapaxes(w1, 0, 1))
# wavfile.write(r'C:\Users\mrshu\reps\music-style-performer\sounds\processed\70_2.wav', sr2, np.swapaxes(w2, 0, 1))


