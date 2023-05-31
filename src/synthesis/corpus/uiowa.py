import librosa
from scipy.io import wavfile
import numpy as np
import os
import sqlite3
from scipy import signal


sample_rate = 44100


def download(dir):
    from bs4 import BeautifulSoup
    import requests

    url = 'https://theremin.music.uiowa.edu/'
    notes = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
    vels = {'pp' : 0, 'mf' : 1, 'ff': 2}

    page = requests.get(url + 'MISpiano.html')
    soup = BeautifulSoup(page.text, "html.parser")

    print('start')
    flag = False

    for a in soup.findAll('a', href=True):
        s = a['href'].split('.')
        if len(s) > 3:
            if a['href'] == 'sound files/MIS/Piano_Other/piano/Piano.mf.E2.aiff':
                flag = True
            if flag:
                tone = s[-2]
                vel = s[-3]

                octave = int(tone[-1]) - 1
                note = tone[:-1]

                tone = 24 + octave * 12 + notes[note] 
                vel = vels[vel]

                response = requests.get(url + a['href'])
                print(a['href'] + ' downloaded')

                open(dir + f'/{tone}_{vel}.aiff', "wb").write(response.content)

import pyloudnorm as pyln
meter = pyln.Meter(sample_rate)


def get_volume(w, sr):
    return meter.integrated_loudness(np.swapaxes(w[:, :sr//2], 0, 1)) + np.sqrt(np.mean(w[:, :sr // 2] ** 2)) * 1000


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def rmse(arr, axis=None):
    return np.sqrt(np.mean(arr ** 2, axis=axis))

fade_in_size = int(sample_rate * 0.006)
fade_curve = np.repeat((np.linspace(0.0, 1.0, fade_in_size) ** (1))[np.newaxis, :], 2, axis=0)

def trim_start(w, sr):
    n = 64

    start = 0
    end = (4 * sr) // n * n
    step = (end - start) // n

    while (step > 8):
        splitted = np.split(w[:, start:end], n, axis=1)
        s = np.stack(splitted, axis=0)
        vol = rmse(s, (1, 2))

        vol_diff = np.diff(vol)
        
        index = np.argmax(vol_diff)
        start = start + index * step
        end = start + 2 * step//n*n

        if n > 4:
            n //= 2

        step = (end - start) // n

    start -= fade_in_size

    return start    


def process_wave(w, sr):
    w[0] = butter_highpass_filter(w[0], 10, sr)
    w[1] = butter_highpass_filter(w[1], 10, sr)

    # w[0] -= np.mean(w[0])
    # w[1] -= np.mean(w[1])
    
    # _, index = librosa.effects.trim(w, top_db=40, frame_length=128, hop_length=128)
    # index[1] = (w.shape[1] + index[1] * 2) / 3
    w = w[:, trim_start(w, sr):]
    w[:, :fade_in_size] *= fade_curve

    return w


def process_corpus(dir, outdir, minvel, maxvel):
    fnames = os.listdir(dir)

    waves = [{}, {}, {}]
    vols = [{}, {}, {}]

    vol_min = [1e5 for i in range(1)]
    vol_max = [-1e5 for i in range(1)]

    for i, fname in enumerate(fnames):
        w, sr = librosa.load(dir + '/' + fname, mono=False, sr=None)
        w = process_wave(w, sr)

        v = int(fname.split('_')[-1][0])
        t = int(fname.split('_')[-2])
        waves[v][t] = [w, sr]
        vols[v][t] = get_volume(w, sr)

        group = 0 # (t - 12) // 12
        if v == 0 and vols[v][t] < vol_min[group]:
            vol_min[group] = vols[v][t]
        elif v == 2 and vols[v][t] > vol_max[group]:
            vol_max[group] = vols[v][t]

        if i % 10 == 9:
            print(f'{i + 1}/{len(fnames)}')

    result = []

    for i in range(3):
        for t in waves[i]:
            group = 0 # (t - 12) // 12

            vol = vols[i][t]
            vel = (vol - vol_min[group]) / (vol_max[group] - vol_min[group]) * (maxvel - minvel) + minvel
            wave = waves[i][t][0]
            sr = waves[i][t][1]

            wave = (wave * 32767).astype(np.int16)
 
            wavfile.write(outdir + f'/{t}_{i}.wav', sr, np.swapaxes(wave, 0, 1))
            # np.save(outdir + f'/{t}_{i}.npy', wave) #.astype(np.float16))
            result.append((f'{t}_{i}.wav', t, int(vel), float(vol)))

    return result

    
dir = r'C:\Users\mrshu\reps\music-style-performer\sounds\converted'
dbdir = r'C:\Users\mrshu\reps\music-style-performer\sounds\corpus_wav_lufs'


def execute_sql(conn, query):
    c = conn.cursor()
    c.execute(query)


def insert_rows(conn, query, data):
    c = conn.cursor()
    c.executemany(query, data)


def main():
    data = process_corpus(dir, dbdir, 0, 127)

    conn = sqlite3.connect(dbdir + '/sounds.db')
    
    sql_create_table = '''
        CREATE TABLE IF NOT EXISTS notes (
            id integer PRIMARY KEY,
            path text NOT NULL UNIQUE,
            tone integer,
            velocity integer,
            volume integer
        );
    '''

    execute_sql(conn, sql_create_table)

    sql_insert_notes = '''
        INSERT INTO notes (path, tone, velocity, volume) VALUES (?,?,?,?);
    '''
    insert_rows(conn, sql_insert_notes, data)

    conn.commit()


if __name__ == '__main__':
    main()

    