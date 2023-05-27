import librosa
from scipy.io import wavfile
import numpy as np
import os
import sqlite3
from scipy import signal


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


def get_volume(w, sr):
    return np.sqrt(np.mean(w[:, :sr//2] ** 2))


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

    # w[0] -= np.mean(w[0])
    # w[1] -= np.mean(w[1])
    
    _, index = librosa.effects.trim(w, top_db=40, frame_length=512, hop_length=128)
    index[1] = (w.shape[1] + index[1] * 2) / 3
    w = w[:, index[0]:index[1]]

    return w


def process_corpus(dir, outdir, minvel, maxvel):
    fnames = os.listdir(dir)

    waves = [{}, {}, {}]
    vols = [{}, {}, {}]

    for i, fname in enumerate(fnames):
        w, sr = librosa.load(dir + '/' + fname, mono=False, sr=None)
        w = process_wave(w, sr)

        v = int(fname.split('_')[-1][0])
        t = int(fname.split('_')[-2])
        waves[v][t] = [w, sr]
        vols[v][t] = get_volume(w, sr)

        if i % 10 == 9:
            print(f'{i + 1}/{len(fnames)}')

    vol_mean = [np.mean(list(vols[0].values())), None, 
                np.mean(list(vols[2].values()))]
    vels = [minvel, None, maxvel]

    result = []

    for i in [0, 2]:
        for t in waves[i]:
            wave = waves[i][t][0] * vol_mean[i] / vols[i][t]
            sr = waves[i][t][1]
            
            wavfile.write(outdir + f'/{t}_{i}.wav', sr, np.swapaxes(wave, 0, 1))
            result.append((f'{t}_{i}.wav', t, vels[i], float(vol_mean[i])))

    for t in waves[1]:
        vol = vols[1][t]
        vel = (vol - vol_mean[0]) / (vol_mean[2] - vol_mean[0]) * (maxvel - minvel) + minvel
        wave = waves[1][t][0]
        sr = waves[1][t][1]

        wavfile.write(outdir + f'/{t}_1.wav', sr, np.swapaxes(wave, 0, 1))
        result.append((f'{t}_1.wav', t, int(vel), float(vol)))

    return result

    
dir = r'C:\Users\mrshu\reps\music-style-performer\sounds\converted'
dbdir = r'C:\Users\mrshu\reps\music-style-performer\sounds'


def execute_sql(conn, query):
    c = conn.cursor()
    c.execute(query)


def insert_rows(conn, query, data):
    c = conn.cursor()
    c.executemany(query, data)


def main():
    data = process_corpus(dir, dbdir, 30, 127)

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

    