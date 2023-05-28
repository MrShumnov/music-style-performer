import sqlite3
import librosa
import numpy as np

class DBSamples:
    def __init__(self, path) -> None:
        self.sample_rate = 44100
        self.path = path
        self.conn = sqlite3.connect(path + '/sounds.db')

        self.cache = {}

    
    def get_samples_descriptors(self):
        c = self.conn.cursor()
        q = '''
            select * from notes;
        '''

        c.execute(q)
        return c.fetchall()
    

    def get_sample(self, name: str):
        if name in self.cache:
            return self.cache[name].copy()

        w, sr = librosa.load(self.path + '/' + name, mono=False, sr=None)
        # w = np.load(self.path + '/' + name)
        self.cache[name] = w.copy()

        return w
    

    def clear_cache(self):
        self.cache = {}
