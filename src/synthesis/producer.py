import numpy as np
from synthesis.sound import PianoSound
import librosa
from synthesis.dbsamples import DBSamples
from scipy.io import wavfile


class SoundProducer:
    def __init__(self, dbsamples: DBSamples, sample_rate=44100):
        self.dbsamples = dbsamples
        self.sample_rate = sample_rate


    def synthesis(self, sounds):
        pass



class PianoSoundProducer(SoundProducer):
    def __init__(self, dbsamples: DBSamples):
        super().__init__(dbsamples)
        self.sample_rate = dbsamples.sample_rate

        self.samples = dbsamples.get_samples_descriptors()
        
        self.id = np.array(list(map(lambda x: x[0], self.samples)), dtype=np.int)
        self.name = np.array(list(map(lambda x: x[1], self.samples)), dtype=np.str)
        self.tone = np.array(list(map(lambda x: x[2], self.samples)), dtype=np.int)
        self.velocity = np.array(list(map(lambda x: x[3], self.samples)), dtype=np.int)
        self.volume = np.array(list(map(lambda x: x[4], self.samples)), dtype=np.float)

        self.min_vel = np.min(self.velocity)
        self.max_vel = np.max(self.velocity)

        self.fade_curve = np.repeat((np.linspace(1.0, 0.0, int(self.sample_rate * 0.4)) ** 3)[np.newaxis, :], 2, axis=0)


    def get_best_sample(self, descr: PianoSound, duration: int, i):
        loss = (np.abs(self.tone - descr.note) + 1) ** 2 * 5 + np.abs(self.velocity - descr.velocity)

        index = np.argmin(loss)
        tone = self.tone[index]
        vel = self.velocity[index]
        sample_w = self.dbsamples.get_sample(self.name[index])
        # wavfile.write(rf'C:\Users\mrshu\reps\music-style-performer\src\synth\test\{i}_before.wav', self.dbsamples.sample_rate, np.swapaxes(sample_w, 0, 1))


        if tone != descr.note:
            sample_w = librosa.effects.pitch_shift(sample_w, sr=self.dbsamples.sample_rate, n_steps=descr.note-tone) 
        if vel != descr.velocity:
            sample_w *= descr.velocity / vel
        
        if sample_w.shape[1] < duration:
            sample_w[:, -self.fade_curve.shape[1]:] *= self.fade_curve
            sample_w = np.pad(sample_w, ((0, 0), (0, duration - sample_w.shape[1])))
        else:
            sample_w = sample_w[:, :duration]
            sample_w[:, -self.fade_curve.shape[1]:] *= self.fade_curve

        # wavfile.write(rf'C:\Users\mrshu\reps\music-style-performer\src\synth\test\{i}_after.wav', self.dbsamples.sample_rate, np.swapaxes(sample_w, 0, 1))

        return sample_w  


    def synthesis(self, shedule, duration):
        result = np.zeros((2, int((duration + 1) * self.sample_rate)))

        for i, descr in enumerate(shedule):
            start = int(descr.start * self.sample_rate)
            end = int(descr.end * self.sample_rate) + self.fade_curve.shape[1]

            sample = self.get_best_sample(descr, end - start, i)
            result[:, start:end ] += sample

            if i % 100 == 99:
                print(f'{i}/{len(shedule)}')

        result = result / np.max(np.abs(result))

        return result
