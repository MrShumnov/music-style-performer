import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf
import mido


MAX_NOTE_INT = 24
MAX_NOTE_DT = 5
        

class Dataset:
    def __init__(self, dataprocessor, batch_size):
        self.dataprocessor = dataprocessor
        self.batch_size = batch_size


    def load_dset(self, train_file, test_len=1000, test_file=None):
        self.test_len = test_len

        dset = np.load(train_file)
        
        np.random.seed(1)

        if test_file:
            testdset = np.load(test_file)
        else:
            testdset = dset[-test_len*self.dataprocessor.notes_qty:]
            dset = dset[:-test_len*self.dataprocessor.notes_qty]
            
        self.train = self.dataprocessor.process_dset(dset, True)
        self.test = self.dataprocessor.process_dset(testdset)
        
        self.test_len = int(len(self.test) / self.dataprocessor.notes_qty)
        np.random.shuffle(self.test)
        self.test = self.test[:test_len * self.dataprocessor.notes_qty]

        self.vel_mess = self.dataprocessor.mess(self.test, self.dataprocessor.vel_mask, test_len)
        self.leg_mess = self.dataprocessor.mess(self.test, self.dataprocessor.leg_mask, test_len)
    


class DataProcessor:
    def __init__(self, notes_qty):
        self.notes_qty = notes_qty
        self.normparams = None
    

    def saveparams(self, filename):
        with open(filename, 'w') as f:
            f.write(' '.join(list(map(str, self.normparams))))


    def loadparams(self, filename):
        with open(filename, 'r') as f:
            self.normparams = list(map(float, f.read().split()))


    def process_dset(self, dset, calcparams=False):
        pass


    def process_test(self, dset):
        pass


    def reshape_test(self, dset):
        pass


    def to_string(self):
        pass


    def mess(self, test, mask, test_len):
        pass


    def validate(self, predicted, test_len, figpath=None):
        pass



class DataProcessorV0(DataProcessor):
    def __init__(self, 
            notes_qty: int,
            include_first_tone: bool, 
            absolute_velocities: bool):
        
        super().__init__(notes_qty)
        
        self.include_first_tone = include_first_tone 
        self.absolute_velocities = absolute_velocities 

        self.fnote_shift = 0 if include_first_tone else 1

        self.input_size = self.notes_qty * 4
        self.dist_mask = np.array([1 if i%4==1 else 0 for i in range(self.input_size)])
        self.vel_mask = np.array([1 if i%4==2 else 0 for i in range(self.input_size)])
        self.leg_mask = np.array([1 if i%4==3 else 0 for i in range(self.input_size)])
        self.weight_mask = self.vel_mask + self.leg_mask + self.dist_mask
        self.first_last_mask = np.array([1] * 4 + [0] * (self.notes_qty * 4 - 8) + [1] * 4)

        if not include_first_tone:
            self.input_size -= 1
            self.weight_mask = self.weight_mask[1:]
            self.dist_mask = self.vel_mask[1:]
            self.vel_mask = self.vel_mask[1:]
            self.leg_mask = self.leg_mask[1:]
            self.first_last_mask = self.first_last_mask[1:]


    def to_string(self):
        return f'notes qty: {self.notes_qty} \ninclude first tone: {self.include_first_tone} \nabsolute velocities: {self.absolute_velocities}'


    def process_dset(self, dset, calcparams=False):
        dset = self.get_features(dset)
        dset, fnotes = self.make_relative(dset)
        dset, fnotes = self.filter_outliers(dset, fnotes)
        dset, fnotes = self.normalize(dset, fnotes, calcparams)
        dset = self.reshape(dset, fnotes)

        return dset
    

    def process_test(self, mid, tracknum, stride):
        line = track2line(mid.tracks[tracknum], mid.ticks_per_beat)
        line = np.array(line)
        
        tones = line[..., 0].copy()
        dists = line[..., 4].copy() / self.normparams[2]
        line = np.delete(line, (1), axis=1)

        line = self.get_features(line)
        # line = np.squeeze(sliding_window_view(line, (self.notes_qty, 4))).copy()
        line = window_np(line, self.notes_qty, stride)
        dset, fnotes = self.make_relative(line)
        dset, fnotes = self.normalize(dset, fnotes, False)

        fnotes = np.concatenate([np.zeros(shape=(fnotes.shape[0], 1)), fnotes], axis=-1)

        return np.concatenate([np.expand_dims(fnotes, axis=1), dset], axis=1), tones, dists
    

    def reshape_test(self, line):
        line = np.reshape(line, [line.shape[0], line.shape[1] * line.shape[2]])
        if not self.include_first_tone:
            line = line[:, 1:]
            
        return line


    def get_features(self, dset):    
        dt = dset[..., 3].copy()
        dset[..., 3] = dset[..., 1] / dset[..., 3]
        dset[..., 1] = dt
        
        # 0 - tone
        # 1 - dt
        # 2 - velocity
        # 3 - legato
        
        return dset
    
    
    def make_relative(self, dset):
        dset[..., 0] = np.diff(dset[..., 0], axis=1, prepend=0)
        if not self.absolute_velocities:
            dset[..., 2] = np.diff(dset[..., 2], axis=1, prepend=0)
        
        dset = np.delete(dset, obj=0, axis=1)
        if not self.include_first_tone:
            first_notes = np.delete(dset[:, 0, :], obj=0, axis=-1)
        
        return dset, first_notes
    

    def filter_outliers(self, dset, fnotes):
        cond = np.logical_and.reduce((
            np.all(np.abs(dset[..., 0]) <= MAX_NOTE_INT, axis=1),
            np.all(dset[..., 1] > 0, axis=1),
            np.all(dset[..., 1] <= MAX_NOTE_DT, axis=1),
            np.all(dset[..., 3] > 0, axis=1),
            
            fnotes[..., 1 - self.fnote_shift] <= MAX_NOTE_DT,
            fnotes[..., 3 - self.fnote_shift] > 0
        ))
        
        dset = dset[cond]
        fnotes = fnotes[cond]
        
        return dset, fnotes


    def normalize(self, dset, fnotes, calculate_params=False):    
        if not calculate_params:
            tone_mean, tone_std, dt_max, vel_mean, vel_std, fvel_mean, fvel_std = self.normparams
        else:
            tone_mean, tone_std = norm.fit(dset[..., 0].flatten())
            dt_max = np.max(np.concatenate((dset[..., 1], fnotes[..., 1 - self.fnote_shift]), axis=None))
            vel_mean, vel_std = norm.fit(dset[..., 2].flatten()) 
            if self.absolute_velocities:
                fvel_mean, fvel_std = vel_mean, vel_std
            else:
                fvel_mean, fvel_std = norm.fit(fnotes[..., 2 - self.fnote_shift].flatten())

            self.normparams = (tone_mean, tone_std, dt_max, vel_mean, vel_std, fvel_mean, fvel_std)
        
        dset[..., 0] = (dset[..., 0] - tone_mean) / tone_std
        dset[..., 1] = dset[..., 1] / dt_max
        dset[..., 2] = (dset[..., 2] - vel_mean) / vel_std
        dset[..., 3] = dset[..., 3] - 1
        
        fnotes[..., 1 - self.fnote_shift] = fnotes[..., 1 - self.fnote_shift] / dt_max
        fnotes[..., 2 - self.fnote_shift] = (fnotes[..., 2 - self.fnote_shift] - fvel_mean) / fvel_std
        fnotes[..., 3 - self.fnote_shift] = fnotes[..., 3 - self.fnote_shift] - 1
        
        return dset, fnotes
        
    
    def reshape(self, dset, fnotes):
        dset = dset[:, : self.notes_qty - 1, :]
        dset = np.reshape(dset, (dset.shape[0], dset.shape[1] * dset.shape[2]))
        dset = np.concatenate((fnotes, dset), axis=1)

        return dset
        

    def mess_n(self, lines, n, mask):
        mu, std = norm.fit(lines[..., mask].flatten())
        
        for line in lines:
            idxs = np.random.choice(self.notes_qty, n, False)
            line[mask[idxs]] = np.random.normal(mu, std, (n))


    def mess(self, test, mask, TEST_LEN):
        test_messed = test.copy()[:self.notes_qty * TEST_LEN]
        mask = np.asarray(mask>0).nonzero()[0]
        for i in range(self.notes_qty):
            self.mess_n(test_messed[i * TEST_LEN: (i + 1) * TEST_LEN], i+1, mask)
            
        return test_messed
    

    def validate(self, predicted, TEST_LEN, figpath=None):
        vel_true_predict, leg_true_predict, vel_mess_predict, leg_mess_predict = predicted

        vel_groups = [(vel_true_predict[i:i+TEST_LEN], vel_mess_predict[i:i+TEST_LEN]) for i in range(0, len(vel_true_predict), TEST_LEN)]
        leg_groups = [(leg_true_predict[i:i+TEST_LEN], leg_mess_predict[i:i+TEST_LEN]) for i in range(0, len(leg_true_predict), TEST_LEN)]

        x = list(range(1, self.notes_qty + 1))
        vel_y_mean = []
        vel_y_var = []
        leg_y_mean = []
        leg_y_var = []
        
        vel_diffs = []
        leg_diffs = []

        for i in range(self.notes_qty):
            vel_diff = vel_groups[i][1] - vel_groups[i][0]
            vel_diffs.append(vel_diff)
            
            leg_diff = leg_groups[i][1] - leg_groups[i][0]
            leg_diffs.append(leg_diff)

            vel_m = np.mean(vel_diff)
            vel_var = np.std(vel_diff) / vel_m
            vel_y_mean.append(vel_m)
            vel_y_var.append(vel_var)
            
            leg_m = np.mean(leg_diff)
            leg_var = np.std(leg_diff) / leg_m
            leg_y_mean.append(leg_m)
            leg_y_var.append(leg_var)
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        
        axs[0][0].title.set_text('Diff - Average')
        axs[0][0].plot(x, vel_y_mean, 'o-', label='velocity')
        axs[0][0].plot(x, leg_y_mean, 'o-', label='legato')
        axs[0][0].legend()
        
        axs[0][1].title.set_text('Diff - Coefficient of variation')
        axs[0][1].plot(x, vel_y_var, 'o-', label='velocity')
        axs[0][1].plot(x, leg_y_var, 'o-', label='legato')
        axs[0][1].legend()

        idxs = [(i, j) for i in range(1, 3) for j in range(0, 2)]
        for i in range(4):
            n = int((self.notes_qty - 1) * i / 3)

            axs[idxs[i][0]][idxs[i][1]].title.set_text(f'Diff {n + 1}')
            axs[idxs[i][0]][idxs[i][1]].hist(vel_diffs[n], bins=100, label='velocity', alpha=0.5)
            axs[idxs[i][0]][idxs[i][1]].hist(leg_diffs[n], bins=100, label='legato', alpha=0.5)
            axs[idxs[i][0]][idxs[i][1]].legend()
        
        plt.subplots_adjust(hspace=0.3)
        if figpath is not None:
            plt.savefig(figpath)

        return vel_y_mean, vel_y_var, leg_y_mean, leg_y_var
    

def track2line(track, ticks_per_beat):
    line = []

    time = 0
    for m in track:
        time += m.time / ticks_per_beat / 2

        if m.type == 'note_on' and m.velocity > 0:
            if len(line) > 0:
                line[-1][4] = time - line[-1][1]
            line.append([m.note, time, -1, m.velocity, -1])
            
        elif m.type == 'note_off' or (m.type == 'note_on' and m.velocity == 0):
            i = 1
            while line[-i][0] != m.note:
                i += 1
            
            line[-i][2] = time - line[-i][1]
    
    if len(line) > 0:
        line[-1][4] = line[-1][2]
            
    return line


def rec2mid(rec_dist, rec_vel, rec_leg, base_tones, ticks_per_beat, filename=None):
    events = []
    time = 0

    rec_dist_np = rec_dist.numpy()
    rec_vel_np = rec_vel.numpy()
    rec_leg_np = rec_leg.numpy()

    for i in range(len(rec_dist)):
        diff = int(rec_dist[i] * ticks_per_beat * 2)

        if rec_leg[i] < 0:
            leg = (0.05 * ticks_per_beat * 2) / diff
        else:
            leg = rec_leg[i]
        
        events.append([time, int(base_tones[i]), min(127, max(1, int(rec_vel[i]))), 'note_on'])
        events.append([time + int(diff * leg), int(base_tones[i]), 0, 'note_off'])
        time += diff
        
    events.sort(key=lambda e: e[0])
        
    mid = mido.MidiFile(type=0)
    mid.ticks_per_beat = ticks_per_beat
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.Message(events[0][3], note=events[0][1], velocity=events[0][2], time=events[0][0] if events[0][0] > 0 else 0))
    prev = 0
    cur = [False] * 127
    ignore = [False] * 127
    cur[events[0][1]] = True

    for e in events[1:]:
        diff = e[0] - prev

        if e[3] == 'note_on':
            if cur[e[1]]:
                track.append(mido.Message('note_off', note=e[1], velocity=0, time=diff))
                diff = 0
                cur[e[1]] = False
                ignore[e[1]] = True

            track.append(mido.Message('note_on', note=e[1], velocity=e[2], time=diff))
            prev = e[0]
            cur[e[1]] = True
        else:
            if ignore[e[1]]:
                ignore[e[1]] = False
            else:
                track.append(mido.Message(e[3], note=e[1], velocity=e[2], time=diff))
                prev = e[0]
                cur[e[1]] = False

    if filename is not None:
        mid.save(filename)

    return mid


def window_np(line, window_size, stride):
    length = line.shape[0]
    windows = []

    for i in range(0, length - window_size + 1, stride):
        windows.append(line[i: i + window_size])
    if (length - window_size) % stride != 0:
        windows.append(line[-window_size:length])

    return np.stack(windows, axis=0)

    
@tf.function()
def window_tf(line, window_size, stride):
    length = line.shape[0]
    windows = []

    for i in range(0, length - window_size + 1, stride):
        windows.append(line[i: i + window_size])
    if (length - window_size) % stride != 0:
        windows.append(line[-window_size:length])

    return tf.stack(windows, axis=0)
