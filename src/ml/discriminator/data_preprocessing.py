import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf

BASE_NOTES_QTY = 20
NOTE_PARAMS = 4
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
            
        self.train = self.process_dset(dset, True)
        self.test = self.process_dset(testdset)
        
        self.test_len = int(len(self.test) / self.dataprocessor.notes_qty)
        np.random.shuffle(self.test)
        self.test = self.test[:test_len * self.dataprocessor.notes_qty]

        self.vel_mess = self.dataprocessor.mess(self.test, self.dataprocessor.vel_mask, test_len)
        self.leg_mess = self.dataprocessor.mess(self.test, self.dataprocessor.leg_mask, test_len)


    def process_dset(self, dset, calcparams=False):
        dset = self.dataprocessor.get_features(dset)
        dset, fnotes = self.dataprocessor.make_relative(dset)
        dset, fnotes = self.dataprocessor.filter_outliers(dset, fnotes)
        dset, fnotes = self.dataprocessor.normalize(dset, fnotes, calcparams)
        dset = self.dataprocessor.reshape(dset, fnotes)

        return dset
    


class DataProcessor:
    def __init__(self, 
            notes_qty: int,
            include_first_tone: bool, 
            absolute_velocities: bool):
        
        self.notes_qty = notes_qty 
        self.include_first_tone = include_first_tone 
        self.absolute_velocities = absolute_velocities 

        self.fnote_shift = 0 if include_first_tone else 1

        self.input_size = self.notes_qty * NOTE_PARAMS
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


    def saveparams(self, filename):
        with open(filename, 'w') as f:
            f.write(' '.join(list(map(str, self.normparams))))


    def loadparams(self, filename):
        with open(filename, 'r') as f:
            tone_mean, tone_std, dt_max, vel_mean, vel_std, fvel_mean, fvel_std = list(map(float, f.read().split()))
            self.normparams = (tone_mean, tone_std, dt_max, vel_mean, vel_std, fvel_mean, fvel_std)
            

    def get_features(self, dset):    
        dt = dset[..., 3].copy()
        dset[..., 3] = dset[..., 1] / dset[..., 3]
        dset[..., 1] = dt
        
        # 0 - tone
        # 1 - dt
        # 2 - velocity
        # 3 - legato
        
        return dset
    

    @tf.function
    def preprocess_test(self, base_features, target_features):
        data = tf.concat([base_features, target_features], -1)

        data = np.squeeze(sliding_window_view(data, (self.notes_qty, 4)))

        data, fnotes = self.make_relative(data)
        data, fnotes = self.normalize(data, fnotes, False)

        data = tf.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
        data = tf.concat((fnotes, data), axis=1)

        return data
    
    
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
    

    def validate(self, vel_true_predict, leg_true_predict, vel_mess_predict, leg_mess_predict, TEST_LEN, figpath=None):
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
    