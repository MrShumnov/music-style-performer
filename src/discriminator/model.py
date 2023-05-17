import os
import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class MLPAutoencoder(keras.Model):
    def __init__(self, 
            input_size,
            latent_dim,
            noise, 
            encoder_layers,
            encoder_dropout, 
            decoder_layers,
            decoder_dropout):
        
        super().__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.noise = noise   
        self.encoder_layers = encoder_layers  
        self.encoder_dropout = encoder_dropout
        self.decoder_layers = decoder_layers  
        self.decoder_dropout = decoder_dropout

        self.build_encoder()
        self.build_decoder()


    def build_encoder(self):
        self.encoder = keras.Sequential()

        self.encoder.add(keras.layers.Input(shape=(self.input_size)))
        self.encoder.add(keras.layers.GaussianNoise(self.noise))
        
        for neurons, dropout in zip(self.encoder_layers, self.encoder_dropout):
            if dropout > 0:
                self.encoder.add(keras.layers.Dropout(dropout))
            self.encoder.add(keras.layers.Dense(neurons, activation='relu'))
        
        self.encoder.add(keras.layers.Dense(self.latent_dim, activation='tanh'))
        
        
    def build_decoder(self):
        self.decoder = keras.Sequential()

        self.decoder.add(keras.layers.Input(shape=(self.latent_dim)))
        
        for neurons, dropout in zip(self.decoder_layers, self.decoder_dropout):
            if dropout > 0:
                self.decoder.add(keras.layers.Dropout(dropout))
            self.decoder.add(keras.layers.Dense(neurons, activation='relu'))

        self.decoder.add(keras.layers.Dense(self.input_size, activation='linear'))

    
    def call(self, x, training=False):
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
    

    def to_string(self):
        stringlist = []
        self.encoder.summary(print_fn=lambda x: stringlist.append(x))
        encoder_summary = "\n".join(stringlist)
        
        stringlist = []
        self.decoder.summary(print_fn=lambda x: stringlist.append(x))
        decoder_summary = "\n".join(stringlist)
        
        return f'latent dim: {self.latent_dim} \nnoise: {self.noise} \n\nencoder: {encoder_summary} \n\ndecoder: {decoder_summary}\n'
    


class WeightedMSE(tf.keras.losses.Loss):
    def __init__(self, vel_mask, leg_mask, first_last_mask, vel_weight, leg_weight, first_last_weight):
        super().__init__()
        
        self.vel_weight = vel_weight
        self.leg_weight = leg_weight

        self.weights = 1 + tf.convert_to_tensor(vel_mask, dtype=tf.float32) * (vel_weight - 1) \
                        + tf.convert_to_tensor(leg_mask, dtype=tf.float32) * (leg_weight - 1) 
        self.weights *= tf.convert_to_tensor(first_last_mask, dtype=tf.float32) * (first_last_weight - 1) + 1
                        
        self.sum_weights = tf.cast(tf.reduce_sum(self.weights), dtype=tf.float32)
    

    def call(self, y_true, y_pred):
        return tf.reduce_sum(self.weights * tf.math.square(y_pred - y_true), axis=1) / self.sum_weights
    

    def to_string(self):
        return f'velocity weight: {self.vel_weight} \nlegato weight: {self.leg_weight}'



class myCallback(tf.keras.callbacks.Callback): 
    def __init__(self, occ_model, patience):
        super().__init__()
        
        self.patience = patience
        self.occ_model = occ_model

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_epoch = 0
        self.best = (np.Inf, np.Inf)

    def on_epoch_end(self, epoch, logs=None): 
        occ_accuracy = self.occ_model.evaluate(epoch)
        val_loss = logs['val_loss']
        logs['occ_accuracy'] = occ_accuracy

        if occ_accuracy > self.best[0] or val_loss < self.best[1]:
            self.best = (occ_accuracy, val_loss)
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f'Last epoch with improvement: {self.best_epoch}')
        
        self.occ_model.save()



class OCCModel():
    def __init__(self, autoencoder, dataprocessor):
        self.autoencoder = autoencoder
        self.dataprocessor = dataprocessor
            

    def compile(self, modelsdir, name, optimizer, loss, ckpt_epochs=1):
        self.modelsdir = modelsdir
        self.name = name
        self.optimizer = optimizer
        self.loss = loss
        self.ckpt_epochs = ckpt_epochs
        self.epochs = 0

        self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)

        self.dir = self.modelsdir + '/' + self.name
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        self.checkpoint_dir = self.dir + '/checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(autoencoder_optimizer=self.optimizer,
                                            autoencoder=self.autoencoder)
        
        self.log_dir = self.dir + '/logs/'
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        self.tensorboard_callbacks = [
            myCallback(self),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, update_freq='batch')
        ]


    def summary(self):
        with open(self.dir + '/desc.txt', 'w') as f:
            f.write('______ DATA _______\n' + self.dataprocessor.to_string() + 
                    '\n\n\n______ MODEL ______\n' + self.autoencoder.to_string() + 
                    '\n\n\n______ LOSS _______\n' + self.loss.to_string() + 
                    '\n\n\n____ OPTIMIZER ____\n' + f'adam \nlearning rate: {self.optimizer.learning_rate}')


    def load(self, path):
        tf.train.Checkpoint(autoencoder=self.autoencoder).restore(path)


    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)


    def restore(self, idx):
        self.epochs = idx * self.ckpt_epochs
        self.checkpoint.restore(self.checkpoint_dir + '/ckpt-' + str(idx))


    @tf.function
    def predict(self, x, weights):
        y = self.autoencoder(x, training=False)
    
        err = tf.reduce_sum(weights * tf.math.square(x - y), axis=1) / tf.reduce_sum(weights)
        # err = tf.math.sqrt(err)
        
        return err
    

    @tf.function
    def discriminate(self, line):
        data = np.squeeze(sliding_window_view(line, (self.dataprocessor.notes_qty, 4)))
        data, fnotes = self.dataprocessor.make_relative(data)
        data, fnotes = self.dataprocessor.normalize(data, fnotes)
        data = self.dataprocessor.reshape(data, fnotes)

        predicted = self.predict(data)

        return tf.reduce_mean(predicted)


    def fit(self, dataset, epochs):
        self.autoencoder.fit(x=dataset.train, y=dataset.train,
                        batch_size=dataset.batch_size, epochs=self.epochs+epochs, 
                        initial_epoch=self.epochs,
                        shuffle=True,
                        validation_data=(dataset.test, dataset.test),
                        callbacks=self.tensorboard_callbacks, 
                        verbose=2)


    def evaluate(self, dataset, epoch):
        vel_mask = tf.convert_to_tensor(dataset.vel_mask, dtype=tf.float32)
        leg_mask = tf.convert_to_tensor(dataset.leg_mask, dtype=tf.float32)

        vel_true_predict = self.predict(dataset.test, vel_mask).numpy()
        leg_true_predict = self.predict(dataset.test, leg_mask).numpy()
        vel_mess_predict = self.predict(dataset.vel_mess, vel_mask).numpy()
        leg_mess_predict = self.predict(dataset.leg_mess, leg_mask).numpy()

        vel_mean, vel_var, leg_mean, leg_var = self.dataprocessor.validate(vel_true_predict, leg_true_predict,
            vel_mess_predict,
            leg_mess_predict,
            self.dir + f'/validation_{epoch}.png')
        
        with open(self.dir + f'/validation_{epoch}.txt', 'w') as f:
            f.write(f'vel_mean: \n{vel_mean} \n\nvel_var: \n{vel_var} \n\nleg_mean: \n{leg_mean} \n\nleg_var: \n{leg_var}')

        return np.mean(vel_mean) * self.loss.vel_weight + \
                np.mean(leg_mean) * self.loss.leg_weight
        