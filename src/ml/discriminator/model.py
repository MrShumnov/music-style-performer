import os
import datetime
import tensorflow as tf
import numpy as np 



class myCallback(tf.keras.callbacks.Callback): 
    def __init__(self, occ_model, dset, patience):
        super().__init__()
        
        self.patience = patience
        self.occ_model = occ_model
        self.dset = dset


    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_epoch = 0
        self.best = (np.Inf, np.Inf)


    def on_epoch_end(self, epoch, logs=None): 
        self.occ_model.save()
        
        val_loss = logs['val_loss']
        occ_accuracy = self.occ_model.evaluate(self.dset, epoch)
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



class OCCModel():
    def __init__(self, autoencoder, dataprocessor, dist_weight, vel_weight, leg_weight):
        self.autoencoder = autoencoder
        self.dataprocessor = dataprocessor

        self.predict_weights = tf.convert_to_tensor(self.dataprocessor.vel_mask * vel_weight + \
                                                    self.dataprocessor.leg_mask * leg_weight + \
                                                    self.dataprocessor.dist_mask * dist_weight, dtype=tf.float32)
        self.predict_weights_sum = tf.reduce_sum(self.predict_weights)
            

    def compile(self, modelsdir, name, optimizer, loss, ckpt_epochs=1, compile_loss=True):
        self.modelsdir = modelsdir
        self.name = name
        self.optimizer = optimizer
        self.loss = loss
        self.ckpt_epochs = ckpt_epochs
        self.epochs = 0

        if compile_loss:
            self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)
        else:
            self.autoencoder.compile(optimizer=self.optimizer)

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
        
        print(f'ckpt-{idx} loaded')


    @tf.function
    def predict(self, x, y=None):
        if y is None:
            y = self.autoencoder(x, training=False)
    
        err = tf.reduce_sum(self.predict_weights * tf.math.square(x - y), axis=1) / self.predict_weights_sum
        # err = tf.math.sqrt(err)
        
        return err
    

    @tf.function
    def discriminate(self, base_features, target_features):
        data = self.dataprocessor.preprocess_test(base_features, target_features)
        predicted = self.predict(data)

        return tf.reduce_mean(predicted)


    def fit(self, dataset, epochs, patience):
        tensorboard_callbacks = [
            myCallback(self, dataset, patience),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, update_freq='batch')
        ]

        self.autoencoder.fit(x=dataset.train, y=dataset.train,
                        batch_size=dataset.batch_size, epochs=self.epochs+epochs, 
                        initial_epoch=self.epochs,
                        shuffle=True,
                        validation_data=(dataset.test, dataset.test),
                        callbacks=tensorboard_callbacks, 
                        verbose=2)


    def evaluate(self, dataset, epoch):
        vel_true_predict = self.predict(dataset.test).numpy()
        leg_true_predict = vel_true_predict
        vel_mess_predict = self.predict(dataset.vel_mess).numpy()
        leg_mess_predict = self.predict(dataset.leg_mess).numpy()

        vel_mean, vel_var, leg_mean, leg_var = self.dataprocessor.validate(vel_true_predict, leg_true_predict,
            vel_mess_predict,
            leg_mess_predict,
            dataset.test_len,
            self.dir + f'/validation_{epoch}.png')
        
        with open(self.dir + f'/validation_{epoch}.txt', 'w') as f:
            f.write(f'vel_mean: \n{vel_mean} \n\nvel_var: \n{vel_var} \n\nleg_mean: \n{leg_mean} \n\nleg_var: \n{leg_var}')

        return np.mean(vel_mean) * self.loss.vel_weight + \
                np.mean(leg_mean) * self.loss.leg_weight
        