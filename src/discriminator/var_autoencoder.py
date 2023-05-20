import tensorflow as tf
from tensorflow import keras
from loss import  WeightedMSE



class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon        



class MLPVarAutoencoder(keras.Model):
    def __init__(self, 
            input_size,
            latent_dim,
            noise, 
            encoder_layers,
            encoder_dropout, 
            decoder_layers,
            decoder_dropout,
            beta,
            mse_loss):
        
        super().__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.noise = noise   
        self.encoder_layers = encoder_layers  
        self.encoder_dropout = encoder_dropout
        self.decoder_layers = decoder_layers  
        self.decoder_dropout = decoder_dropout
        self.beta = beta
        self.mse_loss = mse_loss

        self.build_encoder()
        self.build_decoder()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


    def sample(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=z_mean.shape)
        z_sampled = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_sampled


    def build_encoder(self):
        inp = keras.layers.Input(shape=(self.input_size))
        x = keras.layers.GaussianNoise(self.noise)(inp)
        
        for neurons, dropout in zip(self.encoder_layers, self.encoder_dropout):
            x = keras.layers.Dense(neurons, activation='relu')(x)
            if dropout > 0:
                x = keras.layers.Dropout(dropout)(x)
            
        z_mean = keras.layers.Dense(self.latent_dim, activation='tanh')(x)
        z_log_var = keras.layers.Dense(self.latent_dim, activation='tanh')(x)

        z = Sampling()([z_mean, z_log_var])

        self.encoder = keras.Model(inp, [z, z_mean, z_log_var], name="encoder")
        
        
    def build_decoder(self):
        self.decoder = keras.Sequential()

        self.decoder.add(keras.layers.Input(shape=(self.latent_dim)))
        
        for neurons, dropout in zip(self.decoder_layers, self.decoder_dropout):
            self.decoder.add(keras.layers.Dense(neurons, activation='relu'))
            if dropout > 0:
                self.decoder.add(keras.layers.Dropout(dropout))

        self.decoder.add(keras.layers.Dense(self.input_size, activation='linear'))


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    
    @tf.function
    def call(self, x, training=False):
        z, z_mean, z_log_var = self.encoder(x, training=training)
        decoded = self.decoder(z, training=training)
        return decoded
    

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encoder(data)

            reconstruction = self.decoder(z)
            reconstruction_loss = self.mse_loss(data, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + kl_loss * self.beta

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

    @tf.function
    def test_step(self, data):
        data, _ = data
        z, z_mean, z_log_var = self.encoder(data)

        reconstruction = self.decoder(z)
        reconstruction_loss = self.mse_loss(data, reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    

    def to_string(self):
        stringlist = []
        self.encoder.summary(print_fn=lambda x: stringlist.append(x))
        encoder_summary = "\n".join(stringlist)
        
        stringlist = []
        self.decoder.summary(print_fn=lambda x: stringlist.append(x))
        decoder_summary = "\n".join(stringlist)
        
        return f'latent dim: {self.latent_dim} \nnoise: {self.noise} \n\nencoder: {encoder_summary} \n\ndecoder: {decoder_summary}\n'