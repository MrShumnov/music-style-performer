import tensorflow as tf
from tensorflow import keras



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
            self.encoder.add(keras.layers.Dense(neurons, activation='relu'))
            if dropout > 0:
                self.encoder.add(keras.layers.Dropout(dropout))
        
        self.encoder.add(keras.layers.Dense(self.latent_dim, activation='tanh'))
        
        
    def build_decoder(self):
        self.decoder = keras.Sequential()

        self.decoder.add(keras.layers.Input(shape=(self.latent_dim)))
        
        for neurons, dropout in zip(self.decoder_layers, self.decoder_dropout):
            self.decoder.add(keras.layers.Dense(neurons, activation='relu'))
            if dropout > 0:
                self.decoder.add(keras.layers.Dropout(dropout))

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
        