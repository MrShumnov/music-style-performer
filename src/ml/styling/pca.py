import tensorflow as tf
import numpy as np


class PCA:
    def __init__(self, pca_dim):
        self.pca_dim = pca_dim
        self.principals = None
        self.mean = None


    def calc_principals(self, autoencoder, dset):
        encoded = autoencoder.encoder(dset.train)
        self.principals, self.mean = self.pca(encoded)

    
    def loadparams(self, filename):
        with open(filename, 'r') as f:
            s = f.read().split('\n')
            h, w = list(map(int, s[0].split(' ')))

            l = []
            for i in range(h):
                row = list(map(float, s[1 + i].split(' ')[:-1]))
                l.append(row)
            self.principals = tf.convert_to_tensor(l, dtype=tf.float32)

            w = int(s[1 + h])
            l = list(map(float, s[2 + h].split(' ')[:-1]))
            self.mean = tf.convert_to_tensor(l, dtype=tf.float32)


    def saveparams(self, filename):
        with open(filename, 'w') as f:
            f.write(f'{self.principals.shape[0]} {self.principals.shape[1]}\n')

            for i in range(self.principals.shape[0]):
                for j in range(self.principals.shape[1]):
                    f.write(str(float(self.principals[i, j])) + ' ')
                f.write('\n')

            f.write(f'{self.mean.shape[0]}\n')
            for i in range(self.mean.shape[0]):
                f.write(f'{self.mean[i]} ')
        

    def pca(self, data):
        mean = tf.reduce_mean(data, axis=0)
        centered_data = data - mean
        covariance_matrix = tf.matmul(tf.transpose(centered_data), centered_data) / tf.cast(tf.shape(centered_data)[0], tf.float32)

        eigenvalues, eigenvectors = tf.linalg.eigh(covariance_matrix)

        top_k_eigenvectors = eigenvectors[:, -self.pca_dim:]

        return top_k_eigenvectors, mean


    def project(self, data):
        centered_data = data - self.mean
        projected_data = tf.matmul(centered_data, self.principals) 
        
        return projected_data