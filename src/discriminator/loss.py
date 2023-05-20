import tensorflow as tf



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