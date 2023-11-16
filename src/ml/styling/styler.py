import tensorflow as tf
import numpy as np
import time
from ml.discriminator.data_preprocessing import rec2mid, track2line, window_tf
from ml.styling.pca import PCA


class StylingModel:
    def __init__(self, data_processor, occ, pca) -> None:
        self.dp = data_processor
        self.occ = occ
        self.pca = pca


    def style_loss(self, encoded): 
        predicted_pca = self.pca.project(encoded)
        return tf.reduce_mean(tf.math.square(predicted_pca - self.style_pca))


    def quality_loss(self, predicted, decoded):
        loss = self.occ.predict(predicted, decoded)
        return tf.math.reduce_mean(loss)


    def overall_loss(self, dt, vel, leg):
        predicted = self.process(dt, vel, leg)
        
        encoded = self.occ.encode(predicted)
        decoded = self.occ.decode(encoded)
        
        lstyle = self.style_loss(encoded)
        lquality = self.quality_loss(predicted, decoded)
        
        ltotal = self.A * lstyle + self.B * lquality 
        
        return ltotal, lstyle, lquality


    @tf.function()
    def train_step(self, dt, vel, leg, opt):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(dt)
            tape.watch(vel)
            tape.watch(leg)
            ltotal, lstyle, lquality = self.overall_loss(dt, vel, leg)

        gdt = tape.gradient(ltotal, dt)
        gvel = tape.gradient(ltotal, vel)
        gleg = tape.gradient(ltotal, leg)
        
        opt.apply_gradients([(gvel, vel), (gleg, leg), (gdt, dt)])
        
        return ltotal, lstyle, lquality
    

    def train_cycle(self, dt, vel, leg, opt, steps_per_epoch=500, timelimit=None, verbose=0, epochs=30):
        print('Start training')
        start = time.time()

        for e in range(epochs):
            for s in range(steps_per_epoch):
                ltotal, lstyle, lquality = self.train_step(dt, vel, leg, opt)

            if verbose == 1:
                print(f"Step: {(e + 1) * steps_per_epoch} | total: {ltotal}, style: {lstyle}, quality: {lquality}")

            if timelimit is not None and time.time() - start > timelimit:
                break

        end = time.time()
        if verbose == 1:
            print("Total time: {:.1f}".format(end-start))

        return dt, vel, leg
    

    def style(self, mid_content, mid_style, stride=1, timelimit=None, A=10, B=10, dt_max=0.01, filename=None, verbose=0, seed=101, epochs=30):
        self.stride = stride
        self.A = A
        self.B = B
        self.dt_max = dt_max

        np.random.seed(seed)
        tf.random.set_seed(seed)

        line_content, tones_content, dists_content = self.dp.process_test(mid_content, 0, stride)
        line_style, _, _ = self.dp.process_test(mid_style, 0, stride)

        self.base = line_content[..., :2]
        self.base_tones = tones_content
        self.base_dists = dists_content

        # print(line_content.shape, tones_content.shape, dists_content.shape)

        style = self.dp.reshape_test(line_style)
        style_en = self.occ.encode(style)
        self.style_pca = self.pca.project(style_en)
        self.style_pca = tf.reduce_mean(self.style_pca, axis=0) # np.median(self.style_pca, axis=0)
        self.style_pca = tf.repeat(self.style_pca[tf.newaxis, :], self.base.shape[0], axis=0)

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        dtstart = np.random.normal(-1, 0.01, (self.base_tones.shape[0], 1))
        velstart = np.random.normal(0, 1, (self.base_tones.shape[0], 1))
        legstart = np.random.normal(0, 1, (self.base_tones.shape[0], 1))

        dt = tf.Variable(dtstart, dtype=tf.float32)
        vel = tf.Variable(velstart, dtype=tf.float32)
        leg = tf.Variable(legstart, dtype=tf.float32)

        dt, vel, leg = self.train_cycle(dt, vel, leg, opt, timelimit=timelimit, verbose=verbose, epochs=epochs)

        rec_dist, rec_vel, rec_leg = self.reconstruct(dt, vel, leg)
        mid = rec2mid(rec_dist, rec_vel, rec_leg, self.base_tones, mid_content.ticks_per_beat, filename)

        return mid, (rec_dist, rec_vel, rec_leg)


    def dt_norm(self, dt):
        return -1 * self.dt_max * (1 / (1 + tf.math.exp(-dt + 6))) / self.dp.normparams[2]

    def dt_additive(self, dt):
        dt = tf.concat([self.dt_norm(dt), tf.constant(0, shape=(1,1), dtype=tf.float32)], axis=0)
        return tf.experimental.numpy.diff(dt, axis=0)
    

    def process(self, dt, vel, leg):
        dt = self.dt_additive(dt)

        # window
        pdt = window_tf(dt, self.dp.notes_qty, self.stride)
        pvel = window_tf(vel, self.dp.notes_qty, self.stride)
        pleg = window_tf(leg, self.dp.notes_qty, self.stride)

        # print(dt.shape, vel.shape, leg.shape, self.base.shape, len(pdt), len(pvel), len(pleg))
        
        data = tf.concat([self.base[..., :1], self.base[..., 1:2] + pdt, pvel, pleg], axis=-1)
        data = tf.reshape(data, [data.shape[0], data.shape[1] * data.shape[2]])
        if not self.dp.include_first_tone:
            data = data[:, 1:]
        
        return data


    def reconstruct(self, dt, vel, leg):
        dist = (self.base_dists[:, tf.newaxis] + self.dt_additive(dt)) * self.dp.normparams[2]
        vel = vel * self.dp.normparams[4] + self.dp.normparams[3]
        leg = leg + 1 
        
        return tf.reshape(dist, shape=(self.base_dists.shape[0])), \
                tf.reshape(vel, shape=(self.base_dists.shape[0])), \
                tf.reshape(leg, shape=(self.base_dists.shape[0]))
