import sys  
sys.path.insert(0, r'C:\Users\mrshu\reps\music-style-performer\src')


class Performer:
    def __init__(self):
        self.style_ratio_range = [0, 1]
        self.style_ratio_diff = self.style_ratio_range[1] - self.style_ratio_range[0]
        self.style_ratio_default = 0.5

        self.time_limit_range = [0, 5*60]
        self.time_limit_diff = self.time_limit_range[1] - self.time_limit_range[0]
        self.time_limit_default = 0

        self.max_tenuto_range = [0, 0.02]
        self.max_tenuto_diff = self.max_tenuto_range[1] - self.max_tenuto_range[0]
        self.max_tenuto_default = 0.01

        self.compiled = False


    def compile(self, configpath):
        from ml.discriminator.model import OCCModel
        from ml.discriminator.data_preprocessing import DataProcessorV0
        from ml.discriminator.autoencoder import MLPAutoencoder
        from ml.styling.styler import StylingModel
        from ml.styling.pca import PCA
        import json
        from synthesis.synth import Synth
        from synthesis.dbsamples import DBSamples
        from synthesis.fabric import PianoFabric


        with open(configpath, 'r') as f:
            data = json.load(f)

        dp = DataProcessorV0(data['data_processor']['notes_qty'],
                             data['data_processor']['include_first_tone'],
                             data['data_processor']['absolute_velocities'])
        dp.loadparams(data['data_processor']['config_path'])

        ae = MLPAutoencoder(dp.input_size,
                            data['autoencoder']['latent_dim'],
                            0,
                            data['autoencoder']['encoder_layers'],
                            data['autoencoder']['encoder_dropout'],
                            data['autoencoder']['decoder_layers'],
                            data['autoencoder']['decoder_dropout'])
        occ = OCCModel(ae,
                       dp,
                       data['occ']['dist_weight'],
                       data['occ']['vel_weight'],
                       data['occ']['leg_weight'])
        occ.load(data['occ']['checkpoint_path'])

        pca = PCA(data['pca']['pca_dim'])
        pca.loadparams(data['pca']['config_path'])

        self.styler = StylingModel(dp, occ, pca)

        dbsamples = DBSamples(data['synth']['sounds_dir'])
        fabric = PianoFabric()

        self.synth = Synth(fabric, dbsamples)

        self.compiled = True
        

    def style(self, mid_content, mid_style, stride=1, timelimit=None, A=10, dt_max=0.01, outfile=None, verbose=0):
        return self.styler.style(mid_content, mid_style, stride=stride, timelimit=timelimit, A=A, dt_max=dt_max, filename=outfile, verbose=verbose)
    

    def synthesize(self, mid, outfile=None):
        self.synth.synth_midi(mid, 0, outfile=outfile)

    
    def synth_style(self, mid_content, mid_style, stride=1, timelimit=None, A=10, dt_max=0.01, outfile=None, verbose=0):
        mid = self.style(mid_content, mid_style, stride, timelimit, A, dt_max, verbose=verbose)
        sound = self.synthesize(mid, 0, outfile)

        return sound