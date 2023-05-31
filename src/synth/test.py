from dbsamples import DBSamples
from synth import Synth
from fabric import PianoFabric
import mido

mid = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\out\0023_sibelius_satie.mid')

dbsamples = DBSamples(r'C:\Users\mrshu\reps\music-style-performer\sounds\corpus_wav_1')
fabric = PianoFabric()

synth = Synth(fabric, dbsamples)

synth.synth_midi(mid, 0, outfile=r'C:\Users\mrshu\reps\music-style-performer\test\synth\0023_sibelius_satie.wav')

