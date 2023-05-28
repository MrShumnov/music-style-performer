from dbsamples import DBSamples
from synth import Synth
from fabric import PianoFabric
import mido

mid = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\src\synth\sibelius_clean.mid')

dbsamples = DBSamples(r'C:\Users\mrshu\reps\music-style-performer\sounds\corpus_wav')
fabric = PianoFabric()

synth = Synth(fabric, dbsamples)

synth.synth_midi(mid, 0, outfile=r'C:\Users\mrshu\reps\music-style-performer\src\synth\test_sibelius1.wav')

