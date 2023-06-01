from synthesis.dbsamples import DBSamples
from synthesis.synth import Synth
from synthesis.fabric import PianoFabric
import mido

mid = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\out\1025_sibelius_satie.mid')

dbsamples = DBSamples(r'C:\Users\mrshu\reps\music-style-performer\sounds\corpus_wav_lufs')
fabric = PianoFabric()

synth = Synth(fabric, dbsamples)

synth.synth_midi(mid, 0, outfile=r'C:\Users\mrshu\reps\music-style-performer\test\synth\1025_sibelius_satie.wav')

