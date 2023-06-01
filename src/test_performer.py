from performance.performer import Performer
import mido
import tensorflow as tf

p = Performer()
p.compile(r'C:\Users\mrshu\reps\music-style-performer\config\config_0025\config.json')

content = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\sibelius_clean.mid')
style = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\satie_clean.mid')

p.style(content, style, A=50, B=1, stride=16, dt_max=0.02, outfile=r'C:\Users\mrshu\reps\music-style-performer\test\out\0025_sibelius_satieclean.mid', verbose=1)