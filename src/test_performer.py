from performance.performer import Performer
import mido
import tensorflow as tf

p = Performer()
p.compile(r'C:\Users\mrshu\reps\music-style-performer\config\config_1025\config.json')

content = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\sibelius_clean.mid')
style = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\satie.mid')

p.style(content, style, A=10, stride=32, dt_max=0.01, outfile=r'C:\Users\mrshu\reps\music-style-performer\test\out\1025_sibelius_satie.mid', verbose=1)