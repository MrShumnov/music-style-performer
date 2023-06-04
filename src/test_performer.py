from performance.performer import Performer
import mido
import tensorflow as tf

p = Performer()
p.compile(r'C:\Users\mrshu\reps\music-style-performer\config\config_0025\config.json')

content = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\chopin_clean.mid')
style = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\satie.mid')

p.style(content, style, A=20, B=1, stride=1, dt_max=0.08, outfile=r'C:\Users\mrshu\reps\music-style-performer\test\out\0025_chopin_satie.mid', verbose=1)