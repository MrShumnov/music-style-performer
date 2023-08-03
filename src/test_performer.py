from performance.performer import Performer
import mido

p = Performer()
p.compile(r'C:\Users\mrshu\Documents\music-style-performer\config\config_0025\config.json')

content = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\chopin_clean.mid')
style = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\satie.mid')

print(p.style(content, style, A=20, B=1, stride=1, dt_max=0.08, verbose=1))