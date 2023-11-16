from performance.performer import Performer
import mido

p = Performer()
p.compile(r'C:\Users\mrshu\reps\music-style-performer\config\config_0025', 'config.json')

content = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\rachmaninoff_clean_vel.mid')
style = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\debussy_prelude.mid')

p.style(content, style, A=20, B=1, stride=1, dt_max=0.08, outfile=r'C:\Users\mrshu\reps\music-style-performer\test\out\test.mid', verbose=1)
