import sys
# add path to src folder
sys.path.append(r'C:\Users\mrshu\reps\music-style-performer\src') 

# import facade class
from src.performance.performer import Performer 
import mido
import tensorflow as tf

p = Performer()
# compile with path & filename of the config
p.compile(r'C:\Users\mrshu\reps\music-style-performer\config\config_0025', 'config.json') 

# content midi (>64 notes only!)
content = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\rachmaninoff_clean_vel.mid')
# style midi (>64 notes only!)
style = mido.MidiFile(r'C:\Users\mrshu\reps\music-style-performer\test\debussy_prelude.mid')

p.style(content, style, A=20, B=1, stride=1, dt_max=0.08, outfile=r'C:\Users\mrshu\reps\music-style-performer\test\out\test.mid', verbose=1)