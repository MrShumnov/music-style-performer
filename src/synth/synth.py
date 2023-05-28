from fabric import InstrumentFabric
from event_manager import EventManager
from dbsamples import DBSamples
from scipy.io import wavfile
import numpy as np


class Synth:
    def __init__(self, fabric: InstrumentFabric, dbsamples: DBSamples) -> None:
        self.fabric = fabric
        self.dbsamples = dbsamples
        self.producer = self.fabric.create_producer(dbsamples)


    def synth_midi(self, mid, track_num, outfile=None):
        event_manager = EventManager(self.fabric)
        
        time = 0

        for m in mid.tracks[track_num]:
            time += m.time / mid.ticks_per_beat / 2

            if m.type == 'note_on':
                if m.velocity > 0:
                    event_manager.note_on(time, m.note, m.velocity)
                else:
                    event_manager.note_off(time, m.note, 0)
            
            elif m.type == 'note_off':
                event_manager.note_off(time, m.note, m.velocity)

            elif m.type == 'control_change':
                event_manager.control(time, m.control, m.value)

        result = self.producer.synthesis(event_manager.shedule, event_manager.duration)

        self.dbsamples.clear_cache()

        if outfile is not None:
            wavfile.write(outfile, self.dbsamples.sample_rate, np.swapaxes(result, 0, 1))

        return result
            

            