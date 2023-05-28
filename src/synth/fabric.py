from sound import PianoSound
from producer import PianoSoundProducer


class InstrumentFabric:
    def create_sound(self, note, velocity, start):
        pass 


    def create_producer(self):
        pass



class PianoFabric(InstrumentFabric):
    def create_sound(self, note, velocity, start):
        return PianoSound(note, velocity, start)
    

    def create_producer(self, dbsamples):
        return PianoSoundProducer(dbsamples)