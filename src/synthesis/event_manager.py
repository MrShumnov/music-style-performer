from synthesis.control import Control

MIDI_NOTES = 127

class EventManager:
    def __init__(self, fabric) -> None:
        self.fabric = fabric

        self.cur_sounds = [None] * MIDI_NOTES
        self.cur_controls = {}
        self.shedule = []
        self.duration = 0

    
    def note_on(self, time, note, velocity) -> None:
        prev_sound = self.cur_sounds[note]
        if prev_sound is not None:
            prev_sound.set_end(time)
            self.shedule.append(prev_sound)
            self.duration = max(self.duration, prev_sound.end)

        sound = self.fabric.create_sound(note, velocity, time)
        for cc in self.cur_controls:
            sound.add_control(self.cur_controls[cc])

        self.cur_sounds[note] = sound


    def note_off(self, time, note, velocity) -> None:
        sound = self.cur_sounds[note]

        if sound is not None:
            sound.set_end(time, velocity)

            if not sound.pedal:
                self.shedule.append(sound)
                self.duration = max(self.duration, sound.end)
                self.cur_sounds[note] = None


    def control(self, time, cc, value):        
        control = Control(time, cc, value)
        
        for sound in self.cur_sounds:
            if sound is not None:
                sound.add_control(control)

        self.cur_controls[cc] = control

        # sustain pedal
        if cc == 64 and value < 64:
            for sound in self.cur_sounds:
                if sound is not None and sound.end is not None:
                    self.note_off(time, sound.note, value)
