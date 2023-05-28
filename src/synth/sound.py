from control import Control


class Sound:
    def __init__(self, note, velocity, start, end=None) -> None:
        self.note = note
        self.velocity = velocity
        self.start = start
        self.end = end
        self.pedal = False


    def set_end(self, end: int, end_velocity=0):
        self.end = end
        self.end_velocity = end_velocity

    
    def add_control(self, control: Control):
        if control.cc == 64:
            self.pedal = control.value >= 64


class PianoSound(Sound):
    def __init__(self, note, velocity, start, end=None) -> None:
        super().__init__(note, velocity, start, end)


    def add_control(self, control: Control):
        super().add_control(control)

        pass
