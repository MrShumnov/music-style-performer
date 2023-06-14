import sys
import typing
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget
import gui.resources
import mido
import os
from performance.performer import Performer


class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.performer = Performer()
        self.content = None
        self.style = None

        self.initUI()


    def initUI(self):
        self.setFixedSize(804, 549)
        self.setWindowTitle(' ')
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QtGui.QColor(0, 0, 0, 0))
        self.setWindowIcon(QtGui.QIcon(pixmap))
        self.setStyleSheet('''
            QMessageBox {
                text-align: left;
            }
        ''')

        QtGui.QFontDatabase.addApplicationFont(':/res/fonts/Raleway-Regular.ttf')
        QtGui.QFontDatabase.addApplicationFont(':/res/fonts/Raleway-Medium.ttf')
        id = QtGui.QFontDatabase.addApplicationFont(':/res/fonts/Raleway-SemiBold.ttf')
        families = QtGui.QFontDatabase.applicationFontFamilies(id)
        self.fontname = families[0]

        label = QtWidgets.QLabel(self)
        label.setFixedSize(804, 549)
        label.setStyleSheet("background-image: url(:/res/img/main_window.svg); background-attachment: fixed")

        self.piano_shift = (-21, 23)
        self.text_shift = (0, -3)

        self.create_label("StylePerformer", (65, 21), (150, 33), align=QtCore.Qt.AlignLeft)

        self.create_label("Content MIDI", (17, 121), (173, 36), shift=self.piano_shift)
        self.create_label("Style MIDI", (19, 165), (173, 36), shift=self.piano_shift)
        self.create_label("Style ratio", (89, 253), (101, 36), shift=self.piano_shift)
        self.create_label("Max tenuto", (39, 297), (151, 36), shift=self.piano_shift)
        self.create_label("Time limit", (41, 341), (151, 36), shift=self.piano_shift)

        self.content_button = self.create_label("open...", (240, 100), (290, 39), 'black', QtCore.Qt.AlignLeft, shift=self.piano_shift)
        self.content_button.mousePressEvent = self.btn_content_clicked

        self.style_button = self.create_label("open...", (240, 144), (290, 39), 'black', QtCore.Qt.AlignLeft, shift=self.piano_shift)
        self.style_button.mousePressEvent = self.btn_style_clicked

        self.style_ratio_label = self.create_label("5", (240, 232), (60, 39), 'black', QtCore.Qt.AlignLeft, shift=self.piano_shift)
        self.max_tenuto_label = self.create_label("0", (240, 276), (60, 39), 'black', QtCore.Qt.AlignLeft, shift=self.piano_shift)
        self.time_limit_label = self.create_label("None", (240, 320), (60, 39), 'black', QtCore.Qt.AlignLeft, shift=self.piano_shift)

        self.create_midi_button = self.create_label("Create MIDI", (696, 124), (173, 36), 'black', QtCore.Qt.AlignLeft)
        self.create_midi_button.mousePressEvent = self.create_midi
        self.create_wav_button = self.create_label("Create WAV", (696, 169), (173, 36), 'black', QtCore.Qt.AlignLeft)
        self.create_wav_button.mousePressEvent = self.create_wav
        self.load_config_button = self.create_label("Load config", (696, 213), (173, 36), 'black', QtCore.Qt.AlignLeft)
        self.load_config_button.mousePressEvent = self.load_config

        self.create_midi_icon = IconButton(self, ':/res/img/midi_icon.svg', (623, 126), (33, 33))
        self.create_wav_icon = IconButton(self, ':/res/img/wav_icon.svg', (623, 170), (33, 33))
        self.load_config_icon = IconButton(self, ':/res/img/conf_icon.svg', (623, 214), (33, 33))

        self.style_ratio_slider = self.create_slider((323, 232))
        self.style_ratio_slider.setMinimum(0)
        self.style_ratio_slider.setMaximum(self.performer.style_ratio_diff / 0.1)
        self.style_ratio_slider.setSingleStep(1)
        self.style_ratio_slider.setPageStep(1)
        self.style_ratio_slider.valueChanged.connect(self.style_ratio_slider_value_changed)
        self.style_ratio_slider.setValue(self.performer.style_ratio_default / self.performer.style_ratio_diff * self.style_ratio_slider.maximum())

        self.max_tenuto_slider = self.create_slider((323, 276))
        self.max_tenuto_slider.setMinimum(0)
        self.max_tenuto_slider.setMaximum(self.performer.max_tenuto_diff // 0.001)
        self.max_tenuto_slider.setSingleStep(1)
        self.max_tenuto_slider.setPageStep(1)
        self.max_tenuto_slider.valueChanged.connect(self.max_tenuto_slider_value_changed)
        self.max_tenuto_slider.setValue(self.performer.max_tenuto_default / self.performer.max_tenuto_diff * self.max_tenuto_slider.maximum())

        self.time_limit_slider = self.create_slider((323, 320))
        self.time_limit_slider.setMinimum(0)
        self.time_limit_slider.setMaximum(self.performer.time_limit_diff // 10)
        self.time_limit_slider.setSingleStep(10)
        self.time_limit_slider.setPageStep(10)
        self.time_limit_slider.valueChanged.connect(self.time_limit_slider_value_changed)
        self.time_limit_slider.setValue(6)

    
    def create_midi(self, _):
        if not self.performer.compiled:
            QtWidgets.QMessageBox.about(self, "Error", "Config should be loaded first")
            return
            
        if self.content is None:
            QtWidgets.QMessageBox.about(self, "Error", "Content MIDI should be loaded first")
            return
            
        if self.style is None:
            QtWidgets.QMessageBox.about(self, "Error", "Style MIDI should be loaded first")
            return

        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save MIDI', None, "MIDI (*.mid)")
        
        if filename[0]:
            timelimit = self.get_time_limit()
            if timelimit == 0:
                timelimit = None
            dt_max = self.get_max_tenuto()
            a = self.get_style_ratio()

            try:
                self.performer.style(self.content, self.style, A=a, dt_max=dt_max, timelimit=timelimit, outfile=filename[0], verbose=1)
            except Exception as e:
                QtWidgets.QMessageBox.about(self, "Error", f"Unexpected error: \n{e}")
            

    def create_wav(self, _):
        if not self.performer.compiled:
            QtWidgets.QMessageBox.about(self, "Error", "Config should be loaded first")
            return
            
        if self.content is None:
            QtWidgets.QMessageBox.about(self, "Error", "Content MIDI should be loaded first")
            return

        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save MIDI', None, "Waveform Audio File (*.wav)")
        
        if filename[0]:
            if self.style is None:
                try:
                    self.performer.synthesize(self.content, outfile=filename[0])
                except Exception as e:
                    QtWidgets.QMessageBox.about(self, "Error", f"Unexpected error: \n{e}")
            else:
                timelimit = self.get_time_limit()
                if timelimit == 0:
                    timelimit = None
                dt_max = self.get_max_tenuto()
                a = self.get_style_ratio()

                try:
                    self.performer.synth_style(self.content, self.style, A=a, dt_max=dt_max, timelimit=timelimit, outfile=filename[0], verbose=1)
                except Exception as e:
                    QtWidgets.QMessageBox.about(self, "Error", f"Unexpected error: \n{e}")


    def load_config(self, _):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Load config', None, "JSON (*.json)")
        
        if filename[0]:
            try:
                self.performer.compile(filename[0])
            except Exception as e:
                QtWidgets.QMessageBox.about(self, "Error", f"Unexpected error: \n{e}")


    def get_style_ratio(self):
        return self.style_ratio_slider.value() / self.style_ratio_slider.maximum() * self.performer.style_ratio_diff
    
    def get_max_tenuto(self):
        return self.max_tenuto_slider.value() / self.max_tenuto_slider.maximum() * self.performer.max_tenuto_diff
    
    def get_time_limit(self):
        return self.time_limit_slider.value() / self.time_limit_slider.maximum() * self.performer.time_limit_diff


    def style_ratio_slider_value_changed(self):
        self.style_ratio_label.setText('{:.1f}'.format(self.get_style_ratio()))

    def max_tenuto_slider_value_changed(self):
        self.max_tenuto_label.setText('{:.0f} ms'.format(self.get_max_tenuto() * 1000))

    def time_limit_slider_value_changed(self):
        value = self.get_time_limit() / 60
        self.time_limit_label.setText('{:.1f} min'.format(value) if value > 0 else '∞')


    def btn_content_clicked(self, pos):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Load content MIDI', None, "MIDI (*.mid)")
        
        if filename[0]:
            try:
                content = mido.MidiFile(filename[0])
            except:
                QtWidgets.QMessageBox.about(self, "Error", "Invalid MIDI file")
                return
            
            self.content = content
            self.content_button.setText(os.path.basename(filename[0]))
        else:
            self.content = None
            self.content_button.setText('open...')


    def btn_style_clicked(self, pos):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Load style MIDI', None, "MIDI (*.mid)")
        
        if filename[0]:
            try:
                style = mido.MidiFile(filename[0])
            except:
                QtWidgets.QMessageBox.about(self, "Error", "Invalid MIDI file")
                return
            
            self.style = style
            self.style_button.setText(os.path.basename(filename[0]))
        else:
            self.style = None
            self.style_button.setText('open...')


    def create_label(self, text, pos, size, color='white', align=QtCore.Qt.AlignRight, shift=(0, 0)):
        label_stylesheet = f'''
            color: {color}; font-weight: medium;
        '''

        label = QtWidgets.QLabel(text, self)
        label.move(QtCore.QPoint(pos[0] + shift[0] + self.text_shift[0], pos[1] + shift[1] + self.text_shift[1]))
        label.setFixedSize(*size)
        label.setAlignment(align | QtCore.Qt.AlignVCenter)
        label.setFont(QtGui.QFont(self.fontname, 10))
        label.setStyleSheet(label_stylesheet)

        return label
    

    def create_slider(self, pos):
        slider = QtWidgets.QSlider(self)
        slider.setOrientation(QtCore.Qt.Horizontal)
        slider.setStyleSheet(self.stylesheet())
        slider.move(pos[0] + self.piano_shift[0], pos[1] + self.piano_shift[1])

        return slider
    

    def stylesheet(self):
        return """
            QSlider {
                height: 39px;
                width: 200px;
            }

            QSlider::groove:horizontal {
                background: #F5F5F5;
            }

            QSlider::sub-page:horizontal {
                background: #F5F5F5;
            }

            QSlider::add-page:horizontal {
                background: #121212;
            }

            QSlider::handle:horizontal {
                background: #F5F5F5;
                width: 20px;
            }

            QSlider::handle:horizontal:hover {
                background: #8DC3BA;
            }
        """
        

class IconButton(QtWidgets.QLabel):
    def __init__(self, parent: QWidget, imgpath, pos, size):
        super().__init__(parent)

        self.setStyleSheet(f"background-image: url({imgpath}); background-attachment: fixed")
        self.move(QtCore.QPoint(*pos))
        self.setFixedSize(QtCore.QSize(*size))
