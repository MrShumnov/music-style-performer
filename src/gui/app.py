import sys  
sys.path.insert(0, r'C:\Users\mrshu\reps\music-style-performer\src')
import os
path = os.path.dirname(__file__)

import sys
from PyQt5 import QtCore, QtWidgets, QtGui


class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.initUI()

        self.content = None
        self.style = None


    def initUI(self):
        background = '#035d63'
        self.setMinimumSize(1200, 500)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setStyleSheet('''
            background-color: {:s};
            font-family: Segoe;
        '''.format(background))

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0,0,0,0)

        self.btn_config = self.create_load_button(' Open config file', self.btn_config_clicked)
        self.btn_content = self.create_load_button(' Load content MIDI', self.btn_content_clicked)
        self.btn_style = self.create_load_button(' Load style MIDI', self.btn_style_clicked)
        self.btn_generate = self.create_load_button('Generate result', self.btn_generate_clicked, False)
        self.btn_generate.setVisible(False)

        self.canvas_config = self.create_canvas()
        self.canvas_content = self.create_canvas()
        self.canvas_style = self.create_canvas()

        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.setContentsMargins(0,0,0,0)

        self.layout.addWidget(TaskBar(self, 'StylePerformer'))
        self.layout.addStretch(-1)
        self.layout.addLayout(self.main_layout)
        self.layout.addWidget(self.btn_generate, alignment=QtCore.Qt.AlignCenter)
        self.layout.addStretch(1)
        
        self.layout_config = QtWidgets.QVBoxLayout()
        self.layout_config.setContentsMargins(0,0,0,0)
        self.layout_content = QtWidgets.QVBoxLayout()
        self.layout_content.setContentsMargins(0,0,0,0)
        self.layout_style = QtWidgets.QVBoxLayout()
        self.layout_style.setContentsMargins(0,0,0,0)

        self.main_layout.addLayout(self.layout_config)
        self.main_layout.addLayout(self.layout_content)
        self.main_layout.addLayout(self.layout_style)

        self.layout_config.addWidget(self.canvas_config, alignment=QtCore.Qt.AlignCenter)
        self.layout_config.addWidget(self.btn_config, alignment=QtCore.Qt.AlignCenter)
        self.layout_content.addWidget(self.canvas_content, alignment=QtCore.Qt.AlignCenter)
        self.layout_content.addWidget(self.btn_content, alignment=QtCore.Qt.AlignCenter)
        self.layout_style.addWidget(self.canvas_style, alignment=QtCore.Qt.AlignCenter)
        self.layout_style.addWidget(self.btn_style, alignment=QtCore.Qt.AlignCenter)

        # self.performer = Performer()

        
    def create_load_button(self, text, func, icon=True):
        btn_load = QtWidgets.QPushButton(text)
        btn_load.clicked.connect(func)
        if icon:
            btn_load.setIcon(QtGui.QIcon(path + '/img/add.png'))
            btn_load.setIconSize(QtCore.QSize(25, 25))
        btn_load.setStyleSheet('''
            QPushButton {
                color: #ffffff;
                font-size: 10pt;
                border-radius: 0px;
            }
            QPushButton::hover { 
                background-color: #7ea2aa;
            }
        ''')
        btn_load.setFixedSize(btn_load.sizeHint().width() + 30, 
                                        btn_load.sizeHint().height() + 10)
        
        return btn_load


    def create_canvas(self):
        canvas = QtWidgets.QLabel()
        canvas.setFixedSize(QtCore.QSize(370, 270))
        canvas.setVisible(False)

        return canvas
    
    def btn_config_clicked(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Load config file', None, "JSON (*.json)")
        
        if filename[0]:
            from performance.performer import Performer

            try:
                self.performer = Performer(filename[0])
                self.canvas_content.setVisible(True)
            except:
                QtWidgets.QMessageBox.about(self, "Error", "Invalid configuration")
                return

        if self.content is not None and self.style is not None and self.config is not None:
            self.btn_generate.setVisible(True)        
    
    def btn_content_clicked(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Load content MIDI', None, "MIDI (*.mid)")
        
        if filename[0]:
            self.content = filename[0]
            self.canvas_content.setVisible(True)

        if self.content is not None and self.style is not None and self.config is not None:
            self.btn_generate.setVisible(True)
    
    def btn_style_clicked(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Load style MIDI', None, "MIDI (*.mid)")

        if filename[0]:
            self.style = filename[0]
                
            self.canvas_style.setVisible(True)
                
        if self.content is not None and self.style is not None and self.config is not None:
            self.btn_generate.setVisible(True)

    def btn_generate_clicked(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save result MIDI', None, "MIDI (*.mid)")

        if filename[0]:
            result = self.performer.synth_style(self.content, self.style, timelimit=120, stride=32, outfile=filename[0])


class TaskBar(QtWidgets.QWidget):

    def __init__(self, parent, title):
        super(TaskBar, self).__init__()
        self.parent = parent
        
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.title = QtWidgets.QLabel(title)

        btn_size = 30
        background = '#023c40'

        self.btn_close = QtWidgets.QPushButton()
        self.btn_close.clicked.connect(self.btn_close_clicked)
        self.btn_close.setFixedSize(btn_size,btn_size)
        self.btn_close.setIcon(QtGui.QIcon(path + '/img/close.png'))
        self.btn_close.setIconSize(QtCore.QSize(25, 25))
        self.btn_close.setStyleSheet('''
            background-color: {:s};
            border-radius: 0px;
        '''.format(background))

        self.btn_min = QtWidgets.QPushButton()
        self.btn_min.clicked.connect(self.btn_min_clicked)
        self.btn_min.setFixedSize(btn_size, btn_size)
        self.btn_min.setIcon(QtGui.QIcon(path + '/img/minimize.png'))
        self.btn_min.setIconSize(QtCore.QSize(25, 25))
        self.btn_min.setStyleSheet('''
            background-color: {:s};
            border-radius: 0px;
        '''.format(background))

        self.btn_max = QtWidgets.QPushButton()
        self.btn_max.clicked.connect(self.btn_max_clicked)
        self.btn_max.setFixedSize(btn_size, btn_size)
        self.btn_max.setIcon(QtGui.QIcon(path + '/img/restore.png'))
        self.btn_max.setIconSize(QtCore.QSize(25, 25))
        self.btn_max.setStyleSheet('''
            background-color: {:s};
            border-radius: 0px;
        '''.format(background))

        self.title.setFixedHeight(35)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.btn_min)
        self.layout.addWidget(self.btn_max)
        self.layout.addWidget(self.btn_close)

        self.title.setStyleSheet('''
            background-color: {:s};
            color: #ffffff;
            font-size: 10pt;
        '''.format(background))
        self.setLayout(self.layout)

        self.start = QtCore.QPoint(0, 0)
        self.pressing = False

    def resizeEvent(self, QResizeEvent):
        super(TaskBar, self).resizeEvent(QResizeEvent)
        self.title.setFixedWidth(self.parent.width())

    def mousePressEvent(self, event):
        self.start = self.mapToGlobal(event.pos())
        self.pressing = True

    def mouseMoveEvent(self, event):
        if self.pressing:
            self.end = self.mapToGlobal(event.pos())
            self.movement = self.end-self.start
            self.parent.setGeometry(self.mapToGlobal(self.movement).x(),
                                self.mapToGlobal(self.movement).y(),
                                self.parent.width(),
                                self.parent.height())
            self.start = self.end

    def mouseReleaseEvent(self, QMouseEvent):
        self.pressing = False


    def btn_close_clicked(self):
        self.parent.close()

    def btn_max_clicked(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()

    def btn_min_clicked(self):
        self.parent.showMinimized()