from gui.main_window import MainWindow
from PyQt5 import QtCore, QtWidgets, QtGui
import sys


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())