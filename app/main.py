import sys
sys.path.append('./')
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from dialog import Ui_MainWindow
from model import * 

class CorrectText(QtWidgets.QMainWindow):
    def __init__(self):
        super(CorrectText, self).__init__()
        self.dialog = Ui_MainWindow()
        self.dialog.setupUi(self)
        self.init_UI()
    
    def init_UI(self):
        self.setWindowTitle('Исправление ошибок') 
        self.setWindowIcon(QIcon('Alice1.jpg'))
        self.dialog.textEdit.setPlaceholderText('Текст с ошибками')
        self.dialog.textEdit_2.setPlaceholderText('Исправленный текст')  
        self.dialog.pushButton.clicked.connect(self. generate_button_clicked)
        self.dialog.textEdit_2.setReadOnly(True)
        
    def generate_button_clicked(self):
        #pass
        self.dialog.textEdit_2.clear()
        input_text = self.dialog.textEdit
        if input_text.toPlainText() == "":
            self.dialog.textEdit_2.insertPlainText("Пожалуйста, введите текст!")
        else:
            self.dialog.textEdit_2.insertPlainText(generate_from_text(model_new, input_text.toPlainText(), num_beams=10))   

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    application = CorrectText()
    application.show()
    sys.exit(app.exec())