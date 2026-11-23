import ctypes
import os
import sys
import traceback
from pathlib import Path

def main():
    os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
    if os.name=='nt':
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('ContextOperator.Desktop')
    from PySide6.QtWidgets import QApplication,QMessageBox
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtNetwork import QLocalServer,QLocalSocket
    from PySide6.QtCore import QTimer
    from .settings import load,LOCAL
    from .controller import Controller
    from .ui.window import Window
    from .ui.overlay import FocusOverlay
    app=QApplication(sys.argv);app.setApplicationName('Context Operator');app.setApplicationDisplayName('Mira');app.setOrganizationName('ContextOperator')
    for font in ('segoeui.ttf','seguisb.ttf','segoeuib.ttf','SegUIVar.ttf','CascadiaMono.ttf'):
        path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/font
        if path.is_file():QFontDatabase.addApplicationFont(str(path))
    socket=QLocalSocket();socket.connectToServer('ContextOperator.Desktop')
    if socket.waitForConnected(150):socket.write(b'show');socket.waitForBytesWritten(100);return 0
    QLocalServer.removeServer('ContextOperator.Desktop');server=QLocalServer();server.listen('ContextOperator.Desktop')
    window=Window(Controller(load()))
    if '--gaze-settings' in sys.argv:window.navigate(3)
    def show_existing():
        connection=server.nextPendingConnection()
        if connection:connection.disconnectFromServer();connection.deleteLater()
        window.showNormal();window.raise_();window.activateWindow()
    server.newConnection.connect(show_existing)
    window.show()
    for screen in app.screens():window.overlays.append(FocusOverlay(screen,window.c.win))
    if '--bind-button' in sys.argv:QTimer.singleShot(250,window.bind_usb_button)
    def exception(kind,error,tb):
        LOCAL.mkdir(parents=True,exist_ok=True)
        (LOCAL/'crash.log').write_text(''.join(traceback.format_exception(kind,error,tb)),encoding='utf-8')
        QMessageBox.critical(window,'Mira','Mira encountered an error:'+str(error))
    sys.excepthook=exception
    return app.exec()

if __name__=='__main__':sys.exit(main())
