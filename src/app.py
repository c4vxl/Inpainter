import time
import sys
import threading
import darkdetect

from PyQt6.QtWidgets import QApplication, QMainWindow, QStyleFactory
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QIcon

import server

class WebEngine(QWebEngineView):
    # Disable context menu (right click menu)
    def contextMenuEvent(self, event):
        event.ignore()
    
class MainWindow(QMainWindow):
    def __init__(self, app: QApplication, url: str):
        super().__init__()
        self.setWindowTitle("Inpainter")
        self.setWindowIcon(QIcon("./resources/logo.png"))
        
        self.app = app

        screen_width, screen_height = self._get_screen_size()
        factor = 0.9
        self.resize(int(screen_width * factor), int(screen_height * factor))

        self.view = WebEngine()
        self.setCentralWidget(self.view)
        self.view.setUrl(QUrl(url))

    def _get_screen_size(self) -> tuple[int, int]:
        geometry = self.app.primaryScreen().geometry()
        return geometry.width(), geometry.height()

    def closeEvent(self, event):
        """Stop server when the window is closed."""
        event.accept()

def start_server(port: int) -> threading.Thread:
    print(">> Starting backend...")

    def start():
        server.start(port=port)

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    
    return thread

def start_desktop(port: int, theme: str = "auto"):
    print(">> Starting desktop app...")

    if theme == "auto":
        is_dark = darkdetect.isDark()
        theme = "dark" if is_dark else "light"
        print("  | Detected theme: " + theme)

    app_qt = QApplication(sys.argv)
    app_qt.setStyle(QStyleFactory.create('Fusion'))
    window = MainWindow(app_qt, f"http://127.0.0.1:{port}?__theme={theme}")
    window.show()
    sys.exit(app_qt.exec())

if __name__ == "__main__":
    port = 3321

    start_server(port)
    time.sleep(0.2)
    start_desktop(port)