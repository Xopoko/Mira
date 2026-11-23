from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QImage,QPainter
from PySide6.QtCore import Qt
from PIL import Image
assets=Path(__file__).resolve().parents[1]/'context_operator'/'assets'
app=QApplication([])
image=QImage(256,256,QImage.Format_ARGB32);image.fill(Qt.transparent)
renderer=QSvgRenderer(str(assets/'icon.svg'));p=QPainter(image);renderer.render(p);p.end()
image.save(str(assets/'icon.png'))
Image.open(assets/'icon.png').save(assets/'icon.ico',sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
print('Application icon created.')
