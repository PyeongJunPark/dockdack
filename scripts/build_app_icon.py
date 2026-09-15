"""Render the original vector logo into PNG-compressed Windows ICO sizes."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import struct
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer


def main():
    app = QGuiApplication.instance() or QGuiApplication([])
    folder = Path(__file__).resolve().parents[1] / "dockdack" / "assets"
    renderer = QSvgRenderer(str(folder / "dockdack-mark.svg"))
    if not renderer.isValid():
        raise ValueError("Invalid source SVG")
    frames = []
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        bitmap = QImage(size, size, QImage.Format.Format_ARGB32)
        bitmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(bitmap)
        renderer.render(painter)
        painter.end()
        payload = QByteArray()
        buffer = QBuffer(payload)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not bitmap.save(buffer, "PNG"):
            raise ValueError("PNG rendering failed")
        frames.append((size, bytes(payload)))
    offset = 6 + 16 * len(frames)
    directory = []
    for size, payload in frames:
        directory.append(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(payload), offset))
        offset += len(payload)
    (folder / "dockdack.ico").write_bytes(struct.pack("<HHH", 0, 1, len(frames)) + b"".join(directory) + b"".join(p for _, p in frames))
    print("Generated dockdack/assets/dockdack.ico: 9 sizes, 16-256 px")


if __name__ == "__main__":
    main()
