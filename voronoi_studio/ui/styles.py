"""Dark-theme Qt stylesheet used across the application.

A polished, "premium" dark palette: deep indigo-black surfaces, a violet/teal
accent system, soft rounded cards, restrained borders and hover states.  Used
by both the image-driven Voronoi Studio and the Reservoir Analysis tool so the
two share one consistent identity.
"""
from __future__ import annotations


# ---- palette -----------------------------------------------------------------
#   bg0  app background          #14151A
#   bg1  cards / panels          #1E2027
#   bg2  inputs                  #15171D
#   line subtle borders          #2C2F38
#   txt  primary text            #ECEDF2
#   sub  muted text              #9AA0AE
#   indigo accent                #6C77FF
#   teal  accent (success)       #1FC79B

DARK_STYLESHEET = """
* {
    color: #ECEDF2;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

QMainWindow, QDialog {
    background-color: #14151A;
}
QWidget#card, QWidget {
    background-color: #14151A;
}

QToolBar {
    background-color: #101116;
    border: none;
    border-bottom: 1px solid #2C2F38;
    spacing: 8px;
    padding: 6px 10px;
}
QToolBar QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 6px 12px;
    color: #D7DAE3;
    font-weight: 600;
}
QToolBar QToolButton:hover {
    background-color: #20222B;
    border-color: #2C2F38;
}
QToolBar QToolButton:pressed { background-color: #181A21; }
QToolBar::separator {
    width: 1px;
    background: #2C2F38;
    margin: 4px 6px;
}

QStatusBar {
    background-color: #101116;
    color: #9AA0AE;
    border-top: 1px solid #2C2F38;
}
QStatusBar::item { border: none; }

QGroupBox {
    background-color: #1E2027;
    border: 1px solid #2C2F38;
    border-radius: 12px;
    margin-top: 16px;
    padding: 14px 12px 12px 12px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 2px 8px;
    color: #AEB4FF;
    background-color: #1E2027;
    border: 1px solid #2C2F38;
    border-radius: 7px;
}

QLabel { background: transparent; color: #D7DAE3; }
QLabel#hint { color: #8A90A0; font-size: 12px; }
QLabel#h1 { color: #FFFFFF; font-size: 30px; font-weight: 800; }
QLabel#h2 { color: #AEB4FF; font-size: 14px; font-weight: 700; }
QLabel#metricValue { color: #FFFFFF; font-size: 18px; font-weight: 700; }
QLabel#metricKey { color: #8A90A0; font-size: 11px; }

QPushButton {
    background-color: #272A33;
    border: 1px solid #363A45;
    border-radius: 8px;
    padding: 8px 14px;
    color: #ECEDF2;
    font-weight: 600;
}
QPushButton:hover { background-color: #313542; border-color: #454A58; }
QPushButton:pressed { background-color: #1F222A; }
QPushButton:disabled { color: #5F636E; background-color: #20222A; border-color: #2A2D36; }
QPushButton:checked {
    background-color: #2A2E5C;
    border-color: #6C77FF;
    color: #FFFFFF;
}

QPushButton#primary {
    background-color: #6C77FF;
    border: 1px solid #818BFF;
    color: white;
    font-weight: 700;
}
QPushButton#primary:hover  { background-color: #7B85FF; }
QPushButton#primary:pressed { background-color: #5A65EC; }

QPushButton#accent {
    background-color: #1FC79B;
    border: 1px solid #34D9AE;
    color: #06241C;
    font-weight: 700;
}
QPushButton#accent:hover  { background-color: #2AD3A7; }
QPushButton#accent:pressed { background-color: #16A983; }

QPushButton#danger {
    background-color: #E25563;
    border: 1px solid #F26B78;
    color: white;
}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #15171D;
    border: 1px solid #2C2F38;
    border-radius: 8px;
    padding: 5px 8px;
    min-height: 22px;
    color: #ECEDF2;
    selection-background-color: #6C77FF;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #3A3E49;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #6C77FF;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #9AA0AE;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1E2027;
    border: 1px solid #2C2F38;
    border-radius: 8px;
    padding: 4px;
    selection-background-color: #6C77FF;
    selection-color: white;
    outline: none;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #21242C;
    border: none;
    width: 16px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #2C3039; }

QSlider::groove:horizontal {
    height: 6px;
    background: #2A2D36;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #6C77FF;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -6px 0;
    background: #FFFFFF;
    border-radius: 8px;
    border: 2px solid #6C77FF;
}
QSlider::handle:horizontal:hover { border-color: #818BFF; }

QCheckBox { spacing: 8px; color: #D7DAE3; }
QCheckBox::indicator {
    width: 18px; height: 18px;
    border-radius: 5px;
    border: 1px solid #3A3E49;
    background-color: #15171D;
}
QCheckBox::indicator:hover { border-color: #6C77FF; }
QCheckBox::indicator:checked {
    background-color: #6C77FF;
    border-color: #818BFF;
    image: none;
}

QScrollArea { background-color: transparent; border: none; }
QScrollBar:vertical {
    width: 11px;
    background: transparent;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #353945;
    min-height: 28px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover { background: #454A58; }
QScrollBar:horizontal {
    height: 11px;
    background: transparent;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #353945;
    min-width: 28px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover { background: #454A58; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QProgressBar {
    background-color: #15171D;
    border: 1px solid #2C2F38;
    border-radius: 8px;
    text-align: center;
    color: #ECEDF2;
    min-height: 16px;
}
QProgressBar::chunk {
    background-color: #6C77FF;
    border-radius: 7px;
}

QSplitter::handle {
    background: #14151A;
}
QSplitter::handle:horizontal { width: 6px; }
QSplitter::handle:vertical { height: 6px; }
QSplitter::handle:hover { background: #2C2F38; }

QTabWidget::pane {
    border: 1px solid #2C2F38;
    border-radius: 12px;
    top: -1px;
    background-color: #1A1C22;
}
QTabBar::tab {
    background: transparent;
    padding: 9px 20px;
    margin-right: 4px;
    border: 1px solid transparent;
    border-top-left-radius: 9px;
    border-top-right-radius: 9px;
    color: #8A90A0;
    font-weight: 600;
}
QTabBar::tab:hover { color: #D7DAE3; }
QTabBar::tab:selected {
    background: #1E2027;
    color: #FFFFFF;
    border-color: #2C2F38;
    border-bottom-color: #1E2027;
}

QTableWidget, QTableView {
    background-color: #15171D;
    alternate-background-color: #1A1C23;
    gridline-color: #262932;
    border: 1px solid #2C2F38;
    border-radius: 10px;
    selection-background-color: #2A2E5C;
    selection-color: #FFFFFF;
    color: #D7DAE3;
}
QHeaderView::section {
    background-color: #21242C;
    color: #AEB4FF;
    padding: 6px 8px;
    border: none;
    border-right: 1px solid #2C2F38;
    border-bottom: 1px solid #2C2F38;
    font-weight: 700;
}
QTableCornerButton::section { background-color: #21242C; border: none; }

QTextEdit {
    background-color: #15171D;
    border: 1px solid #2C2F38;
    border-radius: 10px;
    color: #D7DAE3;
    selection-background-color: #6C77FF;
}

QToolTip {
    background-color: #21242C;
    color: #ECEDF2;
    border: 1px solid #3A3E49;
    border-radius: 6px;
    padding: 5px 8px;
}
"""
