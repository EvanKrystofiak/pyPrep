"""pyPrep look: the pyFIB family dark theme (fib_dark_theme) for PySide6.

Palette, stylesheet and the larger control scale follow pyFIB's
``smartfib_executor/assets/fib_dark_theme.py`` and ``gui_qt/theme.py`` so the
tools look alike; retune the constants below and everything moves together.
Additions for widgets pyFIB does not style (tables, plots, the Start button)
use the same palette.
"""

from __future__ import annotations

# ---- palette (identical to pyFIB fib_dark_theme) ------------------------------ #
BG        = "#232833"   # window
BG_ALT    = "#1c212b"   # panels / input fields base
BG_RAISE  = "#2b313d"   # raised surfaces (buttons, headers)
BORDER    = "#3a4150"
TEXT      = "#e7ebf2"
TEXT_DIM  = "#9aa3b2"
ACCENT    = "#17c3b2"   # teal
ACCENT_Hi = "#3ad9c7"
ACCENT_FG = "#06231f"   # text drawn on top of the accent colour
DANGER    = "#e0574a"
WARNING   = "#e3b341"
OK        = "#3fb950"

CANVAS_BG   = "#12151b"
CANVAS_GRID = "#232a35"
CANVAS_AXIS = "#2f7d74"

_QSS = """
* {{ font-size: 12px; }}

QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}

QMenuBar {{ background: {BG}; color: {TEXT}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {BG_RAISE}; border-radius: 4px; }}
QMenu {{ background: {BG_RAISE}; color: {TEXT}; border: 1px solid {BORDER}; }}
QMenu::item {{ padding: 6px 22px; }}
QMenu::item:selected {{ background: {ACCENT}; color: {ACCENT_FG}; }}

QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 8px; margin-top: 10px;
    padding: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {TEXT_DIM};
}}

QLabel {{ background: transparent; }}

/* prominent action / navigation buttons: objectName("toolButton") */
QToolButton#toolButton, QPushButton#toolButton {{
    background: {BG_ALT}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 8px;
    padding: 8px 12px; text-align: left; font-size: 13px;
}}
QToolButton#toolButton:hover, QPushButton#toolButton:hover {{ border-color: {ACCENT}; }}
QToolButton#toolButton:checked, QPushButton#toolButton:checked {{
    background: {ACCENT}; color: {ACCENT_FG}; border-color: {ACCENT_Hi};
    font-weight: 700;
}}

/* generic buttons */
QToolButton, QPushButton {{
    background: {BG_ALT}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 10px;
}}
QToolButton:hover, QPushButton:hover {{ border-color: {ACCENT}; }}
QToolButton:checked, QPushButton:checked {{
    background: {ACCENT}; color: {ACCENT_FG}; border-color: {ACCENT_Hi};
}}
QPushButton:disabled, QToolButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}

QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit, QTextEdit, QAbstractSpinBox {{
    background: {BG_ALT}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 6px;
    selection-background-color: {ACCENT}; selection-color: {ACCENT_FG};
}}
QComboBox:hover, QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{
    background: {BG_RAISE}; color: {TEXT};
    border: 1px solid {BORDER}; selection-background-color: {ACCENT};
    selection-color: {ACCENT_FG};
}}

QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px; border: 1px solid {BORDER};
    border-radius: 4px; background: {BG_ALT};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT_Hi};
}}

QProgressBar {{
    border: 1px solid {BORDER}; border-radius: 6px; text-align: center;
    background: {BG_ALT}; color: {TEXT};
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

QTabBar::tab {{
    background: {BG_ALT}; color: {TEXT_DIM};
    border: 1px solid {BORDER}; padding: 6px 12px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{ background: {BG_RAISE}; color: {TEXT}; }}

QStatusBar {{ background: {BG_RAISE}; color: {TEXT_DIM}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ color: {TEXT_DIM}; }}

QScrollBar:vertical {{ background: {BG}; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar:horizontal {{ background: {BG}; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

/* ---- pyPrep additions, same palette ---- */
QWidget#topBar {{ background: {BG_RAISE}; border-bottom: 1px solid {BORDER}; }}
QWidget#navBar {{ background: {BG_ALT}; border-right: 1px solid {BORDER}; }}
QFrame#card {{ background: {BG_ALT}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#card QLabel, QFrame#card QCheckBox, QFrame#card QRadioButton {{ background: transparent; }}
QWidget#cardBody, QStackedWidget#cardBody {{ background: transparent; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QLabel#sectionTitle {{ color: {TEXT}; font-weight: 700; }}

QPushButton#startButton {{
    background: {ACCENT}; color: {ACCENT_FG}; border: 1px solid {ACCENT_Hi};
    border-radius: 8px; padding: 10px 14px; font-weight: 700;
}}
QPushButton#startButton:hover {{ background: {ACCENT_Hi}; }}
QPushButton#startButton:disabled {{ background: {BG_RAISE}; color: {TEXT_DIM}; border-color: {BORDER}; }}
QPushButton#cancelButton:enabled {{ border-color: {DANGER}; color: #ffd9d4; }}

QTableWidget, QTableView {{
    background: {BG_ALT}; alternate-background-color: {BG_RAISE}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 6px; gridline-color: {BORDER};
    selection-background-color: #1f5f59; selection-color: {TEXT};
}}
QHeaderView::section {{
    background: {BG_RAISE}; color: {TEXT_DIM}; border: none;
    border-right: 1px solid {BORDER}; border-bottom: 1px solid {BORDER}; font-weight: 600;
}}
QTableCornerButton::section {{ background: {BG_RAISE}; border: none; }}
QToolTip {{ background: {BG_RAISE}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
QSplitter::handle {{ background: {BG}; }}
"""

# pyFIB's scale_up: larger fonts and controls, appended on top of the theme.
_SCALE_QSS = """
QWidget { font-size: 15px; }
QPushButton, QToolButton, QLineEdit, QComboBox, QAbstractSpinBox {
    min-height: 30px; padding: 6px 12px;
}
QToolButton#toolButton, QPushButton#toolButton {
    padding: 11px 18px; font-size: 16px;
}
QPushButton#startButton { min-height: 34px; font-size: 16px; }
QHeaderView::section { padding: 8px 10px; font-size: 14px; }
QTableWidget, QPlainTextEdit { font-size: 14px; }
QComboBox { min-width: 120px; }
"""


def stylesheet() -> str:
    return _QSS.format(BG=BG, BG_ALT=BG_ALT, BG_RAISE=BG_RAISE, BORDER=BORDER, TEXT=TEXT,
                       TEXT_DIM=TEXT_DIM, ACCENT=ACCENT, ACCENT_Hi=ACCENT_Hi, ACCENT_FG=ACCENT_FG,
                       DANGER=DANGER)


def apply_theme(app, point_size: int = 11) -> None:
    """Fusion base + dark palette + stylesheet + pyFIB's enlarged scale."""
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(BG_ALT))
    pal.setColor(QPalette.AlternateBase, QColor(BG_RAISE))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(BG_RAISE))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ToolTipBase, QColor(BG_RAISE))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(ACCENT_FG))
    pal.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
    pal.setColor(QPalette.Link, QColor(ACCENT_Hi))
    pal.setColor(QPalette.Mid, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(TEXT_DIM))
    app.setPalette(pal)
    f = app.font()
    f.setPointSize(point_size)
    app.setFont(f)
    app.setStyleSheet(stylesheet() + _SCALE_QSS)

    import pyqtgraph as pg
    pg.setConfigOptions(background=CANVAS_BG, foreground=TEXT_DIM, antialias=True,
                        imageAxisOrder="row-major")


# ---- small builders shared by the pages (pyFIB's _title/_section/_card) ------- #
def title_label(text: str):
    from PySide6.QtWidgets import QLabel
    lab = QLabel(text)
    lab.setObjectName("pageTitle")
    f = lab.font()
    f.setPointSize(20)
    f.setBold(True)
    lab.setFont(f)
    lab.setStyleSheet("font-size: 26px; font-weight: 700;")
    return lab


def section_label(text: str):
    from PySide6.QtWidgets import QLabel
    lab = QLabel(text)
    lab.setObjectName("sectionTitle")
    lab.setStyleSheet("font-size: 18px; font-weight: 700;")
    return lab


def dim_label(text: str, wrap: bool = True):
    from PySide6.QtWidgets import QLabel
    lab = QLabel(text)
    lab.setObjectName("dim")
    lab.setWordWrap(wrap)
    return lab


def card():
    from PySide6.QtWidgets import QFrame
    fr = QFrame()
    fr.setObjectName("card")
    fr.setFrameShape(QFrame.StyledPanel)
    return fr
