
import os
import QPhantom
import matplotlib
from QPhantom.core.metrics.Metrics import Metrics

zh_font = matplotlib.font_manager.FontProperties(fname=os.path.join(QPhantom.__path__[0], 'resource', 'msyh.ttf'))

__all__ = [Metrics, zh_font]
