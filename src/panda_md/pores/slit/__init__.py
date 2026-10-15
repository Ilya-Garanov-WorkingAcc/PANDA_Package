"""Щелевая пора и её формы. Импорт модулей форм регистрирует их в SlitPore."""

from panda_md.pores.slit import doughnut, droplet, perforation, roll
from panda_md.pores.slit.pore import SlitGeometry, SlitPore

__all__ = ["SlitGeometry", "SlitPore", "doughnut", "droplet", "perforation", "roll"]
