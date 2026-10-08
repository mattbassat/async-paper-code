"""Colours and small plotting helpers shared by several figures."""
import matplotlib.patheffects as pe

E_COL, I_COL = "#377e22", "#ea3323"      # excitatory / inhibitory populations
EXT_COL = "#8d8d8d"                      # external (Poisson) population
INK, INK2 = "#1f1f1f", "#52514e"         # text and secondary text
DRIVE_COL = "#3a6ea8"                    # afferent drive
WHITE = [pe.withStroke(linewidth=2.2, foreground="white")]   # halo for labels drawn over traces
