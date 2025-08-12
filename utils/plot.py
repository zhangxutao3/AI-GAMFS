import pandas as pd
import numpy as np
import cmaps

from matplotlib import pyplot as plt

def add_aerosol_ticks(ax, var, metric):
    if var == "BCEXTTAU":
        if metric == "rmse":
            ax.set_ylim(0, 0.0065)
            ax.set_yticks([0, 0.0025, 0.005, 0.0075])
        elif metric == "acc":
            ax.set_ylim(0.4, 1)
            ax.set_yticks([0.4, 0.6, 0.8, 1])
        elif metric == "r":
            ax.set_ylim(0.7, 1)
            ax.set_yticks([0.7, 0.8, 0.9, 1])
    elif var == "BCSMASS":
        if metric == "rmse":
            ax.set_ylim(0.1, 0.5)
            ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
        elif metric == "acc":
            ax.set_ylim(0.2, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
        elif metric == "r":
            ax.set_ylim(0.6, 1)
            ax.set_yticks([0.6, 0.8, 1])
    elif var == "DUEXTTAU":
        if metric == "rmse":
            ax.set_ylim(0, 0.055)
            ax.set_yticks([0, 0.015, 0.03, 0.045, 0.06])
        elif metric == "acc":
            ax.set_ylim(0.5, 1)
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
        elif metric == "r":
            ax.set_ylim(0.7, 1)
            ax.set_yticks([0.7, 0.8, 0.9, 1])
    elif var == "DUSMASS":
        if metric == "rmse":
            ax.set_ylim(10, 80)
            ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80])
        elif metric == "acc":
            ax.set_ylim(0.3, 1)
            ax.set_yticks([0.3, 0.65, 1])
        elif metric == "r":
            ax.set_ylim(0.6, 1)
            ax.set_yticks([0.6, 0.8, 1])
    elif var == "OCEXTTAU":
        if metric == "rmse":
            ax.set_ylim(0, 0.045)
            ax.set_yticks([0, 0.015, 0.03, 0.045])
        elif metric == "acc":
            ax.set_ylim(0.4, 1)
            ax.set_yticks([0.4, 0.6, 0.8, 1])
        elif metric == "r":
            ax.set_ylim(0.5, 1)
            ax.set_yticks([0.5, 0.75, 1])
    elif var == "OCSMASS":
        if metric == "rmse":
            ax.set_ylim(2, 6)
            ax.set_yticks([2, 4, 6])
        elif metric == "acc":
            ax.set_ylim(0.3, 1)
            ax.set_yticks([0.3, 0.65, 1])
        elif metric == "r":
            ax.set_ylim(0.4, 1)
            ax.set_yticks([0.4, 0.6, 0.8, 1])
    elif var == "SUEXTTAU":
        if metric == "rmse":
            ax.set_ylim(0, 0.05)
            ax.set_yticks([0, 0.025, 0.05])
        elif metric == "acc":
            ax.set_ylim(0.3, 1)
            ax.set_yticks([0.3, 0.65, 1])
        elif metric == "r":
            ax.set_ylim(0.5, 1)
            ax.set_yticks([0.5, 0.75, 1])
    elif var == "SO4SMASS":
        if metric == "rmse":
            ax.set_ylim(0.25, 1.75)
            ax.set_yticks([0.25, 0.75, 1.25, 1.75])
        elif metric == "acc":
            ax.set_ylim(0.3, 1)
            ax.set_yticks([0.3, 0.65, 1])
        elif metric == "r":
            ax.set_ylim(0.6, 1)
            ax.set_yticks([0.6, 0.8, 1])
    elif var == "SSEXTTAU":
        if metric == "rmse":
            ax.set_ylim(0.01, 0.06)
            ax.set_yticks([0.01, 0.035, 0.06])
        elif metric == "acc":
            ax.set_ylim(0.25, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1])
        elif metric == "r":
            ax.set_ylim(0.5, 1)
            ax.set_yticks([0.5, 0.75, 1])
    elif var == "SSSMASS":
        if metric == "rmse":
            ax.set_ylim(5, 45)
            ax.set_yticks([5, 25, 45])
        elif metric == "acc":
            ax.set_ylim(0.25, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1])
        elif metric == "r":
            ax.set_ylim(0.5, 1)
            ax.set_yticks([0.5, 0.75, 1])
    elif var == "TOTEXTTAU":
        if metric == "rmse":
            ax.set_ylim(0.01, 0.13)
            ax.set_yticks([0.01, 0.05, 0.09, 0.13])
        elif metric == "acc":
            ax.set_ylim(0.4, 1)
            ax.set_yticks([0.4, 0.6, 0.8, 1])
        elif metric == "r":
            ax.set_ylim(0.5, 1)
            ax.set_yticks([0.5, 0.75, 1])
    elif var == "TOTSCATAU":
        if metric == "rmse":
            ax.set_ylim(0.01, 0.13)
            ax.set_yticks([0.01, 0.05, 0.09, 0.13])
        elif metric == "acc":
            ax.set_ylim(0.4, 1)
            ax.set_yticks([0.4, 0.6, 0.8, 1])
        elif metric == "r":
            ax.set_ylim(0.5, 1)
            ax.set_yticks([0.5, 0.75, 1])


def add_level_ticks(ax, var_name, level, metric="rmse", fontsize=22):

    if metric == "rmse":
        if var_name == "QV":
            if level == "45":
                ax.set_yticks([y * 1e3 for y in [0, 0.0001, 0.0002, 0.0003]])
                ax.set_ylim(0, 0.0003* 1e3)
            elif level == "48":
                ax.set_yticks([y * 1e3 for y in [0, 0.0005, 0.001]])
                ax.set_ylim(0, 0.001* 1e3)
            elif level == "51":
                ax.set_yticks([y * 1e3 for y in [0, 0.0005, 0.001, 0.0015]])
                ax.set_ylim(0, 0.0015* 1e3)
            elif level == "53":
                ax.set_yticks([y * 1e3 for y in [0, 0.001, 0.002]])
                ax.set_ylim(0, 0.002* 1e3)
            elif level == "56":
                ax.set_yticks([y * 1e3 for y in [0, 0.001, 0.002, 0.003]])
                ax.set_ylim(0, 0.003* 1e3)
            elif level == "60":
                ax.set_yticks([y * 1e3 for y in [0, 0.001, 0.002, 0.003]])
                ax.set_ylim(0, 0.003* 1e3)
            elif level == "63":
                ax.set_yticks([y * 1e3 for y in [0, 0.001, 0.002, 0.003]])
                ax.set_ylim(0, 0.003* 1e3)
            elif level == "68":
                ax.set_yticks([y * 1e3 for y in [0, 0.001, 0.002]])
                ax.set_ylim(0, 0.002* 1e3)
            elif level == "72":
                ax.set_yticks([y * 1e3 for y in [0, 0.001, 0.002]])
                ax.set_ylim(0, 0.002* 1e3)

        elif var_name == "T":
            if level == "45":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "48":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "51":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "53":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "56":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "60":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "63":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "68":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "72":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)

        elif var_name == "U":
            if level == "45":
                ax.set_yticks([0, 5, 10, 15])
                ax.set_ylim(0, 15)
            elif level == "48":
                ax.set_yticks([0, 6, 12])
                ax.set_ylim(0, 12)
            elif level == "51":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "53":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "56":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "60":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "63":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "68":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "72":
                ax.set_yticks([0, 3, 6])
                ax.set_ylim(0, 6)

        elif var_name == "V":
            if level == "45":
                ax.set_yticks([0, 5, 10, 15])
                ax.set_ylim(0, 15)
            elif level == "48":
                ax.set_yticks([0, 6, 12])
                ax.set_ylim(0, 12)
            elif level == "51":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "53":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "56":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "60":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "63":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "68":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "72":
                ax.set_yticks([0, 3, 6])
                ax.set_ylim(0, 6)

    elif metric == "acc":
        if var_name == "QV":
            if level == "45":
                ax.set_yticks([0.25, 0.5, 0.75, 1])
                ax.set_ylim(0.25, 1)
            elif level == "48":
                ax.set_yticks([0.25, 0.5, 0.75, 1])
                ax.set_ylim(0.25, 1)
            elif level == "51":
                ax.set_yticks([0.25, 0.5, 0.75, 1])
                ax.set_ylim(0.25, 1)
            elif level == "53":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "56":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "60":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "63":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "68":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "72":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)

        elif var_name == "T":
            if level == "45":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "48":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "51":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "53":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "56":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "60":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "63":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "68":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "72":
                ax.set_yticks([0.8, 0.9, 1])
                ax.set_ylim(0.8, 1)

        elif var_name == "U":
            if level == "45":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "48":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "51":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "53":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "56":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "60":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "63":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "68":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "72":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)

        elif var_name == "V":
            if level == "45":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "48":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "51":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "53":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "56":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "60":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "63":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "68":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "72":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)

    elif metric == "r":
        if var_name == "QV":
            if level == "45":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "48":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "51":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "53":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "56":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "60":
                ax.set_yticks([0.8, 0.9, 1])
                ax.set_ylim(0.8, 1)
            elif level == "63":
                ax.set_yticks([0.85, 1])
                ax.set_ylim(0.85, 1)
            elif level == "68":
                ax.set_yticks([0.94, 0.96, 0.98, 1])
                ax.set_ylim(0.94, 1)
            elif level == "72":
                ax.set_yticks([0.94, 0.96, 0.98, 1])
                ax.set_ylim(0.94, 1)

        elif var_name == "T":
            if level == "45":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "48":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "51":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "53":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "56":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "60":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "63":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "68":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "72":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)

        elif var_name == "U":
            if level == "45":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "48":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "51":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "53":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "56":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "60":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "63":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "68":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "72":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)

        elif var_name == "V":
            if level == "45":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "48":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "51":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "53":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "56":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "60":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "63":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "68":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "72":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)


def add_surface_ticks(ax, var_name, metric):

    if metric == "rmse":
        if var_name == "QLML":
            ax.set_yticks([y * 1e3 for y in [0, 0.0005, 0.001, 0.0015, 0.002]])
            ax.set_ylim(0, 0.002* 1e3)
        elif var_name == "TLML":
            ax.set_yticks([0, 2, 4])
            ax.set_ylim(0, 4)
        elif var_name == "ULML":
            ax.set_yticks([0, 3, 6])
            ax.set_ylim(0, 6)
        elif var_name == "VLML":
            ax.set_yticks([0, 3, 6])
            ax.set_ylim(0, 6)
        elif var_name == "PRECTOT":
            ax.set_yticks([y * 1e3 for y in [0.00005, 0.0001, 0.00015, 0.0002]])
            ax.set_ylim(0.00005* 1e3, 0.0002* 1e3)
        elif var_name == "SLP":
            ax.set_yticks([0, 250, 500, 750])
            ax.set_ylim(0, 750)

    elif metric == "acc":
        if var_name == "QLML":
            ax.set_yticks([0.7, 0.8, 0.9, 1])
            ax.set_ylim(0.7, 1)
        elif var_name == "TLML":
            ax.set_yticks([0.8, 0.9, 1])
            ax.set_ylim(0.8, 1)
        elif var_name == "ULML":
            ax.set_yticks([0.4, 0.6, 0.8, 1])
            ax.set_ylim(0.4, 1)
        elif var_name == "VLML":
            ax.set_yticks([0.4, 0.6, 0.8, 1])
            ax.set_ylim(0.4, 1)
        elif var_name == "PRECTOT":
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_ylim(0, 1)
        elif var_name == "SLP":
            ax.set_yticks([0.6, 0.8, 0.9, 1])
            ax.set_ylim(0.6, 1)

    elif metric == "r":
        if var_name == "QLML":
            ax.set_yticks([0.94, 0.96, 0.98, 1])
            ax.set_ylim(0.94, 1)
        elif var_name == "TLML":
            ax.set_yticks([0.96, 0.98, 1])
            ax.set_ylim(0.96, 1)
        elif var_name == "ULML":
            ax.set_yticks([0.6, 0.8, 1])
            ax.set_ylim(0.6, 1)
        elif var_name == "VLML":
            ax.set_yticks([0.5, 0.75, 1])
            ax.set_ylim(0.5, 1)
        elif var_name == "PRECTOT":
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_ylim(0, 1)
        elif var_name == "SLP":
            ax.set_yticks([0.85, 0.9, 0.95, 1])
            ax.set_ylim(0.85, 1)


def get_var_levels(var, diff=False):
    # 定义变量配置字典
    var_configs = {
        "BCEXTTAU": {
            "tag": "BCEXTTAU",
            "levs": np.arange(0, 0.051, 0.001),
            "cmap": cmaps.MPL_gist_gray_r,
            "diff_levs": np.arange(-0.04, 0.045, 0.005),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "OCEXTTAU": {
            "tag": "OCEXTTAU",
            "levs": np.arange(0, 0.2, 0.01),
            "cmap": cmaps.MPL_Purples,
            "diff_levs": np.arange(-0.2, 0.22, 0.02),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "SSEXTTAU": {
            "tag": "SSEXTTAU",
            "levs": np.arange(0, 0.3, 0.01),
            "cmap": cmaps.MPL_Blues,
            "diff_levs": np.arange(-0.15, 0.16, 0.01),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "SUEXTTAU": {
            "tag": "SUEXTTAU",
            "levs": np.arange(0, 0.51, 0.01),
            "cmap": cmaps.MPL_Reds,
            "diff_levs": np.arange(-0.255, 0.256, 0.01),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "DUEXTTAU": {
            "tag": "DUEXTTAU",
            "levs": np.arange(0, 0.6, 0.02),
            "cmap": cmaps.GMT_hot_r,
            "diff_levs": np.arange(-0.3, 0.32, 0.02),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "TOTEXTTAU": {
            "tag": "TOTEXTTAU",
            "levs": np.arange(0, 1.02, 0.02),
            "cmap": cmaps.MPL_Oranges,
            "diff_levs": np.arange(-0.51, 0.52, 0.02),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "TOTSCATAU": {
            "tag": "",
            "levs": np.arange(0, 1.02, 0.02),
            "cmap": cmaps.MPL_Oranges,
            "diff_levs": np.arange(-0.51, 0.52, 0.02),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "DUSMASS": {
            "tag": "$[unit: \mu g \cdot m^{-3}]$",
            "levs": np.arange(0, 505, 5),
            "cmap": cmaps.GMT_hot_r,
            "diff_levs": np.arange(-200, 210, 10),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "BCSMASS": {
            "tag": "$[unit: \mu g \cdot m^{-3}]$",
            "levs": np.arange(0, 41, 0.01),
            "cmap": cmaps.MPL_gist_gray_r,
            "diff_levs": np.arange(-2, 2.01, 0.01),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "OCSMASS": {
            "tag": "$[unit: \mu g \cdot m^{-3}]$",
            "levs": np.arange(0, 20.1, 0.1),
            "cmap": cmaps.MPL_Purples,
            "diff_levs": np.arange(-10, 10.1, 0.1),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "SSSMASS": {
            "tag": "$[unit: \mu g \cdot m^{-3}]$",
            "levs": np.arange(0, 121, 1),
            "cmap": cmaps.MPL_Blues,
            "diff_levs": np.arange(-90, 95, 5),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "SO4SMASS": {
            "tag": "$[unit: \mu g \cdot m^{-3}]$",
            "levs": np.arange(0, 15.1, 0.1),
            "cmap": cmaps.MPL_Reds,
            "diff_levs": np.arange(-5, 5.1, 0.1),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "QLML": {
            "tag": "$[unit: g \cdot kg^{-1}]$",
            "levs": np.arange(0, 31, 1),
            "cmap": cmaps.WhiteBlue,
            "diff_levs": np.arange(-5, 5.2, 0.2),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "TLML": {
            "tag": "$[unit: K]$",
            "levs": np.arange(200, 355, 5),
            "cmap": cmaps.MPL_YlOrRd,
            "diff_levs": np.arange(-10, 11, 1),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "ULML": {
            "tag": "$[unit: m \cdot s^{-1}]$",
            "levs": np.arange(-25, 26, 1),
            "cmap": cmaps.MPL_RdBu_r,
            "diff_levs": np.arange(-12, 13, 1),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "VLML": {
            "tag": "$[unit: m \cdot s^{-1}]$",
            "levs": np.arange(-25, 26, 1),
            "cmap": cmaps.MPL_RdBu_r,
            "diff_levs": np.arange(-12, 13, 1),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "PRECTOT": {
            "tag": "$[unit: g \cdot m^{-2} \cdot s^{-1}]$",
            "levs": np.arange(0, 0.31, 0.01),
            "cmap": cmaps.precip2_17lev,
            "diff_levs": np.arange(-0.5, 0.55, 0.05),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "SLP": {
            "tag": "$[unit: Pa]$",
            "levs": np.arange(95000, 105200, 250),
            "cmap": cmaps.WhiteBlueGreenYellowRed,
            "diff_levs": np.arange(-2000, 2200, 200),
            "diff_cmap": cmaps.MPL_RdBu_r
        },
        "AAOD":{
            "tag": "",
            "levs": np.arange(0, 0.082, 0.002),
            "cmap": cmaps.MPL_Purples,
            "diff_levs": np.arange(-0.04, 0.042, 0.002),
            "diff_cmap": cmaps.MPL_RdBu_r
        }
    }

    config = var_configs.get(var)

    if config and diff:
        config['tag'] = config['tag']
        config['levs'] = config['diff_levs']
        config['cmap'] = config['diff_cmap']

    return config



def add_level_ticks_V2(ax, var_name, level, metric="rmse", fontsize=22):

    if metric == "rmse":
        if var_name == "QV":
            if level == "45":
                ax.set_yticks([y * 1e3 for y in [0.0001, 0.00015, 0.0002, 0.00025]])
                ax.set_ylim(0.0001*1e3, 0.00025* 1e3)
            elif level == "48":
                ax.set_yticks([y * 1e3 for y in [0, 0.0005, 0.001]])
                ax.set_ylim(0, 0.001* 1e3)
            elif level == "51":
                ax.set_yticks([y * 1e3 for y in [0, 0.0005, 0.001, 0.0015]])
                ax.set_ylim(0, 0.0015* 1e3)
            elif level == "53":
                ax.set_yticks([y * 1e3 for y in [0.0005, 0.001, 0.0015, 0.002]])
                ax.set_ylim(0.0005*1e3, 0.002* 1e3)
            elif level == "56":
                ax.set_yticks([y * 1e3 for y in [0, 0.001, 0.002, 0.003]])
                ax.set_ylim(0, 0.003* 1e3)
            elif level == "60":
                ax.set_yticks([y * 1e3 for y in [0, 0.001, 0.002, 0.003]])
                ax.set_ylim(0, 0.003* 1e3)
            elif level == "63":
                ax.set_yticks([y * 1e3 for y in [0.001, 0.0015, 0.002, 0.0025]])
                ax.set_ylim(0.001*1e3, 0.0025* 1e3)
            elif level == "68":
                ax.set_yticks([y * 1e3 for y in [0, 0.001, 0.002]])
                ax.set_ylim(0, 0.002* 1e3)
            elif level == "72":
                ax.set_yticks([y * 1e3 for y in [0, 0.001, 0.002]])
                ax.set_ylim(0, 0.002* 1e3)

        elif var_name == "T":
            if level == "45":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "48":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "51":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "53":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "56":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "60":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "63":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "68":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)
            elif level == "72":
                ax.set_yticks([0, 2, 4])
                ax.set_ylim(0, 4)

        elif var_name == "U":
            if level == "45":
                ax.set_yticks([0, 5, 10, 15])
                ax.set_ylim(0, 15)
            elif level == "48":
                ax.set_yticks([0, 6, 12])
                ax.set_ylim(0, 12)
            elif level == "51":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "53":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "56":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "60":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "63":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "68":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "72":
                ax.set_yticks([0, 3, 6])
                ax.set_ylim(0, 6)

        elif var_name == "V":
            if level == "45":
                ax.set_yticks([0, 5, 10, 15])
                ax.set_ylim(0, 15)
            elif level == "48":
                ax.set_yticks([0, 6, 12])
                ax.set_ylim(0, 12)
            elif level == "51":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "53":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "56":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "60":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "63":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "68":
                ax.set_yticks([0, 4, 8])
                ax.set_ylim(0, 8)
            elif level == "72":
                ax.set_yticks([0, 3, 6])
                ax.set_ylim(0, 6)

    elif metric == "acc":
        if var_name == "QV":
            if level == "45":
                ax.set_yticks([0.25, 0.5, 0.75, 1])
                ax.set_ylim(0.25, 1)
            elif level == "48":
                ax.set_yticks([0.25, 0.5, 0.75, 1])
                ax.set_ylim(0.25, 1)
            elif level == "51":
                ax.set_yticks([0.25, 0.5, 0.75, 1])
                ax.set_ylim(0.25, 1)
            elif level == "53":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "56":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "60":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "63":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "68":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "72":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)

        elif var_name == "T":
            if level == "45":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "48":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "51":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "53":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "56":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "60":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "63":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "68":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "72":
                ax.set_yticks([0.8, 0.9, 1])
                ax.set_ylim(0.8, 1)

        elif var_name == "U":
            if level == "45":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "48":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "51":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "53":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "56":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "60":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "63":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "68":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)
            elif level == "72":
                ax.set_yticks([0.5, 0.75, 1])
                ax.set_ylim(0.5, 1)

        elif var_name == "V":
            if level == "45":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "48":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "51":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "53":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "56":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "60":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "63":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "68":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "72":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)

    elif metric == "r":
        if var_name == "QV":
            if level == "45":
                ax.set_yticks([0.8, 0.9, 1])
                ax.set_ylim(0.8, 1)
            elif level == "48":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "51":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "53":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "56":
                ax.set_yticks([0.7, 0.8, 0.9, 1])
                ax.set_ylim(0.7, 1)
            elif level == "60":
                ax.set_yticks([0.8, 0.9, 1])
                ax.set_ylim(0.8, 1)
            elif level == "63":
                ax.set_yticks([0.85, 1])
                ax.set_ylim(0.85, 1)
            elif level == "68":
                ax.set_yticks([0.94, 0.96, 0.98, 1])
                ax.set_ylim(0.94, 1)
            elif level == "72":
                ax.set_yticks([0.94, 0.96, 0.98, 1])
                ax.set_ylim(0.94, 1)

        elif var_name == "T":
            if level == "45":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "48":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "51":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "53":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "56":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "60":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "63":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "68":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)
            elif level == "72":
                ax.set_yticks([0.96, 0.98, 1])
                ax.set_ylim(0.96, 1)

        elif var_name == "U":
            if level == "45":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "48":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "51":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "53":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "56":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "60":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "63":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "68":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)
            elif level == "72":
                ax.set_yticks([0.6, 0.8, 1])
                ax.set_ylim(0.6, 1)

        elif var_name == "V":
            if level == "45":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "48":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "51":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "53":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "56":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "60":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "63":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "68":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)
            elif level == "72":
                ax.set_yticks([0.4, 0.6, 0.8, 1])
                ax.set_ylim(0.4, 1)


def add_surface_ticks_V2(ax, var_name, metric):

    if metric == "rmse":
        if var_name == "QLML":
            ax.set_yticks([y * 1e3 for y in [0.0005, 0.001, 0.0015, 0.002]])
            ax.set_ylim(0.5, 0.002* 1e3)
        elif var_name == "TLML":
            ax.set_yticks([1, 2, 3])
            ax.set_ylim(1, 3)
        elif var_name == "ULML":
            ax.set_yticks([0, 3, 6])
            ax.set_ylim(0, 6)
        elif var_name == "VLML":
            ax.set_yticks([0, 3, 6])
            ax.set_ylim(0, 6)
        elif var_name == "PRECTOT":
            ax.set_yticks([y * 1e3 for y in [0.0001, 0.0002, 0.0003]])
            ax.set_ylim(0.0001* 1e3, 0.0003* 1e3)
        elif var_name == "SLP":
            ax.set_yticks([0, 250, 500, 750])
            ax.set_ylim(0, 750)

    elif metric == "acc":
        if var_name == "QLML":
            ax.set_yticks([0.7, 0.8, 0.9, 1])
            ax.set_ylim(0.7, 1)
        elif var_name == "TLML":
            ax.set_yticks([0.8, 0.9, 1])
            ax.set_ylim(0.8, 1)
        elif var_name == "ULML":
            ax.set_yticks([0.4, 0.6, 0.8, 1])
            ax.set_ylim(0.4, 1)
        elif var_name == "VLML":
            ax.set_yticks([0.4, 0.6, 0.8, 1])
            ax.set_ylim(0.4, 1)
        elif var_name == "PRECTOT":
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_ylim(0, 1)
        elif var_name == "SLP":
            ax.set_yticks([0.6, 0.8, 0.9, 1])
            ax.set_ylim(0.6, 1)

    elif metric == "r":
        if var_name == "QLML":
            ax.set_yticks([0.96, 0.98, 1])
            ax.set_ylim(0.96, 1)
        elif var_name == "TLML":
            ax.set_yticks([0.98, 0.99, 1])
            ax.set_ylim(0.98, 1)
        elif var_name == "ULML":
            ax.set_yticks([0.6, 0.8, 1])
            ax.set_ylim(0.6, 1)
        elif var_name == "VLML":
            ax.set_yticks([0.5, 0.75, 1])
            ax.set_ylim(0.5, 1)
        elif var_name == "PRECTOT":
            ax.set_yticks([0, 0.25, 0.5])
            ax.set_ylim(0, 0.5)
        elif var_name == "SLP":
            ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1])
            ax.set_ylim(0.8, 1)




