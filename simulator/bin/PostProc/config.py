import numpy as np
import matplotlib.pyplot as plt


class Configuration:
    def __init__(self):
        prefix = '../'

        self.pin = prefix
        self.pout = prefix + 'Plots/'

        self.nfill = 10

        plt.rcParams['font.size'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['figure.figsize'] = (10, 5)
        # plt.rcParams['figure.dpi'] = 300

        with open(prefix + 'parameters', 'r') as file:
            for line in file:
                wlist = line.strip().split(' ')
                while '' in wlist:
                    wlist.remove('')

                if (wlist[0] == 'omax'):
                    self.omax = int(wlist[2])  #assign omax (the format in the file is: omax=number so on wlist is the element number 2)

                if (wlist[0] == 'nn'):
                    self.nn = int(wlist[2])  # assign nn (the format in the file is: nn=number so on wlist is the element number 2)

                if (wlist[0] == 'nu'):
                    self.nu = float(wlist[2])  # assign nu (the format in the file is: nu=number so on wlist is the element number 2)

        self.colors = [
            '#1f77b4',  # muted blue
            '#ff7f0e',  # safety orange
            '#2ca02c',  # cooked asparagus green
            '#d62728',  # brick red
            '#9467bd',  # muted purple
            '#8c564b',  # chestnut brown
            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow-green
            '#17becf',  # blue-teal
            '#000000',  # black
            '#a52a2a',  # brown
            '#008000',  # green
            '#800080',  # purple
            '#000080',  # navy
            '#808000',  # olive
            '#008b8b',  # darkcyan
            '#b8860b',  # darkgoldenrod
            '#00ced1',  # darkturquoise
            '#ff4500',  # orangered
            '#b22222',  # firebrick
            '#dc143c',  # crimson
            '#ff0000',  # red
            '#ff6347',  # tomato
            '#ffa500',  # orange
            '#daa520',  # goldenrod
            '#bdb76b',  # darkkhaki
            '#556b2f',  # darkolivegreen
            '#2f4f4f',  # darkslategray
            '#483d8b'   # darkslateblue
        ]

    def add_param_text(self, ax):
        param_text = f'$\\nu = {self.nu:.0e}$\n$nn = {self.nn}$'
        ax.text(0.02, 0.02, param_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                fontsize=10, verticalalignment='bottom')
