import matplotlib.pyplot as plt
import platform
import numpy as np
import seaborn as sns


# originally specified in Datta lab DA paper here
# https://github.com/dattalab/dopamine-reinforces-spontaneous-behavior/blob/main/rl_analysis/io/fig.py
# use sane settings for fonts and size
# mscorefonts needs to be installed
# sudo apt install msttcorefonts -qq
# rebuild matplotlib cache...
all_fig_dct = {
    "pdf.fonttype": 42,
    "font.sans-serif": ["Arial","Helvetica","Liberation Sans"],
    "font.family": "sans-serif",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "lines.linewidth": 1,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}

if platform.system() != "Darwin":
    all_fig_dct["ps.usedistiller"] = "xpdf"

# all in points
font_dct = {
    "axes.labelpad": 3.5,
    "font.size": 7,
    "figure.titlesize": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "xtick.major.size": 3.6,
    "ytick.major.size": 3.6,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
}

plot_config = {**all_fig_dct, **font_dct}

scarcamp_rgb_dk = np.array([165, 56, 61]) / 255.
pal = {
    "scarcamp": sns.color_palette("bright")[3],
    # "scarcamp": scarcamp_rgb,
    "jrcamp1b": sns.color_palette("bright")[1],
    "jrcamp1a": sns.color_palette("bright")[-5],
    "jrgeco1a": sns.color_palette("bright")[-4],
}


def setup_plotting_env():
    plt.style.use("default")
    sns.set_style("white", sns.axes_style("ticks") | plot_config)
    sns.set_context("paper", rc=plot_config)
    if platform.system() != "Darwin":
        plt.rcParams["ps.usedistiller"] = "xpdf"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["savefig.dpi"] = 600
