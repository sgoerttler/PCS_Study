import os
import numpy as np
from scipy.stats import norm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors


def norm_cdf_mAFC(stim_x, loc, scale, m=2):
    """Calculate cumulative distribution function for mAFC task."""
    guess_rate = 1. / m
    lapse_rate = 0.05
    return guess_rate + (1 - guess_rate - lapse_rate) * norm.cdf(stim_x, loc=loc, scale=scale)


def get_files(dir):
    """Return list of files in directory."""
    return sorted([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])


def add_legend(colors, linestyles, labels, plot_axis, loc=None):
    """Add customized legend to plot."""
    custom_lines = []
    for color, linestyle in zip(colors, linestyles):
        if linestyle == 'full':
            custom_lines.append(Patch(facecolor=color))
        else:
            custom_lines.append(Line2D([0], [0], color=color, linestyle=linestyle))
    plot_axis.legend(custom_lines, labels, loc=loc)


def transparent_to_opaque(color, alpha):
    """Convert transparent color with alpha to opaque color."""
    color = np.array(colors.to_rgb(color))
    inv_color = 1 - color
    return 1 - (inv_color * alpha)


def normalize_data(X, axis=None):
    """Normalize data to have zero mean and unit variance."""
    return (X - np.mean(X, axis=axis, keepdims=True)) / np.std(X, axis=axis, keepdims=True)


def print_execution_title(title):
    """Print title of executed code block."""
    chars_before = '#' * (40 - len(title) // 2 - len(title) % 2 - 3)
    chars_after = '#' * (40 - len(title) // 2 - 3)
    print('\n' + '#' * 80 + f'\n{chars_before}   {title.upper()}   {chars_after}\n' + '#' * 80 + '\n')
