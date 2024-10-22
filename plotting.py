import os

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import torchvision.utils as tvutils

import utils

FONTSIZE = 15
ALPHA = 0.5


def format_plot(x_label, y_label, title, fontsize=None):
    if not fontsize:
        fontsize = FONTSIZE
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()


def plot_line(x, ys, contours, labels, x_label, y_label, cfg):
    for i in range(len(ys)):
        if contours:
            plt.errorbar(x, ys[i], yerr=contours[i], label=labels[i], alpha=ALPHA)
        else:
            plt.plot(x, ys[i], label=labels[i], alpha=ALPHA)
    format_plot(x_label, y_label, title='{} vs {}'.format(y_label, x_label))
    plt.savefig('{}/{}_{}-vs-{}'.format(os.path.join(cfg.plot_dir, cfg.model_type, cfg.model_name), cfg.model_name, y_label.lower(), x_label.lower()))
    plt.close()


def plot_hist(xs, colors, epoch, mb, index, layer, x_label, y_label, cfg, embeds=None):
    for i in range(len(xs)):
        fig, ax = plt.subplots()
        ax.hist(xs[i], label='layer: {}'.format(layer), color=colors[i], alpha=ALPHA)
        if embeds is not None:
            imagebox = OffsetImage(embeds[i], zoom=1.0, cmap='gray')
            ab = AnnotationBbox(imagebox, (0.7, 0.7), xycoords='figure fraction')
            ax.add_artist(ab)
    format_plot(x_label, y_label, title='Histogram of {}'.format(x_label))
    plt.savefig('{}/{}_{}_epoch{}_mb{}_index{}_layer{}'.format(os.path.join(cfg.plot_dir, cfg.model_type, cfg.model_name), cfg.model_name, x_label.replace(' ', '-').lower(), epoch, mb, index, layer))
    plt.close()


def make_grid(x, filepath):
    np_img = utils.tensor2array(tvutils.make_grid(x))
    plt.imshow(np.transpose(np_img, (1,2,0)), interpolation='nearest')
    plt.savefig(filepath)
    plt.close()
