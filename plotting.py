import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
    plt.savefig('{}/{}_{}-vs-{}.png'.format(os.path.join(cfg.plot_dir, cfg.nn_type, cfg.model_name), cfg.model_name, y_label.lower(), x_label.lower()))
    plt.close()


def plot_hist(xs, color, epoch, batch, index, layer, x_label, y_label, cfg):
    for i in range(len(xs)):
        plt.hist(xs[i], label='layer: {}'.format(layer), color=color, alpha=ALPHA)
    format_plot(x_label, y_label, title='Histogram of {}'.format(x_label))
    plt.savefig('{}/{}_{}_epoch{}_batch{}_index{}_layer{}.png'.format(os.path.join(cfg.plot_dir, cfg.nn_type, cfg.model_name), cfg.model_name, x_label.replace(' ', '-').lower(), epoch, batch, index, layer))
    plt.close()
