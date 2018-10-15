""" Plots the development of the parameters of an autoencoder during
evolution

Author: Tim Silhan
"""

import sys
from os.path import dirname, realpath
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
sys.path.append(dirname(dirname(realpath(__file__))))
from utils.history_saver import HistorySaver

def main():
    """ Main function of the history plotter """

    # Adjust layout of the plot so labels are not cut off
    rcParams.update({'figure.autolayout': True})

    hist_file = ""

    if len(sys.argv) > 1:
        hist_file = sys.argv[1]

    hs = HistorySaver.deserialize(hist_file)

    # plot_two_values_together(hs.learning_rates, "Learning Rate",
    #                          hs.momentums, "Momentum", False)
    # plot_two_values_separate(hs.learning_rates, "Learning Rate",
    #                          hs.momentums, "Momentum")
    plot_values(hs.learning_rates, "Learning Rate")


def plot_values(values, label):
    """ Plots a list of values

    Args:
        values: The y-values to plot
        label: Label for the values
    """
    _, axis = plt.subplots(1, 1, figsize=(8,4))
    axis.xaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.xlabel("Generation")
    plt.ylabel(label)

    axis.plot(values, 'o')
    axis.plot(values, '-')

    saver = plt.gcf()
    plt.show()
    save_plot(saver)

def plot_two_values_separate(vals1, label1, vals2, label2):
    """ Plots to lists of values in separate diagrams

    Args:
        vals: The y-values to plot
        label: Label for the values
    """
    # Digits in the parameter are rows, cols, fig_num
    axis = plt.subplot(211)
    axis.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xlabel("Generation")
    plt.ylabel(label1)
    plt.plot(vals1, 'o')
    plt.plot(vals1, '-')

    axis = plt.subplot(212)
    axis.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xlabel("Generation")
    plt.ylabel(label2)
    plt.plot(vals2, 'o')
    plt.plot(vals2, '-')

    saver = plt.gcf()
    plt.show()
    save_plot(saver)

def plot_two_values_together(vals1, label1, vals2, label2, show_exact_value=False):
    """ Plots to lists of values in the same diagram

    Args:
        vals: The y-values to plot
        label: Label for the values
        show_exact_value: Show the exact values as labels on the points
    """
    fig, axis = plt.subplots(1, 1)
    axis.xaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.xlabel("Generation")
    plt.ylabel("{} / {}".format(label1, label2))

    axis.plot(vals1, 'o')
    axis.plot(vals1, '-', label=label1)
    axis.plot(vals2, 'o')
    axis.plot(vals2, '-', label=label2)

    if show_exact_value:
        for i, j in zip(range(len(vals1)), vals1):
            axis.annotate("{:.4f}".format(j), xy=(i, j),
                          xytext=(-10, 10), textcoords='offset points')

        for i, j in zip(range(len(vals2)), vals2):
            axis.annotate("{:.4f}".format(j), xy=(i, j),
                          xytext=(-10, 10), textcoords='offset points')

    plt.legend()
    saver = plt.gcf()
    plt.show()
    save_plot(saver)

def save_plot(fig):
    """ Saves the plot to the the hard drive """
    save = input("Do you want to save the plot? (y/n): ")
    if save == "y":
        # save_path = sys.argv[1].split(".")[0] + ".png"
        save_path = "/Users/tim/Downloads/ypmsdlr.png"
        print("Saving plot to ", save_path)
        fig.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":
    main()
