from IPython.display import HTML
from numpy import fft
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import hamming

# Animation utility functions
def animate_bump(x, yframes, xlabel="", ylabel=""):
    """Animate the bump over time.

    Args:
    x: x values to plot over, length num_samples.
    yframes: y values, shape (num_frames, num_samples).
    xlabel: label for x axis.
    ylabel: label for y axis.
    """

    fig, ax = plt.subplots()
    max = np.max(yframes)+1
    ax.set_ylim(0, max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    markerline, stemlines, baseline = ax.stem(yframes[0])

    def _updatefig(frame):
        ax.cla()
        ax.set_ylim(0, max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        markerline, stemlines, baseline = ax.stem(yframes[frame])
        ax.set_title("step {}".format(frame))
        return (markerline, stemlines, baseline)

    anim = animation.FuncAnimation(
        fig, _updatefig, interval=30, frames=len(yframes), blit=True)
    html = HTML(anim.to_jshtml())
    display(html)
    plt.close()