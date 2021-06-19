import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib import cm
from qutip import *
from qutip.ipynbtools import plot_animation
from matplotlib import animation
from IPython.display import HTML

def plot_setup(result):
    
    rho_cavity = ptrace(result.states[1], 1)

    fig, axes = plot_wigner(rho_cavity, colorbar=True)
    
    
    return fig, axes


def plot_result(result,n, fig=None, axes=None, trace_dim=1, time=None):
    # trace out the qubit
    rho_cavity = ptrace(result.states[n], trace_dim)

    fig, axes = plot_wigner(rho_cavity, fig=fig, ax = axes)
    if time is not None:
        axes.text(-7,-7,"time: %.6f" %time, fontsize=15)
    return fig, axes

def wigner_animate(state_object, trace_dimension=1, skip_frames=1, interval=100, tlist=None):
    ''' interval in ms'''
    fig,ax = plt.subplots()

    def animate(i):
        ax.clear()
        time = None
        if tlist is not None:
            time=tlist[i*skip_frames]
        plot_result(state_object, i*skip_frames, fig=fig, axes = ax, trace_dim=trace_dimension, time=time)

    return animation.FuncAnimation(fig,animate, frames=int(np.floor(len(state_object.times)/skip_frames)),blit=False, interval = interval)

    