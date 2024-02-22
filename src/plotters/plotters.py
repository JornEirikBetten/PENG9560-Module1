import matplotlib.pyplot as plt 
import numpy as np 


def moving_average(returns, n): 
    moving_averages = np.zeros(len(returns)) 
    moving_stds = np.zeros(len(returns))
    for i in range(len(returns)): 
        if i >= n: 
            moving_averages[i] = np.mean(returns[i-n:i])
            moving_stds[i] = np.std(returns[i-n: i])
        else: 
            moving_averages[i] = np.mean(returns[0:i])
            moving_stds[i] = np.std(returns[0:i])
            
    return moving_averages, moving_stds




def lineplot_with_stds(x, y, y_stds, label, color, xlabel, ylabel, figname):
    fig = plt.figure() 
    ax = plt.gca() 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 
    plt.ylim(0, np.max(y[100:-1]) + np.max(y_stds[100:-1]))
    plt.plot(x, y, color = color, linestyle = "solid", label = label, alpha = 1.0)
    ax.fill_between(x, y - 1.96 * y_stds, y + 1.96 * y_stds, color = color, alpha = 0.5)
    plt.legend() 
    ax.grid(True)
    plt.savefig(figname, format = "pdf", bbox_inches = "tight")
    plt.close() 
    

def multiple_lineplots_with_stds(xs, ys, y_stds, labels, colors, xlabel, ylabel, figname, xticks = None):
    fig = plt.figure() 
    ax = plt.gca() 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 
    plt.ylim(0, np.max(ys[0][100:-1]) + np.max(y_stds[0][100:-1]))
    for i, y in enumerate(ys): 
        plt.plot(xs[i], y, color = colors[i], linestyle = "solid", label = labels[i], alpha = 0.5)
        ax.fill_between(xs[i], y - 1.96 * y_stds[i], y + 1.96 * y_stds[i], color = colors[i], alpha = 0.0)
    if xticks != None: 
        plt.xticks(xticks[0], xticks[1])
    plt.legend() 
    ax.grid(True)
    plt.savefig(figname, format = "pdf", bbox_inches = "tight")
    plt.close() 
    
    