import numpy as np 
import os 
import pandas as pd 
import src

log_path = os.getcwd() + "/training_runs/freeway/dqn/"
fig_path = os.getcwd() + "/figures/freeway/dqn/"
policy_names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
policy_labels = [r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$", r"$\epsilon$", r"$\zeta$"]
policy_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

n = 100
for i, policy_name in enumerate(policy_names): 
    data_path = log_path + policy_name + "/monitor.csv"
    df = pd.read_csv(data_path, skiprows=[0])
    returns = df["r"]
    moving_avgs, moving_stds = src.moving_average(returns, n)


    src.lineplot_with_stds([x+1 for x in range(len(moving_avgs))], moving_avgs, moving_stds, "moving average return " + policy_labels[i], policy_colors[i], "episode", f"average return of \n last {n} episodes", fig_path + policy_name + "_training_avg_return.pdf" )