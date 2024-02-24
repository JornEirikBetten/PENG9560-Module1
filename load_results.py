import numpy as np 
import os 
import pandas as pd 
import src

algo = "dqn"
environment = "seaquest/"

log_path = os.getcwd() + "/training_runs/" + environment + algo + "/"
fig_path = os.getcwd() + "/figures/" + environment + algo + "/"
policy_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
selected_policy_names = ["delta", "epsilon"]
policy_labels = [r"$\pi_\alpha$", r"$\pi_\beta$", r"$\pi_\gamma$", r"$\pi_\delta$", r"$\pi_\epsilon$"]#, r"$\pi_\zeta$"]
selected_policy_labels = [r"$\pi_\epsilon$", r"$\pi_\zeta$"]
policy_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]# "tab:brown"]
selected_policy_colors = ["tab:red", "tab:purple"]
avgs_all = []; stds_all = []; xs = []
n = 100
xticks = [[i*1_000_000 for i in range(11)], ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]]
for i, policy_name in enumerate(policy_names): 
    data_path = log_path + policy_name + "/monitor.csv"
    df = pd.read_csv(data_path, skiprows=[0])
    returns = df["r"]
    lengths = df["l"]
    moving_avgs, moving_stds = src.moving_average(returns, n)
    avgs_all.append(moving_avgs); stds_all.append(moving_stds)
    
    x = [np.sum(lengths[:i]) for i in range(len(lengths))]
    src.lineplot_with_stds(x, 
                           moving_avgs, 
                           moving_stds, 
                           "moving average return " + policy_labels[i], policy_colors[i], 
                           "time step", f"average return of \n last {n} episodes", 
                           fig_path + policy_name + "_training_avg_return.pdf", 
                           xticks=xticks,
                           alpha = 0.2)
    xs.append(x)

src.multiple_lineplots_with_stds(xs, 
                                 avgs_all, 
                                 stds_all, 
                                 policy_labels, 
                                 policy_colors, 
                                 "time step (in millons)", 
                                 f"average return of \n last {n} episodes", 
                                 fig_path + "all_policies_training_avg_return.pdf",
                                 xticks = xticks, 
                                 alpha = 0.4)