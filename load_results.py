import numpy as np 
import os 
import pandas as pd 
import src

algo = "dqn"
environment = "freeway/"


def seaquest_plotter_dqn(n): 
    log_path = os.getcwd() + "/training_runs/seaquest/dqn/"
    fig_path = fig_path = os.getcwd() + "/figures/seaquest/dqn/"
    policy_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    policy_labels = [r"$\pi_\alpha^{\text{DQN}}$", r"$\pi_\beta^{\text{DQN}}$", r"$\pi_\gamma^{\text{DQN}}$", r"$\pi_\delta^{\text{DQN}}$", r"$\pi_\epsilon^{\text{DQN}}$"]
    policy_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    avgs_all = []; stds_all = []; xs = []
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
    return 0 

def seaquest_plotter_ppo(n): 
    log_path = os.getcwd() + "/training_runs/seaquest/ppo/"
    fig_path = fig_path = os.getcwd() + "/figures/seaquest/ppo/"
    policy_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    policy_labels = [r"$\pi_\alpha^{\text{PPO}}$", r"$\pi_\beta^{\text{PPO}}$", r"$\pi_\gamma^{\text{PPO}}$", r"$\pi_\delta^{\text{PPO}}$", r"$\pi_\epsilon^{\text{PPO}}$"]
    policy_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    avgs_all = []; stds_all = []; xs = []
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
    return 0 

def freeway_plotter_dqn(n): 
    log_path = os.getcwd() + "/training_runs/freeway/dqn/"
    fig_path = fig_path = os.getcwd() + "/figures/freeway/dqn/"
    policy_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    policy_labels = [r"$\pi_\alpha^{\text{DQN}}$", r"$\pi_\beta^{{DQN}}$", r"$\pi_\gamma^{\text{DQN}}$", r"$\pi_\delta^{\text{DQN}}$", r"$\pi_\epsilon^{\text{DQN}}$"]
    policy_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    avgs_all = []; stds_all = []; xs = []
    xticks = [[i*1_000_000 for i in range(11)], ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]]
    for i, policy_name in enumerate(policy_names): 
        print(policy_name)
        data_path = log_path + policy_name + "/monitor.csv"
        df = pd.read_csv(data_path)
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
    
    return 0 

def freeway_plotter_ppo(n): 
    log_path = os.getcwd() + "/training_runs/freeway/ppo/"
    fig_path = fig_path = os.getcwd() + "/figures/freeway/ppo/"
    policy_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    policy_labels = [r"$\pi_\alpha^{\text{PPO}}$", r"$\pi_\beta^{\text{PPO}}$", r"$\pi_\gamma^{\text{PPO}}$", r"$\pi_\delta^{\text{PPO}}$", r"$\pi_\epsilon^{\text{PPO}}$"]
    policy_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    avgs_all = []; stds_all = []; xs = []
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
    
    return 0 
        
def doorkey_plotter_dqn(n): 
    log_path = os.getcwd() + "/training_runs/doorkey/dqn/"
    fig_path = fig_path = os.getcwd() + "/figures/doorkey/dqn/"
    policy_name = "alpha"
    policy_label = r"$\pi_\alpha^{\text{DQN}}$"
    policy_color = "tab:blue"
    data_path = log_path + policy_name + "/monitor.csv"
    df = pd.read_csv(data_path, skiprows=[0])
    returns = df["r"]
    lengths = df["l"]
    xticks = [[i*100_000 for i in range(11)], [f"{100*i}K" for i in range(11)]]
    moving_avgs, moving_stds = src.moving_average(returns, n)
    x = [np.sum(lengths[:i]) for i in range(len(lengths))]
    src.lineplot_with_stds(x, 
                        moving_avgs, 
                        moving_stds, 
                        "moving average return " + policy_label, policy_color, 
                        "time step", f"average return of \n last {n} episodes", 
                        fig_path + policy_name + "_training_avg_return.pdf", 
                        xticks=xticks, 
                        alpha = 0.2) 
    return 0         

def doorkey_plotter_ppo(n): 
    log_path = os.getcwd() + "/training_runs/doorkey/ppo/"
    fig_path = fig_path = os.getcwd() + "/figures/doorkey/ppo/"
    policy_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    policy_labels = [r"$\pi_\alpha^{\text{PPO}}$", r"$\pi_\beta^{\text{PPO}}$", r"$\pi_\gamma^{\text{PPO}}$", r"$\pi_\delta^{\text{PPO}}$", r"$\pi_\epsilon^{\text{PPO}}$"]
    policy_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    avgs_all = []; stds_all = []; xs = []
    xticks = [[i*100_000 for i in range(11)], [f"{100*i}K" for i in range(11)]]
    for i, policy_name in enumerate(policy_names): 
        data_path = log_path + policy_name + "/monitor.csv"
        df = pd.read_csv(data_path, skiprows=[0])
        returns = df["r"]
        lengths = df["l"]
        moving_avgs, moving_stds = src.moving_average(returns, n)
        avgs_all.append(moving_avgs); stds_all.append(moving_stds)
        moving_avgs, moving_stds = src.moving_average(returns, n)
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
    return 0           
        
""" log_path = os.getcwd() + "/training_runs/" + environment + algo + "/"
fig_path = os.getcwd() + "/figures/" + environment + algo + "/"
policy_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
selected_policy_names = ["delta", "epsilon"]
policy_labels = [r"$\pi_\alpha^{\text{DQN}}$", r"$\pi_\beta^{\text{DQN}}$", r"$\pi_\gamma^{\text{DQN}}$", r"$\pi_\delta^{\text{DQN}}$", r"$\pi_\epsilon^{\text{DQN}}$"]#, r"$\pi_\zeta$"]
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
                                 alpha = 0.4) """
                                 
                                 
#seaquest_plotter_dqn(100)
#seaquest_plotter_ppo(100)
#freeway_plotter_dqn(100)
freeway_plotter_ppo(100)
doorkey_plotter_dqn(100)
doorkey_plotter_ppo(100)