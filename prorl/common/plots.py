import os
from collections import OrderedDict
from typing import List, Dict, Any, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from prorl import SingleRunConfig
from prorl.common.filesystem import create_directory_from_filepath
from prorl.common.print_utils import beautify_str
from prorl.common.stats import RunStats, RunStatsCondensed

sns.set(style="darkgrid", font_scale=2)


def filter_multi_run_code(multi_run_code: str) -> str:
    parts = multi_run_code.split('-')
    stop_index = 0
    for i, part in enumerate(parts):
        if 'scheduled_at=' in part:
            stop_index = i
            break
    code = '-'.join(parts[0: stop_index])
    return code.replace('v2-', '')


def plot_smoothed(data: pd.DataFrame, data_field: str, smoothing_factor=0.6, figsize=(8, 8)):
    fig = plt.figure(figsize=figsize)
    smoothed = data.ewm(alpha=(1 - smoothing_factor)).mean()
    plt.plot(data[data_field], alpha=0.35)
    plt.plot(smoothed[data_field])
    return fig


def plot_reward(df: pd.DataFrame, config: SingleRunConfig, figsize=(16, 6), reward_field='reward',
                smoothing_factor=0.85, opacity=0.2, plot_name='Reward'):
    model = config.emulator.model.type
    agent = config.environment.agent.type
    time_steps = len(df) // config.run.step_size
    sup_title = f'{plot_name} - Run {time_steps} steps using agent {agent.value} and generator model {model.value}'
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(sup_title)

    axs[0].plot(df[reward_field], alpha=opacity)
    axs[0].plot(df.ewm(alpha=(1 - smoothing_factor)).mean()[reward_field], 'tab:blue')
    axs[0].set_title('Single step reward')
    axs[0].set_ylabel('Reward')
    axs[0].set_xlabel('Time step')

    cumulative_reward = np.cumsum(df[reward_field].to_numpy(), dtype=np.float64)

    axs[1].plot(cumulative_reward)
    axs[1].set_title('Cumulative reward')
    axs[1].set_ylabel('Cumulative reward')
    axs[1].set_xlabel('Time step')
    # plt.show()
    # return fig


def plot_nodes_satisfied(df: pd.DataFrame, config: SingleRunConfig, figsize=(8, 8)):
    model = config.emulator.model.type
    agent = config.environment.agent.type
    title = f'Number of nodes with sufficient resources per time step -' \
            f' agent {agent.value} and generator model {model.value}'
    fig = plt.figure(figsize=figsize)
    plot = sns.lineplot(data=df, x=df.index, y='nodes_satisfied')
    plt.xlabel('Time step')
    plt.ylabel('# nodes satisfied')
    plt.title(title)
    plt.axhline(y=config.environment.nodes.get_n_nodes(), lw=5, color='r', label='Target')
    plt.legend(['Nodes satisfied', 'Target'])
    if 'action_is_random' in df:
        end_exploration = config.environment.agent.double_dqn.epsilon_parameters['total']
        not_satisfied_random = 0
        not_satisfied_greedy = 0
        n_target = config.environment.nodes.get_n_nodes()
        for i in range(len(df)):
            n_satisfied = df.nodes_satisfied[i]
            is_random = df.action_is_random[i]
            if i > end_exploration:
                if n_satisfied < n_target:
                    if bool(is_random) is True:
                        not_satisfied_random += 1
                    else:
                        not_satisfied_greedy += 1
        print('After the end of the exploration (at time {}) the number of not succeeded steps is {}'.format(
            end_exploration, not_satisfied_greedy+not_satisfied_random))
        print('\t- the number not succeeded steps with greedy action is: {}'.format(not_satisfied_greedy))
        print('\t- the number not succeeded steps with random action is: {}'.format(not_satisfied_random))
        is_random = df.action_is_random.tolist()
        is_random = [n_target + 0.1 if v is True else None for v in is_random]
        np_random = np.array(is_random)
        plt.scatter(x=np.arange(0, len(df)), y=np_random, s=4, color='g', marker='x', label='Random Action')
        plt.legend(['Nodes satisfied', 'Target', 'Random actions'])
    # plt.show()
    # return plot.get_figure()


def line_plot_nodes_satisfied(df: pd.DataFrame, config: SingleRunConfig, figsize=(8, 8)):
    model = config.emulator.model.type
    agent = config.environment.agent.type
    n_nodes = config.environment.nodes.get_n_nodes()
    title = f'Number of steps for which the problem was solved -' \
            f' agent {agent.value} and generator model {model.value}'
    fig = plt.figure(figsize=figsize)

    np_nodes_satisfied = np.array(df.nodes_satisfied)
    np_nodes_satisfied[np_nodes_satisfied < n_nodes] = 0
    np_nodes_satisfied[np_nodes_satisfied == n_nodes] = 1
    cum_sum = np.cumsum(np_nodes_satisfied)

    plot = sns.lineplot(x=np.arange(0, len(cum_sum)), y=cum_sum)
    plt.xlabel('Time step')
    plt.ylabel('# problem solved')
    plt.title(title)
    plt.axhline(y=config.environment.nodes.get_n_nodes(), lw=5, color='r', label='Target')
    plt.legend(['Nodes satisfied', 'Target'])
    # plt.show()
    # return plot.get_figure()


def plot_nodes(df: pd.DataFrame, config: SingleRunConfig, figsize=(40, 20)):
    model = config.emulator.model.type
    agent = config.environment.agent.type
    time_steps = len(df) // config.run.step_size
    n_nodes = config.environment.nodes.get_n_nodes()
    sup_title = f'Nodes - Run {time_steps} steps using agent {agent.value} and generator model {model.value}'
    fig, axs = plt.subplots(n_nodes // 3 + 1, 3, figsize=figsize)
    fig.suptitle(sup_title)
    all_allocated = np.zeros(len(df.reward))
    all_demanded = np.zeros(len(df.reward))
    row = 0
    col = 0
    for i in range(n_nodes):
        allocated = df[f'node_{i}_allocated']
        demanded = df[f'node_{i}_demanded']
        all_allocated += allocated
        all_demanded += demanded
        sns.lineplot(x=df.index, y=allocated.to_numpy(), ax=axs[row, col])
        sns.lineplot(x=df.index, y=demanded.to_numpy(), ax=axs[row, col])
        axs[row, col].set_title(f'Node {i} - resource allocated and demanded')
        axs[row, col].set_ylabel('Resource')
        axs[row, col].set_xlabel('Time step')
        axs[row, col].legend(['allocated', 'demanded'])
        col += 1
        if col == 3:
            row += 1
            col = 0
    if col > 2:
        col = 1
    sns.lineplot(x=df.index, y=np.array(all_allocated, dtype=np.float64), ax=axs[-1, col])
    sns.lineplot(x=df.index, y=np.array(all_demanded, dtype=np.float64), ax=axs[-1, col])
    axs[-1, col].set_title(f'All nodes - resource allocated and demanded')
    axs[-1, col].set_ylabel('Resource')
    axs[-1, col].set_xlabel('Time step')
    axs[-1, col].legend(['allocated', 'demanded'])
    # plt.show()
    # return fig


def plot_run_loss(df: pd.DataFrame, config: SingleRunConfig, figsize=(30, 15), is_combined: bool = False,
                  smoothing_factor=0.85, opacity=0.2):
    model = config.emulator.model.type
    agent = config.environment.agent.type
    time_steps = len(df) // config.run.step_size
    sup_title = f'Loss - Run {time_steps} steps using agent {agent.value} and generator model {model.value}'
    n_plots = 2
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(sup_title)

    axs[0].plot(df['loss/add_node'], alpha=opacity)
    axs[0].plot(df.ewm(alpha=(1 - smoothing_factor)).mean()['loss/add_node'], 'tab:blue')
    axs[0].set_title('Add net loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Time step')

    axs[1].plot(df['loss/movement'], alpha=opacity)
    axs[1].plot(df.ewm(alpha=(1 - smoothing_factor)).mean()['loss/movement'], 'tab:blue')
    axs[1].set_title('Movement net loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Time step')

    # if not is_combined:
    #     axs[1, 0].plot(df['resource_class_net_loss'], alpha=opacity)
    #     axs[1, 0].plot(df.ewm(alpha=(1 - smoothing_factor)).mean()['resource_class_net_loss'], 'tab:blue')
    #     axs[1, 0].set_title('Resource class net loss')
    #     axs[1, 0].set_ylabel('Loss')
    #     axs[1, 0].set_xlabel('Time step')
    #
    #     axs[1, 1].plot(df['quantity_net_loss'], alpha=opacity)
    #     axs[1, 1].plot(df.ewm(alpha=(1 - smoothing_factor)).mean()['quantity_net_loss'], 'tab:blue')
    #     axs[1, 1].set_title('Quantity net loss')
    #     axs[1, 1].set_ylabel('Loss')
    #     axs[1, 1].set_xlabel('Time step')
    #
    # else:
    #     axs[1, 0].plot(df['resource_class_net_loss'], alpha=opacity)
    #     axs[1, 0].plot(df.ewm(alpha=(1 - smoothing_factor)).mean()['resource_class_net_loss'], 'tab:blue')
    #     axs[1, 0].set_title('Combined net loss')
    #     axs[1, 0].set_ylabel('Loss')
    #     axs[1, 0].set_xlabel('Time step')

    # return fig


def plot_run_q_estimates(df: pd.DataFrame, config: SingleRunConfig, figsize=(20, 10)):
    model = config.emulator.model.type
    agent = config.environment.agent.type
    time_steps = len(df) // config.run.step_size
    sup_title = f'Q-Estimates - Run {time_steps} steps using agent {agent.value} and generator model {model.value}'
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(sup_title)

    sns.lineplot(data=df, x=df.index, y='remove_net_q_val', ax=axs[0])
    axs[0].set_title('Remove net q estimates')
    axs[0].set_ylabel('Q-values')
    axs[0].set_xlabel('Time step')

    sns.lineplot(data=df, x=df.index, y='add_net_q_val', ax=axs[1])
    axs[1].set_title('Add net loss')
    axs[1].set_ylabel('Q-values')
    axs[1].set_xlabel('Time step')

    sns.lineplot(data=df, x=df.index, y='quantity_net_q_val', ax=axs[2])
    axs[2].set_title('Quantity net loss')
    axs[2].set_ylabel('Q-values')
    axs[2].set_xlabel('Time step')

    # return fig


def zoom_plot(data: pd.DataFrame, field: str, starts: List[int], ends: List[int], figsize=(20, 6)):
    values = data[field].to_numpy()
    fig = plt.figure(figsize=figsize)
    plot = sns.lineplot(x=np.arange(0, len(values)), y=values)
    if len(starts) % 3 == 0:
        rows = len(starts) // 3
    else:
        rows = len(starts) // 3 + 1
    if rows == 0:
        rows = 1
    cols = len(starts)
    if cols >= 3:
        cols = 3
    sub_fig, axs = plt.subplots(rows, cols, figsize=(figsize[0], 6*rows))
    row = 0
    col = 0
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        if rows == 1:
            sns.lineplot(x=np.arange(start, end), y=values[start: end], ax=axs[col])
        else:
            sns.lineplot(x=np.arange(start, end), y=values[start: end], ax=axs[row, col])
        col += 1
        if col == 3:
            row += 1
            col = 0
    # plt.show()

    # return plot.get_figure(), sub_fig


def plot_single_run_with_validation(df: pd.DataFrame, figsize=(20, 10)):
    val_total_timesteps = df.run_steps[0]
    # plot_reward(df, config, figsize, reward_field='train_reward')
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    plt.suptitle('Validation rewards')
    sns.lineplot(data=df, x='train_start_step', y='reward/add_node', ax=axs[0])
    axs[0].set_xlabel('Train time step')
    axs[0].set_ylabel('Validation add node reward')
    axs[0].set_title(f'Validation add node total reward. Total time steps: {val_total_timesteps}')
    sns.lineplot(data=df, x='train_start_step', y='reward/movement', ax=axs[1])
    axs[1].set_xlabel('Train time step')
    axs[1].set_ylabel('Validation movement reward')
    axs[1].set_title(f'Validation movement total reward. Total time steps: {val_total_timesteps}')

    plt.figure(figsize=(figsize[0], figsize[1] / 1.2))
    sns.lineplot(data=df, x='train_start_step', y='problem_solved/count')
    plt.xlabel('Train time step')
    plt.ylabel('Steps with problem solved')
    plt.title(
        f'Validation runs number of problem solved per time step. Total time steps: {val_total_timesteps}')


def plot_multi_run_df(multi_run_df: pd.DataFrame, agent_name: str, param_1, param_2=None, figsize=(20, 10), title=None,
                      metric='validation_best_total_reward', save_path=None):
    seeds_count = {}
    for index, row in multi_run_df.iterrows():
        if row['seed'] not in seeds_count:
            seeds_count[row['seed']] = True
    n_seeds = len(seeds_count)
    fig = plt.figure(figsize=figsize)
    plot = sns.boxplot(data=multi_run_df, x=param_1, y=metric, hue=param_2)
    multi_run_code = multi_run_df.iloc[0]['multi_run_code']
    total_reward_title = f'Multi run over {n_seeds} seeds - {beautify_str(filter_multi_run_code(multi_run_code))}'
    if param_2 is not None:
        total_reward_title += f' and {param_2}'
    if title is not None:
        total_reward_title = title
    plt.title(total_reward_title)
    if save_path is not None:
        full_path = os.path.join(save_path, f'{agent_name}_training.pdf')
        create_directory_from_filepath(full_path)
        fig = plot.get_figure()
        fig.savefig(full_path, bbox_inches='tight', dpi=plt.gcf().dpi)


def plot_multi_run_df_sub_figure(multi_run_df: pd.DataFrame, agent_name: str,
                                 params: List[str], figsize=(30, 10), title=None,
                                 metric='validation/problem_solved/count', save_path=None):
    seeds_count = {}
    for index, row in multi_run_df.iterrows():
        if row['hp/seed'] not in seeds_count:
            seeds_count[row['hp/seed']] = True
    n_seeds = len(seeds_count)
    multi_run_code = multi_run_df.iloc[0]['multi_run_code']
    multi_run_code_title = multi_run_code.replace('alpha-', 'alpha=')
    total_reward_title = f'Multi run over {n_seeds} seeds -' \
                         f' {beautify_str(filter_multi_run_code(multi_run_code_title))} - ' \
                         f'{beautify_str(metric.replace("validation/performance/", "validation/"), "/")}'
    if len(params) > 2:
        fig, axs = plt.subplots(1, len(params), figsize=figsize)
        if title is not None:
            total_reward_title = title
        plt.suptitle(total_reward_title)
        for i in range(len(params)):
            param_1 = params[i]
            param_2 = params[i + 1] if i + 1 < len(params) else params[0]
            sns.boxplot(data=multi_run_df, x=param_1, y=metric, hue=param_2, ax=axs[i])
            axs[i].set_xlabel(beautify_str(param_1.replace('hp/', '')))
            axs[i].set_ylabel(beautify_str(metric, '/'))
            axs[i].set_title(f'Filtered by {beautify_str(param_1)} and {beautify_str(param_2)}')
    elif len(params) == 1:
        fig = plt.figure(figsize=figsize)
        sns.boxplot(data=multi_run_df, x=params[0], y=metric)
        plt.title(total_reward_title)
        plt.xlabel(beautify_str(params[0].replace('hp/', '')))
        plt.ylabel(beautify_str(metric, '/'))
    else:
        fig = plt.figure(figsize=figsize)
        sns.boxplot(data=multi_run_df, x=params[0], y=metric, hue=params[1])
        plt.title(total_reward_title)
        plt.xlabel(beautify_str(params[0].replace('hp/', '')))
        plt.ylabel(beautify_str(metric, '/'))
    if save_path is not None:
        metric_name = metric.replace('/', '-')
        full_path = os.path.join(save_path, f'{agent_name}_{metric_name}_training.pdf')
        create_directory_from_filepath(full_path)
        fig.savefig(full_path, bbox_inches='tight', dpi=plt.gcf().dpi)


def plot_multi_run_validations_df(multi_run_df: pd.DataFrame, validation_df: pd.DataFrame, best_params,
                                  figsize=(20, 10), metric='validation/problem_solved/count',
                                  step_name='iteration', validation_step_name='total_steps', save_path=None):
    seeds_count = {}
    for index, row in multi_run_df.iterrows():
        if row['hp/seed'] not in seeds_count:
            seeds_count[row['hp/seed']] = True
    n_seeds = len(seeds_count)
    multi_run_code = multi_run_df.iloc[0]['multi_run_code']
    best_runs = validation_df[validation_df.params == best_params]
    worst_runs = validation_df[validation_df.params != best_params]
    validation_steps = validation_df.iloc[0][validation_step_name]
    total_train_steps = validation_df[step_name].max()
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    plt.suptitle(f'Validation average total'
                 f' {beautify_str(metric.replace("eval/", "").replace("total/", "").replace("reward/", ""), "/")}'
                 f' over {n_seeds} seeds -'
                 f' {beautify_str(filter_multi_run_code(multi_run_code))}')
    sns.lineplot(data=best_runs, x=step_name, y=metric.replace('validation', 'metric'), hue='params', ax=axs[0])
    # axs[0].axhline(validation_steps, xmin=0, xmax=total_train_steps, linestyle='--',
    #                c='tab:red', label=f'Total steps: {validation_steps}')
    axs[0].set_title(f'Best params combination')
    axs[0].set_xlabel(step_name)
    axs[0].set_ylabel(beautify_str(metric.replace('eval/', ''), '/'))
    axs[0].legend()
    sns.lineplot(data=worst_runs, x=step_name, y=metric.replace('validation', 'metric'), hue='params', ax=axs[1])
    # axs[1].axhline(validation_steps, xmin=0, xmax=total_train_steps, linestyle='--',
    #                c='tab:red', label=f'Total steps: {validation_steps}')
    axs[1].set_title(f'Worst params combinations')
    axs[1].set_xlabel(step_name)
    axs[1].set_ylabel(beautify_str(metric.replace('eval/', ''), '/'))
    axs[1].legend(loc='upper center', bbox_to_anchor=(1.12, 0.9))
    if save_path is not None:
        metric_name = metric.replace('eval/', '').replace('/', '-')
        full_path = os.path.join(save_path, f'validations_{metric_name}_training.pdf')
        create_directory_from_filepath(full_path)
        fig.savefig(full_path, bbox_inches='tight', dpi=plt.gcf().dpi)



def plot_eval_df(df: pd.DataFrame, eval_name, figsize=(20, 10), title=None, metric='train_total_reward',
                 seed_prop='agent-seed', save_path=None):
    seeds_count = {}
    for index, row in df.iterrows():
        if row[seed_prop] not in seeds_count:
            seeds_count[row[seed_prop]] = True
    n_seeds = len(seeds_count)
    fig = plt.figure(figsize=figsize)
    plot = sns.boxplot(data=df, x=metric, y='agent')
    total_reward_title = f'Evaluation run over {n_seeds} seeds "{eval_name}" - {beautify_str(metric)}'
    plt.xlabel(beautify_str(metric))
    plt.ylabel('Agent')
    if title is not None:
        total_reward_title = title
    plt.title(total_reward_title)
    if save_path is not None:
        full_path = os.path.join(save_path, f'overall_result.pdf')
        create_directory_from_filepath(full_path)
        fig.savefig(full_path, bbox_inches='tight', dpi=plt.gcf().dpi)
    return plot.get_figure()


def plot_eval_metric(df: pd.DataFrame, eval_name, y: str, x: str = 'step', hue='agent', title=None,
                     figsize=(20, 10), save_path=None):
    fig = plt.figure(figsize=figsize)
    plot = sns.lineplot(data=df, x=x, y=y, hue=hue)
    plt.xlabel(beautify_str(x))
    plt.ylabel(beautify_str(y))
    plot_title = f'Evaluation run "{eval_name}" - {beautify_str(y)}'
    if title is not None:
        plot_title = title
    plt.title(plot_title)
    if save_path is not None:
        full_path = os.path.join(save_path, f'{y}.pdf')
        create_directory_from_filepath(full_path)
        fig.savefig(full_path, bbox_inches='tight', dpi=plt.gcf().dpi)
    return plot.get_figure()


def plot_generated_data(df: pd.DataFrame, n_base_stations: int, step_size: int,
                        print_total_demand: bool = False, figsize=(25, 25),
                        x=None, style=None,
                        n_seeds=1,
                        save_path=None,
                        stress_test_probability=None,
                        **kwargs):
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    x_filter = x if x is not None else df.index // (n_base_stations * step_size)
    stress_test = ''
    if stress_test_probability is not None:
        stress_test = f' - stress test probability {stress_test_probability}'
    plt.suptitle(f'1 week of data generated by {n_base_stations} base stations - using {n_seeds} seeds{stress_test}')
    sns.lineplot(data=df, x=x_filter, y='value', hue='base_station', style=style, ci='sd', ax=axs[0], **kwargs)
    axs[0].set_title('std values')
    axs[0].set_xlabel('Hour in the week')
    axs[0].set_ylabel('Demand requested')
    sns.lineplot(data=df, x=x_filter, y='value', hue='base_station', style=style, ci=95, ax=axs[1], **kwargs)
    axs[1].set_title('95 ci values')
    axs[1].set_xlabel('Hour in the week')
    axs[1].set_ylabel('Demand requested')
    if print_total_demand:
        max_total_demand = df.groupby(['base_station']).max().loc['total_demand']['value']
        print(f'Total demand max value is {max_total_demand}')
    if save_path is not None:
        create_directory_from_filepath(save_path)
        fig.savefig(save_path, bbox_inches='tight', dpi=plt.gcf().dpi)


def plot_generated_data_avg(df: pd.DataFrame, n_base_stations: int, avg_field='hour', figsize=(20, 10)):
    fig = plt.figure(figsize=figsize)
    sns.lineplot(data=df, x=avg_field, y='value', hue='base_station')
    plt.title(f'Average load per hour over 1 month of generated data using {n_base_stations}')
    plt.xlabel('Hour')
    plt.ylabel('Average demand')


def plot_reward_info(reward_info_history: List[Dict[str, float]], figsize=(10, 10), smoothing_factor=0.85, opacity=0.2):
    fig = plt.figure(figsize=figsize)
    columns = list(reward_info_history[0].keys())
    data = []
    for elem in reward_info_history:
        tmp = []
        for _, value in elem.items():
            tmp.append(value)
        data.append(tmp)
    df = pd.DataFrame(data=data, columns=columns)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tag:red', 'tab:purple']
    i = 0
    for col in columns:
        plt.plot(df[col], alpha=opacity)
        plt.plot(df.ewm(alpha=(1 - smoothing_factor)).mean()[col], colors[i])
        i += 1
        if i >= len(colors):
            i = 0
    plt.xlabel('Time step')
    plt.ylabel('Reward value')
    plt.title('Reward components')


def print_action_history(action_history: Dict[str, List[Union[int, bool, None]]], figsize=(20, 10)):
    print('\tAction History')
    action_count = {}
    columns = ['action', 'count', 'sub-action']
    data = []
    for sub_action, history in action_history.items():
        if sub_action != 'is_random':
            tmp = {}
            for action in history:
                if action not in tmp:
                    tmp[action] = 1
                else:
                    tmp[action] += 1
            tmp = dict(OrderedDict(sorted(tmp.items())))
            for a, count in tmp.items():
                data.append([a, count, sub_action])
            action_count[sub_action] = tmp
            print(f'\t\tSub action {sub_action} actions selection: {tmp}')
    action_df = pd.DataFrame(data=data, columns=columns)
    fig = plt.figure(figsize=figsize)
    sns.barplot(data=action_df, x='sub-action', y='count', hue='action')
    plt.xlabel('Sub action')
    plt.ylabel('Count')
    plt.title('Sub-actions distribution')


def plot_action_history(df: pd.DataFrame, figsize=(20, 10)):
    actions_count: Dict[str, Dict[str, Dict[str, int]]] = {
        'add_node': {},
        'remove_node': {},
        'resource_class': {},
        'quantity': {},
    }

    def add_action(sub_action, action, is_random):
        if action not in actions_count[sub_action]:
            actions_count[sub_action][action] = {
                'count': 0,
                'random_count': 0
            }
        if is_random:
            actions_count[sub_action][action]['random_count'] += 1
        else:
            actions_count[sub_action][action]['count'] += 1

    for _, row in df.iterrows():
        add_action('add_node', row['action_add'], row['action_is_random'])
        add_action('remove_node', row['action_remove'], row['action_is_random'])
        add_action('resource_class', row['action_resource_class'], row['action_is_random'])
        add_action('quantity', row['action_quantity'], row['action_is_random'])
    columns = ['action', 'sub-action', 'count', 'is-random']
    data = []
    for sub_action, counts in actions_count.items():
        for action, action_counts in counts.items():
            data.append([
                action,
                sub_action,
                action_counts['count'],
                False
            ])
            data.append([
                action,
                sub_action,
                action_counts['random_count'],
                True
            ])

    fig = plt.figure(figsize=figsize)
    df = pd.DataFrame(data=data, columns=columns)
    sns.catplot(data=df, x="sub-action", y="count", hue="action", col="is-random", kind="bar", height=15, aspect=1)
    # sns.barplot(data=df, x='sub-action', y='count', hue='action')
    plt.xlabel('Sub action')
    plt.ylabel('Count')
    plt.title('Sub-actions distribution')


def plot_single_run(run_data: Dict[str, Any], figsize=(22, 22), smoothing_factor=0.9, opacity=0.4):
    train_df = run_data['train_df']
    validation_df = run_data['validation_df']
    config: SingleRunConfig = run_data['config']
    model = config.emulator.model.type
    agent = config.environment.agent.type
    time_steps = len(train_df) // config.run.step_size
    print('\n--------------------------------------')
    print(f'Training result for run with code: {run_data["run_code"]}')
    print(f'\tAgent: {agent} - model: {model} - training steps: {time_steps}')
    print(f'\t\tSub-actions combined: {config.environment.agent.combine_last_sub_actions}')
    print(f'\t\tAgent config: {config.environment.agent[agent.value.replace("-", "_")]}')
    print(f'\t\tReward type: {config.environment.reward.type}')
    print(f'\t\t\tReward additional: {config.environment.reward.parameters}')
    print(f'\t\tResource classes number: {len(config.environment.resources[0].classes)}')
    plot_reward(df=train_df, config=config, figsize=figsize, reward_field='reward/add_node',
                smoothing_factor=smoothing_factor, opacity=opacity, plot_name='Reward Add Node Sub-agent')
    plot_reward(df=train_df, config=config, figsize=figsize, reward_field='reward/movement',
                smoothing_factor=smoothing_factor, opacity=opacity, plot_name='Reward Movement Sub-agent')
    plot_single_run_with_validation(df=validation_df, figsize=(figsize[0] * 0.8, figsize[1] * 0.8))
    plot_run_loss(df=train_df, config=config, figsize=(figsize[0], figsize[1] / 2),
                  smoothing_factor=smoothing_factor, opacity=opacity)
    # fig = plt.figure(figsize=(figsize[1] / 2, figsize[1] / 3))
    # plt.plot(df['train_n_problem_solved'].cumsum())
    # plt.title('Number of problem solved per time step')
    # plt.xlabel('Time step')
    # plt.ylabel('Problem solved count')
    # has_reward_info_history = getattr(run_stats, 'reward_info_history', None)
    # if has_reward_info_history is not None and len(run_stats.reward_info_history) > 0:
    #     plot_reward_info(run_stats.reward_info_history, figsize=(figsize[1] / 2, figsize[1] / 3),
    #                      smoothing_factor=smoothing_factor, opacity=opacity)

    plot_action_history(train_df, figsize=(figsize[0] * 2, figsize[1]))
    # print_action_history(run.action_history, figsize=(figsize[0] * 1.5, figsize[0] / 2))
    plt.show()

