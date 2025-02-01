import os
from typing import Union, List, Dict, Any, Optional

import pandas as pd

from prorl import SingleRunConfig, ROOT_DIR, logger
from prorl.common.object_handler.base_handler import ObjectHandler
from prorl.common.object_handler.minio_handler import MinioObjectHandler
from prorl.common.print_utils import print_status
from prorl.common.stats_tracker import Tracker
from prorl.common.stats import RunStats, RunStatsCondensed
from prorl.run import RunStatus
from prorl.run.remote import MongoRunWrapper


def parse_list_of_run_stats(list_data: List[Tracker]) -> Dict[str, Any]:
    if len(list_data) > 0:
        all_data = {
            'run_code': list_data[0].run_code,
            'config': list_data[0].config,
            'data': None,
            'run_stats': list_data[0]
        }
        columns = ['reward/add_node', 'reward/movement', 'reward/penalty', 'movement_cost', 'nodes_satisfied',
                   'action_add', 'action_remove', 'action_resource_class', 'action_quantity', 'action_is_random',
                   'choose_action_time', 'learn_time',
                   'loss/add_node', 'loss/movement', 'td_estimate/add_node', 'td_estimate/movement',
                   'reward/add_node/60-steps-avg', 'reward/movement/60-steps-avg',
                   ]
        # TODO: Add support to full stats (nodes information)
        # if len(list_data[0].resources) == 1:
        #     resource = list_data[0].resources[0]
        #     for i in range(len(list_data[0].nodes_resources_history[resource])):
        #         columns.append(f'node_{i}_allocated')
        #         columns.append(f'node_{i}_demanded')
        # else:
        #     for resource in list_data[0].resources:
        #         for i in range(len(list_data[0].nodes_resources_history[resource])):
        #             columns.append(f'{resource}_node_{i}_allocated')
        #             columns.append(f'{resource}_node_{i}_demanded')
        data = []
        for run in list_data:
            j = 0
            skip_loss = True
            loss_reward_len_diff = 0
            if 'train/loss/add_node' in run or 'train/loss/movement' in run:
                skip_loss = False
                if len(run['train/loss/add_node']) > 0:
                    loss_reward_len_diff = len(run['reward/add_node/history']) - len(run['train/loss/add_node'])
                else:
                    loss_reward_len_diff = len(run['reward/add_node/history']) - len(run['train/loss/movement'])
            total_steps = len(run['reward/add_node/history'])
            for i in range(total_steps):
                tmp = [
                    run['reward/add_node/history'][i],
                    run['reward/movement/history'][i],
                    run['reward/penalties'][i],
                    run['movement_cost/history'][i],
                    run['problem_solved/nodes_satisfied'][i],
                    run['action_history/add_node'][i],
                    run['action_history/remove_node'][i],
                    run['action_history/resource_class'][i],
                    run['action_history/quantity'][i],
                    run['action_history/is_random'][i],
                ]
                if 'times/choose_action' in run and 'times/learn_step' in run:
                    tmp.append(run['times/choose_action'][i])
                    tmp.append(run['times/learn_step'][i])
                else:
                    tmp += [None, None]
                if not skip_loss:
                    if i >= loss_reward_len_diff:
                        if len(run['train/loss/add_node']) > 0:
                            tmp.append(run['train/loss/add_node'][j])
                        else:
                            tmp.append(None)
                        if len(run['train/loss/movement']) > 0:
                            tmp.append(run['train/loss/movement'][j])
                        else:
                            tmp.append(None)
                        if len(run['train/td_estimate_avg/add_node']) > 0:
                            tmp.append(run['train/td_estimate_avg/add_node'][j])
                        else:
                            tmp.append(None)
                        if len(run['train/td_estimate_avg/movement']) > 0:
                            tmp.append(run['train/td_estimate_avg/movement'][j])
                        else:
                            tmp.append(None)
                        j += 1
                    else:
                        tmp += [None, None, None, None]
                else:
                    tmp += [None, None, None, None]
                # add 60 steps average values
                if i % 60 == 0:
                    tmp.append(run['reward/add_node/60-steps-avg'][i // 60])
                    tmp.append(run['reward/movement/60-steps-avg'][i // 60])
                else:
                    tmp += [None, None]
                # TODO: Add support to full stats (nodes information)
                # if len(run.resources) == 1:
                #     resource = run.resources[0]
                #     for node_index in range(len(run.nodes_resources_history[resource])):
                #         tmp.append(run.nodes_resources_history[resource][f'n_{node_index}'][i])
                #         tmp.append(run.nodes_demand_history[resource][f'n_{node_index}'][i])
                # else:
                #     for resource in list_data[0].resources:
                #         for node_index in range(len(run.nodes_resources_history[resource])):
                #             tmp.append(run.nodes_resources_history[resource][f'n_{node_index}'][i])
                #             tmp.append(run.nodes_demand_history[resource][f'n_{node_index}'][i])
                data.append(tmp)
                print_status(current=i, total=total_steps,
                             pre_message=f'Parsing data for run with code "{all_data["run_code"]}"',
                             loading_len=40)
        print()
        all_data['data'] = pd.DataFrame(data=data, columns=columns)
        return all_data


def load_single_run_data_from_disk(
        folder_path: str,
        handler: Union[ObjectHandler, MinioObjectHandler],
        parsed=False,
        condensed_stats: bool = False,
        recursive=True
) -> Union[List[Tracker], Dict[str, Any]]:
    if folder_path[-1] != '/':
        folder_path += '/'
    files = handler.list_objects_name(folder_path, recursive=recursive, file_suffix='.json')
    loaded_data: List[Tracker] = []
    for file_path in files:
        loaded_data.append(Tracker.from_dict(handler.load(file_path)))
    if parsed:
        return parse_list_of_run_stats(list_data=loaded_data)
    return loaded_data


def build_single_run_df(run_data: Dict[str, Any]):
    columns = ['step', 'train_reward', 'train_n_problem_solved',
               'val_total_reward', 'val_n_problem_solved',
               'add_net_loss', 'remove_net_loss', 'resource_class_net_loss', 'quantity_net_loss',
               'add_net_q_val', 'remove_net_q_val', 'resource_class_net_q_val', 'quantity_net_q_val']
    data = []
    step_size = run_data['config'].training.step_size
    val_run = run_data['validation_runs'][0]
    val_run_index = 1
    for run in run_data['train_stats']:
        j = 0
        nodes_number = run_data['config'].environment.nodes.get_n_nodes()
        loss_reward_len_diff = len(run.reward_history) - len(run.loss_history['remove_net'])
        for i in range(len(run.reward_history)):
            tmp = [
                i,
                run.reward_history[i],
                1 if run.nodes_satisfied_history[i] == nodes_number else 0,
            ]
            if i * step_size < val_run['train_start_step']:
                tmp.append(None)
                tmp.append(None)
            elif i * step_size == val_run['train_start_step']:
                tmp.append(val_run['total_reward'])
                tmp.append(val_run['n_problem_solved'])
                val_run = run_data['validation_runs'][val_run_index]
                val_run_index += 1
            if i >= loss_reward_len_diff:
                if len(run.loss_history['remove_net']) > 0 and j < len(run.loss_history['remove_net']):
                    tmp.append(run.loss_history['add_net'][j])
                    tmp.append(run.loss_history['remove_net'][j])
                    tmp.append(run.loss_history['resource_class_net'][j])
                    if len(run.loss_history['quantity_net']) > j:
                        tmp.append(run.loss_history['quantity_net'][j])
                    else:
                        tmp.append(None)
                if len(run.q_estimates_history['remove_net']) > 0 and j < len(run.q_estimates_history['remove_net']):
                    tmp.append(run.q_estimates_history['add_net'][j])
                    tmp.append(run.q_estimates_history['remove_net'][j])
                    tmp.append(run.q_estimates_history['resource_class_net'][j])
                    if len(run.q_estimates_history['quantity_net']) > j:
                        tmp.append(run.q_estimates_history['quantity_net'][j])
                    else:
                        tmp.append(None)
                j += 1
            else:
                tmp += [None, None, None, None, None, None]
            data.append(tmp)
    return pd.DataFrame(data=data, columns=columns)


def build_validation_df(validation_run_stats: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = ['reward/add_node', 'reward/movement', 'movement_cost', 'problem_solved/count',
               'train_start_step', 'val_run_code', 'run_steps']
    data = []
    for run in validation_run_stats:
        data.append([
            run['metric/reward/add_node'],
            run['metric/reward/movement'],
            run['metric/movement_cost'],
            run['metric/problem_solved/count'],
            run['train_start_step'],
            run['val_run_code'],
            run['total_steps'],
        ])
    return pd.DataFrame(data=data, columns=columns)


def load_single_run_data(
        run_code: str,
        mongo: MongoRunWrapper,
        handler: Union[ObjectHandler, MinioObjectHandler]
) -> Dict[str, Any]:
    run_db_data = mongo.get_by_run_code(run_code, populate=False)
    config = SingleRunConfig(root_dir=ROOT_DIR, **run_db_data['config'])
    run_stats = load_single_run_data_from_disk(
        folder_path=run_db_data['result_path']['path'],
        handler=handler,
        parsed=False,
        condensed_stats=config.saver.stats_condensed,
        recursive=False
    )
    # val_run_disk_data: List[RunStats] = []
    validation_runs: List[dict] = mongo.get_run_best_validation_runs(
        run_code=run_code,
        limit=0,
        metric=config.run.validation_run.keep_metric,
        status=RunStatus.COMPLETED
    )
    val_run_stats = []
    for i, val_run in enumerate(validation_runs):
        tmp = {
            'val_run_code': val_run['validation_run_code'],
            'train_iteration': val_run['iteration'],
            'total_steps': val_run['total_steps'],
        }
        for key, value in val_run.items():
            if key.find('train/') == 0:
                tmp[key] = value
        val_run_stats.append(tmp)
    val_run_stats.sort(key=lambda x: x['train_iteration'])
    # rain_df = run_stats['data']
    # del run_stats['data']
    data = {
        'run_code': run_code,
        'config': config,
        'total_steps': run_db_data['total_steps'],
        'train_stats': run_stats,
        # 'train_df': train_df
    }
    for key, value in run_db_data.items():
        if key.find('metric/') == 0:
            data[key] = value
    data['validation_runs'] = val_run_stats
    # validation_df = build_validation_df(val_run_stats)
    # data['validation_df'] = validation_df
    return data


def load_multi_run_data(multi_run_code: str, mongo: MongoRunWrapper):
    runs = mongo.get_multi_run_data(multi_run_code)
    runs_processed = []
    for run in runs:
        del run['_id']
        for key in list(run.keys()):
            if 'hp' in key:
                if isinstance(run[key], list):
                    run[key] = str(run[key])

        # if 'hp/hidden-units' in run and isinstance(run['hp/hidden-units'], list):
        #     run['hp/hidden-units'] = str(run['hp/hidden-units'])
        # if 'hp/add-node-units' in run and isinstance(run['hp/add-node-units'], list):
        #     run['hp/add-node-units'] = str(run['hp/add-node-units'])
        runs_processed.append(run)
    df = pd.DataFrame(runs_processed)
    return df


def load_multi_run_validation_runs(multi_run_df: pd.DataFrame, parameters: List[str], mongo: MongoRunWrapper,
                                   metric_prefix='validation/avg/'):
    run_codes: List[str] = multi_run_df.run_code.tolist()
    n_runs = len(run_codes)
    columns = [p for p in parameters]
    columns += ['params', 'run code', 'iteration', 'validation_steps']
    data = []
    for i, run_code in enumerate(run_codes):
        run_data = mongo.get_by_run_code(run_code, populate=True)
        for validation_run in run_data['validation_runs']:
            tmp = []
            params_combined = ''
            for param in parameters:
                param_value = multi_run_df.iloc[i][param]
                tmp.append(param_value)
                if params_combined == '':
                    params_combined += f'{param.replace("hp/", "")}: {str(param_value)}'
                else:
                    params_combined += f' | {param.replace("hp/", "")}: {str(param_value)}'
            tmp.append(params_combined)
            tmp.append(run_code)
            tmp += [
                validation_run['iteration'],
                validation_run['total_steps']
            ]
            for key, value in validation_run.items():
                if metric_prefix in key:
                    tmp.append(value)
                    replaced_key = key.replace(metric_prefix, '')
                    if replaced_key not in columns:
                        columns.append(replaced_key)
            data.append(tmp)
        print_status(i+1, n_runs, f'Loading validation runs for run code "{run_code}"')
    return pd.DataFrame(data=data, columns=columns)


def load_eval_data(eval_code: str,
                   mongo: MongoRunWrapper,
                   handler: Union[ObjectHandler, MinioObjectHandler],
                   load_data=False,
                   seed_prop='agent-seed',
                   average=20) -> Dict[str, Any]:
    result = {
        'eval_df': load_multi_run_data(eval_code, mongo),
        'eval_code': eval_code,
        'eval_data': None
    }
    if load_data:
        columns = ['agent', 'seed', 'reward', 'cumulative_reward', 'nodes_satisfied', 'problem_solved', 'step']
        data = []
        count = 1
        total = len(result['eval_df'])
        for _, row in result['eval_df'].iterrows():
            run_code = row.run_code
            run_db_data = mongo.get_by_run_code(run_code, populate=False)
            config = SingleRunConfig(root_dir=ROOT_DIR, **run_db_data['config'])
            n_nodes = config.environment.nodes.get_n_nodes()
            loaded = load_single_run_data_from_disk(
                folder_path=run_db_data['result_path']['path'],
                handler=handler,
                parsed=False,
                condensed_stats=config.saver.stats_condensed,
                recursive=False
            )
            run_stats: RunStats = loaded[0]
            agent_name = row.agent
            seed = row[seed_prop]
            tmp_avg_rew = []
            tmp_avg_sat = []
            tmp_avg_solved = []
            run_cumulative = 0
            for t in range(len(run_stats.reward_history)):
                reward = run_stats.reward_history[t]
                nodes_satisfied = run_stats.nodes_satisfied_history[t]
                problem_solved = 0
                if nodes_satisfied == n_nodes:
                    problem_solved = 1
                tmp_avg_rew.append(reward)
                tmp_avg_sat.append(nodes_satisfied)
                tmp_avg_solved.append(problem_solved)
                if len(tmp_avg_rew) == average:
                    avg_reward = sum(tmp_avg_rew) / average
                    avg_nodes_satisfied = sum(tmp_avg_sat) / average
                    avg_problem_solved = sum(tmp_avg_solved) / average
                    run_cumulative += sum(tmp_avg_rew)
                    data.append([
                        agent_name,
                        seed,
                        avg_reward,
                        run_cumulative,
                        avg_nodes_satisfied,
                        avg_problem_solved,
                        t
                    ])
                    tmp_avg_rew = []
                    tmp_avg_sat = []
                    tmp_avg_solved = []
            print_status(count, total, pre_message=f'{eval_code} evaluation runs loaded')
            count += 1
        print()
        eval_data = pd.DataFrame(data=data, columns=columns)
        result['eval_data'] = eval_data
    return result


def build_generator_data_df(generated_data: List[dict], add_total_demand=False) -> pd.DataFrame:
    columns = ['value', 'resource', 'base_station',
               'second_step', 'second', 'minute', 'hour', 'week_day', 'week', 'month', 'year',
               'total_steps', 'total_steps_in_hours', 'step_str', 'seed']
    data = []
    for step_data in generated_data:
        step = step_data['step']
        for resource, res_values in step_data['data'].items():
            bs_total = 0
            for bs_values in res_values:
                for base_station, value in bs_values.items():
                    bs_total += value
                    total_steps = step['total_steps']
                    data.append([
                        value,
                        resource,
                        base_station,
                        step['second_step'],
                        step['second'],
                        step['minute'],
                        step['hour'],
                        step['week_day'],
                        step['week'],
                        step['month'],
                        step['year'],
                        total_steps,
                        total_steps // 3600,
                        str(step),
                        step_data['seed']
                    ])
            if add_total_demand:
                total_steps = step['total_steps']
                data.append([
                    bs_total,
                    resource,
                    'total_demand',
                    step['second_step'],
                    step['second'],
                    step['minute'],
                    step['hour'],
                    step['week_day'],
                    step['week'],
                    step['month'],
                    step['year'],
                    total_steps,
                    total_steps // 3600,
                    str(step)
                ])
    return pd.DataFrame(data=data, columns=columns)


def build_generator_wide_form_df(generated_data: List[dict], add_total_demand=False) -> Optional[pd.DataFrame]:
    base_stations: List[str] = []
    data = []
    for step_data in generated_data:
        step = step_data['step']
        for resource, res_values in step_data['data'].items():
            bs_total = 0
            tmp = []
            for bs_values in res_values:
                for base_station, value in bs_values.items():
                    bs_total += value
                    if base_station not in base_stations:
                        base_stations.append(base_station)
                    tmp.append(value)
            if add_total_demand:
                if 'total_demand' not in base_stations:
                    base_stations.append('total_demand')
                tmp.append(bs_total)
            total_steps = step['total_steps']
            # tmp += [
            #     step['second_step'],
            #     step['second'],
            #     step['minute'],
            #     step['hour'],
            #     step['week_day'],
            #     step['week'],
            #     step['month'],
            #     step['year'],
            #     total_steps,
            #     total_steps // 3600
            # ]
            data.append(tmp)

    columns = base_stations
    # columns += ['second_step', 'second', 'minute', 'hour', 'week_day', 'week', 'month', 'year',
    #             'total_steps', 'total_steps_in_hours']
    return pd.DataFrame(data=data, columns=columns)


def merge_cost_and_no_cost_runs(code: str, mongo: MongoRunWrapper, no_cost_eval_codes: List[str]):
    with_cost_df = load_multi_run_data(code, mongo)
    parts = code.split('-')
    no_cost_code_sub_part = ''
    for i, part in enumerate(parts):
        if 'scheduled_at=' in part:
            no_cost_code_sub_part += f'-no-cost'
            break
        else:
            if i == 0:
                no_cost_code_sub_part = part
            else:
                no_cost_code_sub_part += f'-{part}'

    no_cost_code = ''
    for c in no_cost_eval_codes:
        if no_cost_code_sub_part in c:
            no_cost_code = c
            break
    no_cost_df = load_multi_run_data(no_cost_code, mongo)
    with_cost_df.replace('double-dqn-sub-optimal', 'double-dqn', inplace=True)
    no_cost_df.replace('double-dqn-sub-optimal', 'double-dqn', inplace=True)
    no_cost_df['agent'] = no_cost_df['agent'] + '_no-cost'
    return pd.concat([with_cost_df, no_cost_df], ignore_index=True)
