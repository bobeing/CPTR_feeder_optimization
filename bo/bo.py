from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import torch
from ax import *
from ax.runners.synthetic import SyntheticRunner

from ax.modelbridge.cross_validation import cross_validate, compute_diagnostics
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.pareto_utils import get_observed_pareto_frontiers
import matplotlib.pyplot as plt

import numpy as np

# Plotting imports and initialization
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier

from ax.service.utils.report_utils import exp_to_df
from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_PAREGO
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax import Runner

from ax import Data
from ax.metrics.noisy_function import NoisyFunctionMetric
# from ax.core import Metric
import pandas as pd

import sys
import time

def param_dict_to_sorted_list(param_dict):
    param_keys = list(param_dict.keys()).sort()
    global param_values
    param_values = []
    for key in param_keys:
        param_values.append(param_dict[key])
    return param_values

def sorted_param_list_to_dict(param_list):
    if len(param_list) != len(sorted_keys):
        raise ValueError

    global param_dict
    param_dict = {}
    for i, key in enumerate(sorted_keys):
        param_dict[key] = param_list[i]
    
    return param_dict


def my_cross_validate(client):
    cv = cross_validate(client.generation_strategy.model)
    
    diagnostics = compute_diagnostics(cv)
    render(interact_cross_validation(cv))
    

def visualize_pareto(client):

    objectives = client.experiment.optimization_config.objective.objectives
    
    frontier_ht_d = compute_posterior_pareto_frontier(
        experiment=client.experiment,
        data=client.experiment.fetch_data(),
        primary_objective=objectives[1].metric,
        secondary_objective=objectives[0].metric,
        absolute_metrics=["HT", "D"],
        num_points=40,
    )

    observed_pareto = get_observed_pareto_frontiers(
        experiment=client.experiment,
        data = client.experiment.fetch_data())

    plt.plot(np.array(hv_list))
    plt.ylabel('Normalized hypervolume')
    plt.xlabel('The number of iterations')
    
    render(plot_pareto_frontier(frontier_ht_d, CI_level=0.90))


    for op in observed_pareto:
        render(plot_pareto_frontier(op, CI_level=0.90))

    

# def evalutate(parameters, exp2_x, exp2_xbounds, exp2_y, exp2_ybounds):
#     return {
#         'HT':(8.977390559037276e-16, noise_sd['HT']),
#         'D':(1.2072785164571571, noise_sd['D']),
#     }




# exp2_x = ['feeder_distance',  'feeder_theta']
exp2_x = ['feeder_distance']
sorted_keys = exp2_x.copy()

exp2_xbounds = {
    'feeder_distance': [48.08, 48.28],
    # 'feeder_theta': [120, 170]
}


"""
    Zernike polynomials
    (n, m) 
    (0, 0) : Pistone (P)
    (1, -1) : Vetical tilt (VT)
    (1, 1) : Horizontal tilt (HT)
    (2, -2) : Oblique astigmatism
    (2, 0) : Defocus (D)
    (2, 2) : Horizontal astigmatism
""" 

# distance 만 바꿀 때 ['Gain','Horizontal Tilt', 'Defocus']
# elevation 만 바꿀 때 ['Gain','Vertical Tilt', 'Defocus']

# exp2_y = ['P', 'VT', 'HT', 'D']
exp2_y = ['HT', 'D']

exp2_ybounds = {
    # 'G': [-120, -20],
    'HT' : [0, 50],
    'D' : [0, 4.5],
}

    
parameters = []
for key in sorted_keys:
    temp = {}
    temp["name"] = key
    temp["type"] = 'range'
    temp["bounds"] = list(exp2_xbounds[key])
    temp["value_type"] ='float'
    temp["log_scale"] = False

    parameters.append(temp)

params = [RangeParameter(name=key, lower=list(exp2_xbounds[key])[0], upper=list(exp2_xbounds[key])[1], parameter_type=ParameterType.FLOAT) for key in sorted_keys]
# print(params)

search_space = SearchSpace(
        parameters=params
)

ax_client = AxClient(verbose_logging=False)
ax_client.create_experiment(
    name = 'zernike polynomial exp',
    parameters = parameters,
    objectives = {
        # 'G': ObjectiveProperties(minimize=False, threshold=-20),
        # 'VT': ObjectiveProperties(minimize=False, threshold=100),
        'HT': ObjectiveProperties(minimize=True, threshold=0),
        'D': ObjectiveProperties(minimize=True, threshold=0),
    },
    # outcome_constraints= ['YS <= 105', 'YS >= 91.4'],
    overwrite_existing_experiment=False,
    is_test=False,
)

noise_sd = {
    # 'G': 0,
    # 'VT': 0,
    'HT': 0,
    'D': 0,
}


input_parameters = [
    {
        'feeder_distance': 48.106788
    },
    {
        'feeder_distance': 48.2452
    },
    {
        'feeder_distance': 48.12332    
    },
    {
        'feeder_distance': 48.144808
    },
    # 여기부턴 추천 받은 샘플 포인트
    # {
    #     # 추천 받은 값은 48.209231536388394 인데 TICRA 에서 올림 돼서 48.20923154
    #     'feeder_distance': 48.20923154
    # },
    # {
    #     # 추천 받은 값은 48.119034497737886 이지만 TICRA에선 48.1190345
    #     'feeder_distance': 48.1190345
    # },
    # {
    #     'feeder_distance': 48.22236639
    # },
    # {
    #     'feeder_distance': 48.26965919017792
    # }
    
    # 범위 값 바꿈
    {
        'feeder_distance': 48.20494164
    },
    {
        'feeder_distance': 48.16653184
    },
    {
        'feeder_distance': 48.11249312
    },
    {
        'feeder_distance': 48.18863725
    },
    {
        'feeder_distance': 48.1597859
    },
    {
        'feeder_distance': 48.24960251
    },
    {
        'feeder_distance': 48.10903039
    },
    {
        'feeder_distance': 48.16331369
    },
]

#절대값으로 넣기
result = [
    {
        'HT': 3.34862450e+01,
        'D': 2.40187288e+00
    },
    {
        'HT': 19.36041583,
        'D': 1.80729646
    },
    {
        'HT': 4.19701596e+00,
        'D': 1.05900694e+01
    },
    {
        'HT': 1.72007758e+01,
        'D': 1.20727852e+00
    },
    # # 여기부턴 추천 받은 샘플 포인트의 coefficient 값
    # {
    #     'HT': 1.03855590e+01,
    #     'D': 7.47149493e-01
    # },
    # {
    #     'HT': 2.82405575e+01,
    #     'D': 2.01462340e+00
    # },
    # {
    #     'HT': 1.60142709e+01,
    #     'D': 1.14068015e+00
    # }
    # 범위값 바꿈
    {
        'HT': 6.41059264,
        'D': 0.5996039
    },
    {
        'HT': 7.89858357e+00,
        'D': 5.38983174e-01
    },
    {
        'HT': 3.10425937e+01,
        'D': 2.22129714e+00
    },
    {
        'HT': 1.56556940e+00,
        'D': 1.30034172e-01
    },
    {
        'HT': 1.07868448e+01,
        'D': 7.45196760e-01
    },
    {
        'HT': 2.77001490e+01,
        'D': 1.96279065e+00
    },
    {
        'HT': 3.25257942e+01,
        'D': 2.33086919e+00
    },
    {
        'HT': 1.25886723e+02,
        'D': 4.19451985e+01
    },
]


hv_list = []
timelist = []
for idx, input_data in enumerate(input_parameters):
    t1 = time.time()
    output, trial_idx = ax_client.attach_trial(input_data)
    ax_client.complete_trial(trial_index=trial_idx, raw_data=result[idx])

    dummy_model = get_MOO_EHVI(
    experiment=ax_client.experiment, 
    data=ax_client.experiment.fetch_data(),
    )

    # hv = observed_hypervolume(modelbridge=dummy_model)
    
    try:
        hv = observed_hypervolume(modelbridge=dummy_model)
    except:
        hv = 0
        print("Failed to compute hv")
        
    hv_list.append(hv)
    spendtime = round(time.time()-t1, 2)
    timelist.append(spendtime)
    print(f"Iteration: {idx}, HV: {hv} || time per iteration: { spendtime}")



# print('visualize the results...')
# visualize_pareto(ax_client)
# my_cross_validate(ax_client)

# outcomes = np.array(exp_to_df(ax_client.experiment)[['HT', 'D']], dtype=np.double)
# print('outcomes', outcomes)

# plt.figure(figsize=(8,8))
# cm = plt.cm.get_cmap('viridis')
# plt.scatter(outcomes[:50,0], outcomes[:50,1], c = range(50))
# plt.xlabel("HT")
# plt.ylabel("D")
# plt.show()

parameters, trial_index = ax_client.get_next_trial()
print('trial_index', trial_index)
print(parameters)




