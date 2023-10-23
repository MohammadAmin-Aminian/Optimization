#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:08:28 2023

@author: mohammadamin
"""

import optuna
import inv_compy
import numpy as np

n_layer = 12
iteration = 10000
Alpha = 0

def seismic_inversion_objective(trial):
    # Define your seismic inversion and misfit calculation using trial.params
    Vs_Step = trial.suggest_int('Vs_Step', 10, 50)
    H_Step = trial.suggest_int('H_Step', 0, 30)
    Alpha = trial.suggest_float('Alpha', 0, 1)
    
    # Run the seismic inversion with the current hyperparameters and calculate misfit
    starting_model,vs,mis_fit,ncompl,likeli_hood,accept_rate = inv_compy.invert_compliace(Data,
                                                                              f,
                                                                              depth_s =4560,
                                                                              starting_model = None,
                                                                              n_layer= n_layer,
                                                                              sigma_v = Vs_Step,
                                                                              sigma_h = H_Step,
                                                                              iteration = iteration,
                                                                              alpha = Alpha)
    
    return np.min(mis_fit[0])

study = optuna.create_study(direction='minimize')
study.optimize(seismic_inversion_objective, n_trials=100)

# You should access the best parameters after the optimization process is completed.
best_Vs_Step = study.best_params['Vs_Step']
best_H_Step = study.best_params['H_Step']
best_Alpha = study.best_params['Alpha']
best_misfit = study.best_value

iteration = 200000
starting_model,vs,mis_fit,ncompl,likeli_hood,accept_rate = inv_compy.invert_compliace(Data,
                                                                              f,
                                                                              depth_s =4560,
                                                                              starting_model = None,
                                                                              n_layer= n_layer,
                                                                              sigma_v = best_Vs_Step,
                                                                              sigma_h = best_H_Step,
                                                                              iteration = iteration,
                                                                              alpha = best_Alpha)
    
burnin = 500
mis_fit_trsh = 4

inv_compy.plot_inversion(starting_model,
                         vs,mis_fit,
                         ncompl,
                         Data,
                         likelihood_data=likeli_hood,
                         freq=f,
                         sta=sta,
                         iteration=iteration,
                         s=np.std(Data)  / 2,
                         sigma_v = best_Vs_Step,
                         sigma_h = best_H_Step,
                         n_layer = n_layer,
                         alpha = best_Alpha, 
                         burnin = burnin,
                         mis_fit_trsh = mis_fit_trsh)

