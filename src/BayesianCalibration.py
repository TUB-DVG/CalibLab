import numpy as np
import pymc3 as pm
import pandas as pd
import os
import theano
import theano.tensor as t
try:
    import inputs 
except:
    import src.inputs as inputs
try:
    import paths 
except:
    import src.paths as paths
import pickle
import arviz as az
import matplotlib.pyplot as plt
import arviz as az
from datetime import datetime


def run_calibration(scr_gebaeude_id, num_bc_param, draws, tune, chains, num_samples_gp, kernel_index, output_resolution, training_ratio, start_time_cal, end_time_cal, threshold):

    dir_path = os.path.join(paths.RES_DIR, "calibration", str(scr_gebaeude_id), "{}_bc_param".format(num_bc_param))

    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        # Another process in the multiprocessing may have created the directory already; just pass
        pass

    # Get the parameter values which will be used for the prior probability distributions
    param_names_list, prior_params_mean, prior_params_lower, prior_params_upper, prior_params_sigma = inputs.get_prior_param_values(scr_gebaeude_id, num_bc_param, num_samples_gp, output_resolution, training_ratio)
    prior_prob = pd.read_excel(os.path.join(paths.DATA_DIR, 'prior_probability_definition.xlsx'), index_col=0)

    observations = pd.read_excel(os.path.join(paths.DATA_DIR, 'HeatEnergyDemand_{}_{}.xlsx'.format(scr_gebaeude_id, output_resolution)), index_col=0)
    observations = observations.loc[:end_time_cal]
    observations = observations['Consumption'].tolist()[1:]

    # Get trained GP-metamodel
    gaussian_process, scaler_X, X_train, X_test, y_train, y_test, y_pred, sigma_gp, mse, mae, rmse, r2, mape = inputs.gp_metamodel(scr_gebaeude_id, num_bc_param, num_samples_gp, output_resolution, training_ratio, kernel_index)

    if output_resolution == None:
        start_time = datetime.strptime(start_time_cal, '%Y-%m-%d %H:%M:%S').year
        end_time = datetime.strptime(end_time_cal, '%Y-%m-%d %H:%M:%S').year

        Climate_Factor = inputs.CF_factors(scr_gebaeude_id, start_time, end_time)
    

    @theano.compile.ops.as_op(itypes=[t.dscalar]*num_bc_param, otypes=[t.dvector])
    def exterior_model(*parameters):     

        X = np.array([*parameters])
        X = X.reshape(1,-1)

        X_scaled = scaler_X.transform(X)
        Y = gaussian_process.predict(X_scaled)
        mean_prediction = Y[0] 


        if output_resolution == None:
            mean_prediction = np.divide(mean_prediction, Climate_Factor)
        else:
            mean_prediction=mean_prediction

        return mean_prediction

 
    while True:
        with pm.Model() as model:
            #Priors:
            list_var_distr = []
            for var in param_names_list:
                if prior_prob.loc['{}'.format(var), 'bc'] == 'Normal':
                    list_var_distr.append(pm.TruncatedNormal(var, mu=prior_params_mean[var], sigma=prior_params_sigma[var], lower=prior_params_lower[var], upper=prior_params_upper[var]))

                if prior_prob.loc['{}'.format(var), 'bc'] == 'Uniform':
                    list_var_distr.append(pm.Uniform(var, lower=prior_params_lower[var], upper=prior_params_upper[var]))

            model_discrepancy = pm.Normal("model_discrepancy", mu=0, sigma=sigma_gp)
            error_obs = pm.Exponential("error_obs", lam=10/np.mean(observations)) 

            if num_bc_param == 2:
                sim_result = pm.Deterministic("sim_result", exterior_model(list_var_distr[0], list_var_distr[1]) + model_discrepancy) 

            if num_bc_param == 3:
                sim_result = pm.Deterministic("sim_result", exterior_model(list_var_distr[0], list_var_distr[1], list_var_distr[2]) + model_discrepancy) 

            if num_bc_param == 4:
                sim_result = pm.Deterministic("sim_result", exterior_model(list_var_distr[0], list_var_distr[1], list_var_distr[2], list_var_distr[3]) + model_discrepancy) 

            if num_bc_param == 5:
                sim_result = pm.Deterministic("sim_result", exterior_model(list_var_distr[0], list_var_distr[1], list_var_distr[2], list_var_distr[3], list_var_distr[4]) + model_discrepancy) 

            if num_bc_param == 6:
                sim_result = pm.Deterministic("sim_result", exterior_model(list_var_distr[0], list_var_distr[1], list_var_distr[2], list_var_distr[3], list_var_distr[4], list_var_distr[5]) + model_discrepancy) 

            #Likelihood: The probability of the simulation result = observation
            likelihood = pm.Normal("likelihood", mu=sim_result, sigma=error_obs, observed=observations)

            trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=46, progressbar=True) #, nuts={'target_accept':0.9}, max_treedepth=15, target_accept=0.8)

            # Get the summary statistics including Rhat values
            # trace_summary = pm.summary(trace)
            trace_summary = az.summary(trace)
            rhat_scores = trace_summary["r_hat"].values[:6]  # Check the rhat values of the first 6 parameters
            
            if all(rhat <= threshold for rhat in rhat_scores):
                print("Converged with R-hat values:", rhat_scores)
                # Saving trace as pymc3
                # pm.save_trace(trace, os.path.join(paths.RES_DIR, "calibration/{}/{}_bc_param/{}_obs_{}_{}_{}_{}.trace".format(scr_gebaeude_id, num_bc_param, num_obs_train, output_resolution, draws, tune, chains)), overwrite=True)
                # Saving trace as net_cdf
                az.to_netcdf(trace, os.path.join(paths.RES_DIR, "calibration/{}/{}_bc_param/{}_obs_{}_{}_{}_{}_area.nc".format(scr_gebaeude_id, num_bc_param, training_ratio, output_resolution, draws, tune, chains)))
                return trace, True, draws, tune, chains  # Converged
        
            else:
                print('Number of draws: {}, Number of tunes: {}'.format(draws, tune))
                print("R-hat scores:", rhat_scores)
                print("Sampling again as R-hat is greater than or equal to", threshold)
                draws = draws*2  # Increase draws for next sampling
                tune = draws*2 #int(tune * 0.1)  # Optionally increase tuning steps as well to ensure better convergence
                print('Increasing the number of draws: {}, and number of tunes: {}'.format(draws, tune))