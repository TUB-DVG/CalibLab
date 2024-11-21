import numpy as np
import pymc3 as pm
import pandas as pd
import arviz as az
from datetime import datetime
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
import matplotlib.pyplot as plt
def get_building_area(scr_gebaeude_id):
    """
    Get the building area (ebf) from variables_df.xlsx for a specific building
    """
    fixed_variables = pd.read_excel(os.path.join(paths.DATA_DIR, 'variables_df.xlsx'))
    area = fixed_variables.loc[fixed_variables['scr_gebaeude_id'] == scr_gebaeude_id, 'ebf'].values[0]
    return area
def run_calibration(scr_gebaeude_id, num_bc_param, draws, tune, chains, num_samples_gp, kernel_index, output_resolution, training_ratio, start_time_cal, end_time_cal, threshold):

    dir_path = os.path.join(paths.RES_DIR, 'calibration', str(scr_gebaeude_id), '{}_bc_param'.format(num_bc_param))
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        # Another process in the multiprocessing may have created the directory already; just pass
        pass

    # Fetch parameter values which will be used for the prior probability distributions
    param_names_list, prior_params_mean, prior_params_lower, prior_params_upper, prior_params_sigma = inputs.get_prior_param_values(scr_gebaeude_id, num_bc_param, num_samples_gp, output_resolution, training_ratio)
    prior_prob = pd.read_excel(os.path.join(paths.DATA_DIR, 'prior_probability_definition.xlsx'), index_col=0)

    observations = pd.read_excel(os.path.join(paths.DATA_DIR, 'HeatEnergyDemand_{}_{}.xlsx'.format(scr_gebaeude_id, output_resolution)), index_col=0)
    observations = observations.loc[:end_time_cal]
    observations = observations['Consumption'].tolist()[1:]

    # Open trained GP-metamodel
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

            model_discrepancy = pm.Normal('model_discrepancy', mu=0, sigma=sigma_gp)
            error_obs = pm.Exponential('error_obs', lam=10/np.mean(observations)) 

            if num_bc_param == 2:
                sim_result = pm.Deterministic('sim_result', exterior_model(list_var_distr[0], list_var_distr[1]) + model_discrepancy) 

            if num_bc_param == 3:
                sim_result = pm.Deterministic('sim_result', exterior_model(list_var_distr[0], list_var_distr[1], list_var_distr[2]) + model_discrepancy) 

            if num_bc_param == 4:
                sim_result = pm.Deterministic('sim_result', exterior_model(list_var_distr[0], list_var_distr[1], list_var_distr[2], list_var_distr[3]) + model_discrepancy) 

            if num_bc_param == 5:
                sim_result = pm.Deterministic('sim_result', exterior_model(list_var_distr[0], list_var_distr[1], list_var_distr[2], list_var_distr[3], list_var_distr[4]) + model_discrepancy) 

            if num_bc_param == 6:
                sim_result = pm.Deterministic('sim_result', exterior_model(list_var_distr[0], list_var_distr[1], list_var_distr[2], list_var_distr[3], list_var_distr[4], list_var_distr[5]) + model_discrepancy) 

            #Likelihood: Probability of the simulation result, given the observation
            likelihood = pm.Normal('likelihood', mu=sim_result, sigma=error_obs, observed=observations)

            trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=46, progressbar=True)

            # Fetch summary statistics, including Rhat values
            trace_summary = az.summary(trace)
            rhat_scores = trace_summary['r_hat'].values[:6]  # Check the rhat values of the first 6 parameters
            
            if all(rhat <= threshold for rhat in rhat_scores):
                print('Converged with R-hat values:', rhat_scores)
                # Saving trace as net_cdf
                az.to_netcdf(trace, os.path.join(paths.RES_DIR, f'calibration/{scr_gebaeude_id}/{num_bc_param}_bc_param/{training_ratio}_obs_{output_resolution}_{draws}_{tune}_{chains}.nc'))
                return trace, True, draws, tune, chains  # Converged
        
            else:
                print('Number of draws: {}, Number of tunes: {}'.format(draws, tune))
                print('R-hat scores:', rhat_scores)
                print('Sampling again as R-hat is greater than or equal to', threshold)
                draws = draws*2     # Increase draws for next sampling
                tune = draws*2     # Increase tune for next sampling it was 2 earlier
                print('Increasing the number of draws: {}, and number of tunes: {}'.format(draws, tune))

def plot_calibration_results(scr_gebaeude_id, num_bc_param, output_resolution, training_ratio, draws, tune, chains):
    """
    Create plots from saved trace file with dynamic variable name mapping
    """
    # Complete name mapping dictionary
    name_mapping = {
        'q25_1': 'Maximale Belegung',
        'aw_fl': 'Aussenwandfläche über Gelände',
        'qd1': 'Flächenverhältnis Fenster/Wand',
        'facade_area': 'Fläche der Fassade',
        'geb_f_flaeche_n_iwu': 'Nordfassade',
        'd_fl_be': 'Dachfläche',
        'nrf_2': 'Nettoraumfläche',
        'ebf': 'Energiebezugsfläche',
        'n_og': 'Anzahl der Geschosse ohne Keller',
        'geb_f_hoehe_mittel_iwu': 'Höhe',
        'u_fen': 'U-Wert Fenster',
        'u_aw': 'U-Wert Aussenwand',
        'd_u_ges': 'U-Wert Dach',
        'u_ug': 'U-Wert Bodenplatte',
        'heat_recovery_efficiency': 'Wärmerückgewinnnung der RLT',
        'thermal_capacitance': 'Wärmekapazität',
        'heating_coefficient': 'Heizeffizienz',
        'glass_solar_transmittance': 'Tranmissionsfaktor Glas',
        'qd8': 'Lichttransmissionsfaktor',
        'p_j_lx': 'Beleuchtungsdichte',
        'k_L': 'Faktor der Lichtbereitstellung',
        'model_discrepancy': 'Modelldiskrepanz',
        'error_obs': 'Messfehler',
        'sim_result': 'Simulationsergebniss'
    }

    # Get building-specific area
    fixed_variables = pd.read_excel(os.path.join(paths.DATA_DIR, 'variables_df.xlsx'))
    area = fixed_variables.loc[fixed_variables['scr_gebaeude_id'] == scr_gebaeude_id, 'ebf'].values[0]
    
    # Load the saved trace
    trace_path = os.path.join(paths.RES_DIR, 
                             'calibration', 
                             str(scr_gebaeude_id), 
                             f'{num_bc_param}_bc_param',
                             f'{training_ratio}_obs_{output_resolution}_{draws}_{tune}_{chains}.nc')
    
    print(f"Reading trace from: {trace_path}")
    inference_data = az.from_netcdf(trace_path)
    
    # Create plots directory
    plots_dir = os.path.join(paths.RES_DIR, 
                            'calibration', 
                            str(scr_gebaeude_id), 
                            f'{num_bc_param}_bc_param', 
                            'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Normalize data
    normalized_data = inference_data.copy()
    normalized_data.posterior['error_obs'] = normalized_data.posterior['error_obs'] / area
    for var in normalized_data.posterior.data_vars:
        if var.startswith('sim_result'):
            normalized_data.posterior[var] = normalized_data.posterior[var] / area
    
    # Get available variables in the trace and create dynamic mapping
    available_vars = list(normalized_data.posterior.data_vars)
    trace_name_mapping = {var: name_mapping[var] for var in available_vars if var in name_mapping}
    
    print("Variables in trace:", available_vars)
    print("Applied name mapping:", trace_name_mapping)
    
    # Create renamed version for plotting
    inference_data_renamed = normalized_data.copy()
    new_posterior = normalized_data.posterior.rename(trace_name_mapping)
    inference_data_renamed.posterior = new_posterior
    
    # Generate plots
    n_vars = len(inference_data_renamed.posterior.data_vars)
    
    print("Creating trace plot...")
    # Trace plot
    plt.figure(figsize=(n_vars * 4, 4))
    az.plot_trace(inference_data_renamed)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'trace_plot_{training_ratio}_{draws}_{tune}_{chains}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("Generating summary statistics...")
    # Summary statistics
    summary = az.summary(normalized_data)
    summary_path = os.path.join(plots_dir, f'summary_{training_ratio}_{draws}_{tune}_{chains}.csv')
    summary.to_csv(summary_path)
    
    print(f"Plots and summary saved in: {plots_dir}")
    return summary

# Example usage
if __name__ == "__main__":
    # First run calibration
    trace, converged, draws, tune, chains = run_calibration(
        scr_gebaeude_id=30034001,
        num_bc_param=5,
        draws=1000,
        tune=1000,
        chains=4,
        num_samples_gp=300,
        kernel_index=1,
        output_resolution=None,
        training_ratio=0.75,
        start_time_cal="2020-01-01 00:00:00",
        end_time_cal="2021-12-31 23:59:59",
        threshold=1.1
    )
    
    # Then create plots
    if converged:
        summary = plot_calibration_results(
            scr_gebaeude_id=30034001,
            num_bc_param=5,
            output_resolution="None",
            training_ratio=0.8,
            draws=draws,
            tune=tune,
            chains=chains
        )