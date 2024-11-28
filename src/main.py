import os
os.chdir(os.path.join(os.getcwd(), 'src'))

import time
import pandas as pd
import numpy as np
import arviz as az
import multiprocessing as mp
from functools import partial
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.metrics import r2_score
try:
    import inputs 
except:
    import src.inputs as inputs
try:
    import paths 
except:
    import src.paths as paths

from run_DIBS import run_model
from SensitivityAnalysis import find_converged_sa_sample_size
from run_SA import run_SA
from GaussianProcesses import perform_gp_convergence
from run_GP_samples import sample_gp
from run_GP_train import train_gp
from BayesianCalibration import run_calibration
from multisimulationFunction import run_simulation



def calculate_cvrmse_r2(measured, simulated):
    '''Calculate CV(RMSE) and R2 for a single simulation run.'''
    rmse = np.sqrt(np.mean((simulated - measured)**2))
    cvrmse = (rmse / np.mean(measured)) * 100
    r2 = r2_score(measured, simulated)
    return cvrmse, r2



def main():
    start_process = time.time()

    ''' INPUTS '''
    scr_gebaeude_id = 30387001          # Building ID
    calib_type = 'AMY'                  # AMY: Actual Meteorological Year, TRY: Test Reference Year (works only for Germany)
    output_resolution = 'M'             # Time resolution for the metered data and calibration: Y = Yearly, M = Monthly, W = Weekly, etc, None = for TRY version
    climate_file = 'AMY_2010_2022.epw'  # Name of the climate file



    ''' OPTIONAL INPUTS '''
    num_bc_param = 5                    # Number of model parameters to be calibrated
    SA_Convergence_Required = 'Y'       # Preference to run automatized convergence check for Sensitivity Analysis
    SA_sampling_lowerbound = 4          # Minimum number of samples (N) for Sensitivity Analysis ((N*(2D+2))). D=Dimensions, number of parameters
    SA_sampling_upperbound = 9          # Maximum number of samples for Sensitivity Analysis (This limit prevents excessively long runtimes if the sensitivity index parameter order does not converge)
    GP_Convergence_Required = 'Y'       # Preference to run automatized convergence check for meta-model creation
    min_gp_samples = 30                 # Minimum number of samples for meta-model training
    max_gp_samples = 400                # Maximum number of samples for meta-model training
    step_gp = 10                        # Increase in sample size for each new meta-model training
    rmse_threshold = 0.0007             # RMSE criteria for the meta-model
    gp_test_size = 25/100               # Proportion of samples for GP to be used for the evaluation of the meta-model
    training_ratio = 75                 # Percentage of observed data to be used for the calibration
    draws, tune, chains = 100, 200, 4   # Bayesian Calibration parameters



    ''' IMPORT DATA '''
    metered=pd.read_excel(os.path.join(paths.DATA_DIR, 'HeatEnergyDemand_{}_{}.xlsx'.format(scr_gebaeude_id, output_resolution)), index_col=0)
    nr_train_data = round(metered[1:].shape[0]*training_ratio/100)
    start_time_cal, end_time_cal = metered.index[0].strftime('%Y-%m-%d %H:%M:%S'), metered.index[nr_train_data].strftime('%Y-%m-%d %H:%M:%S')
    start_time, end_time = metered.index[0].strftime('%Y-%m-%d %H:%M:%S'), metered.index[-1].strftime('%Y-%m-%d %H:%M:%S')



    ''' (0) Process Controll File'''
    df = pd.DataFrame()
    df['Name'] = ['Step done', 'Duration time in seconds', 'Other info', 'Other info', 'Other info', 'Other info', 'Other info', 'Other info']
    ctrl_file = os.path.join(paths.CTRL_DIR, f'process_intermittent_check_{scr_gebaeude_id}_{calib_type}_{output_resolution}_{str(training_ratio)}.xlsx')
    df.to_excel(ctrl_file)




    ''' (1) DIBS Simulation '''
    start_calc = time.time()

    HeatingEnergy_sum = run_model(scr_gebaeude_id, 
                                  climate_file, 
                                  start_time, 
                                  end_time, 
                                  output_resolution, 
                                  training_ratio)   

    end_calc = time.time()
    if output_resolution == None:
        HeatingEnergy_sum = HeatingEnergy_sum.tolist()
    else: 
        HeatingEnergy_sum = HeatingEnergy_sum['HeatingEnergy'].values.tolist()
    df['DIBS'] = ['DIBS simulation is working', end_calc-start_calc, 'Uncalibrated sim result, kWh: ', f'{HeatingEnergy_sum}', '-', '-', '-', '-']
    df.to_excel(ctrl_file, index=False)




    ''' (2) Sensitivity Analysis'''
    if SA_Convergence_Required == 'Y':             
        num_samples_sa, sa_converged, calc_time_SA = find_converged_sa_sample_size(scr_gebaeude_id, 
                                                                                   calib_type, 
                                                                                   climate_file, 
                                                                                   start_time_cal, 
                                                                                   end_time_cal, 
                                                                                   output_resolution, 
                                                                                   training_ratio,  
                                                                                   SA_sampling_lowerbound, 
                                                                                   SA_sampling_upperbound)

        ''' No automatized convergence '''
    else:
        sa_converged = 0
        num_samples_sa = 32     # Default number of samples for determination of most sensible parameters.
        total_SI, calc_time_SA = run_SA(scr_gebaeude_id, num_samples_sa, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio) 
        
    df['SA'] = ['SA is done', f'Convergence time: {calc_time_SA}', f'Num samples: {num_samples_sa}', f'SA Converged: {sa_converged}', f'SA_Convergence_Required: {SA_Convergence_Required}', '-', '-', '-']
    df.to_excel(ctrl_file, index=False)




    ''' (4) Gaussian Regression -- Meta-model creation'''
    if GP_Convergence_Required == 'Y':
        best_result, best_kernel_path, best_samples_df_path, conv_time = perform_gp_convergence(scr_gebaeude_id, 
                                                                                                climate_file, 
                                                                                                output_resolution, 
                                                                                                calib_type, 
                                                                                                start_time_cal, 
                                                                                                end_time_cal,
                                                                                                min_gp_samples,
                                                                                                max_gp_samples, 
                                                                                                num_bc_param, 
                                                                                                num_samples_sa, 
                                                                                                step_gp, 
                                                                                                rmse_threshold, 
                                                                                                gp_test_size, 
                                                                                                training_ratio)
        best_rmse_norm = best_result['RMSE_NORM']
        kernel_index = best_result['Kernel_Index']
        kernel = best_result['Kernel']
        num_samples_gp = best_result['Num_Samples_GP']

        df['GP convergence'] = ['GP is done', conv_time, f'num_samples_gp: {num_samples_gp}', f'num_bc_param: {num_bc_param}', f'num_samples_sa: {num_samples_sa}', f'climate_file: {climate_file}', f'output_resolution: {output_resolution}', '-']
        df.to_excel(ctrl_file, index=False)

        ''' No automatized convergence '''
    else:
        num_samples_gp = 80     # Default number of samples for the training & testing of the meta-model.
        kernel, kernel_index = 1 * RationalQuadratic(), 1   # Default Kernel

        # Sampling for the Gaussian Processes
        start = time.time()
        samples_df, calc_time = sample_gp(scr_gebaeude_id, num_bc_param, num_samples_sa, num_samples_gp, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio)
        finish = time.time()
        df['GP sample'] = ['GP samples is done', finish-start, f'num_samples_gp: {num_samples_gp}', f'num_bc_param: {num_bc_param}', f'num_samples_sa: {num_samples_sa}', f'climate_file: {climate_file}', f'start_time, end_time: {start_time}, {end_time}', f'output_resolution: {output_resolution}']
        df.to_excel(ctrl_file, index=False)
        
        # Training the meta-model
        start = time.time()
        kernel_trained, mse, mae, rmse, r2, sigma, y_test_mean, mape = train_gp(scr_gebaeude_id, num_bc_param, kernel, kernel_index, num_samples_gp, gp_test_size, output_resolution, training_ratio) 
        finish = time.time()
        df['GP train'] = ['GP train is done', finish-start, f'bc_param: {num_bc_param}, training_ratio: {training_ratio}, samples: {num_samples_gp}', f'Kernel: {kernel_trained}', f'Kernel index: {kernel_index}, GP test size: {gp_test_size}', f'MAPE: {mape} / MSE: {mse} / RMSE: {rmse} / MAE: {mae}', f'r2-score: {r2}', f'std of prediction: {sigma}']
        df.to_excel(ctrl_file, index=False)
        



    ''' (5) Bayesian Calibration '''
    threshold = 1.01    # Gelman-Rubin convergence diagnostic criteria

    start = time.time()
    trace, converged, draws, tune, chains = run_calibration(scr_gebaeude_id, 
                                                            num_bc_param, 
                                                            draws, 
                                                            tune, 
                                                            chains, 
                                                            num_samples_gp, 
                                                            kernel_index, 
                                                            output_resolution, 
                                                            training_ratio, 
                                                            start_time_cal, 
                                                            end_time_cal, 
                                                            threshold)

    if converged:
        print(f'Bayesian calibration converged at {tune} tunes and {draws} draws')
    else:
        print(f'Did not converge for training_ratio = {training_ratio}')
    finish = time.time()

    df[f'BC training_ratio: {training_ratio}, num_bc_param: {num_bc_param}'] = ['BC is done', finish-start, f'num bc param: {num_bc_param}', f'{tune} tunes, {draws} draws, {chains} chains', f'output resolution: {output_resolution}','-','-','-']
    df.to_excel(ctrl_file, index=False)



    finish_process = time.time()
    print(f'The framework required {(finish_process-start_process)/3600} hours to complete.')




    ''' (6) Calibrated model output distribution '''
    if converged:
        # Load calibration results
        parameter_combination = draws * chains 
        inference_data = az.from_netcdf(os.path.join(paths.RES_DIR,f'calibration/{scr_gebaeude_id}/{num_bc_param}_bc_param/{training_ratio}_obs_{output_resolution}_{draws}_{tune}_{chains}.nc'))
        
        # Setup for parallel simulation
        variable_names = list(inference_data.posterior.data_vars.keys())[1:6]
        variables_df = inputs.get_building_inputdata(scr_gebaeude_id)
        be_data_original = inputs.setup_be_data_original(scr_gebaeude_id, variables_df)
        
        # Create parameter combinations DataFrame
        df_params = pd.DataFrame()
        for var in variable_names:
            df_params[var] = inference_data.posterior.data_vars[var].values.flatten()
        df_params = df_params.head(parameter_combination)
        
        # Setup parallel simulation
        run_simulation_partial = partial(
            run_simulation,
            scr_gebaeude_id=scr_gebaeude_id,
            be_data_original=be_data_original,
            variable_names=variable_names,
            climate_file=climate_file,
            start_time=start_time,
            end_time=end_time,
            output_resolution=output_resolution)
        
        # Run parallel simulations
        print(f'Setting up multiprocessing with {mp.cpu_count()} processes...')
        start_parallel = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            print('Starting parallel simulations...')
            outputs = pool.map(run_simulation_partial, [row for _, row in df_params.iterrows()])
        end_parallel = time.time()
        
        df_params['HeatingEnergyPredicted'] = outputs

        # Save simulation results
        print('Saving results to Excel file...')
        df_params.to_excel(f'CalibratedSimulationResults_{scr_gebaeude_id}_numBC_{num_bc_param}_samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx',index=False)
        
        # Calculate and save metrics
        measured_values = metered['Consumption'].values[1:]
        results = []
        for i, row in df_params.iterrows():
            simulated_values = np.array(row['HeatingEnergyPredicted'])
            if output_resolution == 'W':
                simulated_values = simulated_values[:-1]
            
            min_length = min(len(simulated_values), len(measured_values))
            simulated_values = simulated_values[:min_length]
            measured_values_adjusted = measured_values[:min_length]
            
            cvrmse, r2 = calculate_cvrmse_r2(measured_values_adjusted, simulated_values)
            results.append({'Simulation': i+1, 'CV(RMSE)': cvrmse, 'R2': r2})
        
        # Save evaluation metrics
        results_df = pd.DataFrame(results)
        results_df.to_excel(f'EvaluationMetrics_{scr_gebaeude_id}_numBC_{num_bc_param}_samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx',index=False)
        
        # Calculate and save summary statistics
        summary_stats = results_df.describe()
        best_cvrmse = results_df.loc[results_df['CV(RMSE)'].idxmin()]
        best_r2 = results_df.loc[results_df['R2'].idxmax()]
        
        with open(f'SummaryStatistics_{scr_gebaeude_id}_numBC_{num_bc_param}_samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.txt', 'w') as f:
            f.write('Summary Statistics for Evaluation Metrics:\n\n')
            f.write(summary_stats.to_string())
            f.write('\n\nBest Performing Simulations:\n')
            f.write(f'Lowest CV(RMSE): Simulation {best_cvrmse["Simulation"]}, CV(RMSE) = {best_cvrmse["CV(RMSE)"]:.2f}\n')
            f.write(f'Highest R2: Simulation {best_r2["Simulation"]}, R2 = {best_r2["R2"]:.4f}\n')
        
        df['Parallel Simulations'] = [
            'Simulations done', 
            end_parallel - start_parallel,
            f'Parameter combinations: {parameter_combination}',
            f'Best CV(RMSE): {best_cvrmse["CV(RMSE)"]:.2f}',
            f'Best R2: {best_r2["R2"]:.4f}',
            '-', '-', '-']
        df.to_excel(ctrl_file, index=False)

    # Create final output file
    output_folder = os.path.join(paths.CTRL_DIR, 'Output')
    os.makedirs(output_folder, exist_ok=True)
    file_name = f'Building{scr_gebaeude_id}_numBC_{num_bc_param}_TrainSplit_{training_ratio}_CalibType_{calib_type}_Resolution_{output_resolution}.txt'
    file_path = os.path.join(output_folder, file_name)

    # Write parameters to the file
    with open(file_path, 'w') as f:
        f.write(f'scr_gebaeude_id: {scr_gebaeude_id}\n')
        f.write(f'num_bc_param: {num_bc_param}\n')
        f.write(f'training_ratio: {training_ratio}\n')
        f.write(f'SA_sampling_lowerbound: {SA_sampling_lowerbound}\n')
        f.write(f'SA_sampling_upperbound: {SA_sampling_upperbound}\n')
        f.write(f'GP_Convergence_Required: {GP_Convergence_Required}\n')
        f.write(f'SA_Convergence_Required: {SA_Convergence_Required}\n')
        f.write(f'rmseNorm_threshold for GP Convergence: {rmse_threshold}\n')
        f.write(f'gp_test_size: {gp_test_size}\n')
        f.write(f'calib_type: {calib_type}\n')
        f.write(f'climate_file: {climate_file}\n')
        f.write(f'output_resolution: {output_resolution}\n')
        f.write(f'num_samples_sa: {num_samples_sa}\n')
        f.write(f'sa_converged: {sa_converged}\n')
        f.write(f'num_samples_gp: {num_samples_gp}\n')
        f.write(f'kernel: {kernel}\n')
        f.write(f'kernel_index: {kernel_index}\n')
        f.write(f'gp_test_size: {gp_test_size}\n')
        f.write(f'Tune: {tune}\n')
        f.write(f'Draws: {draws}\n')

    print(f'Parameters have been written to {file_path}')

if __name__ == '__main__':
    mp.freeze_support()
    main()