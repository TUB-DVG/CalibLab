import os
os.chdir(os.path.join(os.getcwd(), 'src'))

import time
import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
import multiprocessing as mp
from functools import partial
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic
from sklearn.metrics import r2_score
try:
    import inputs 
except:
    import src.inputs as inputs
try:
    import paths 
except:
    import src.paths as paths

from SensitivityAnalysis import find_converged_sa_sample_size
from run_DIBS import model_run
from run_SA import run_SA
from run_GP_samples import sample_gp
from run_GP_train import train_gp
from BayesianCalibration import run_calibration
from multisimulationFunction import run_simulation
from GaussianProcesses import perform_gp_convergence



def calculate_cvrmse_r2(measured, simulated):
    """Calculate CV(RMSE) and R2 for a single simulation run."""
    n = len(measured)
    mean_measured = np.mean(measured)
    rmse = np.sqrt(np.mean((simulated - measured)**2))
    cvrmse = (rmse / mean_measured) * 100
    r2 = r2_score(measured, simulated)
    return cvrmse, r2

def main():
    start_process = time.time()

    ''' INPUTS '''
    scr_gebaeude_id = 30387001
    calib_type = "AMY"
    output_resolution = "W"
    climate_file = 'AMY_2010_2022.epw'

    ''' OPTIONAL INPUTS '''
    num_bc_param = 5
    RUN_SA_CONVERGENCE = 'Y'
    SA_sampling_lowerbound = 4
    SA_sampling_upperbound = 9
    RUN_GP_CONVERGENCE = 'Y'
    rmse_threshold = 0.0007
    gp_test_size = 0.25
    training_ratio = 75

    # BC parameters
    draws, tune, chains = 100, 200, 4
     # Number of parameter combinations for parallel simulation

    ''' IMPORTING DATA '''
    if RUN_SA_CONVERGENCE != 'Y':
        sa_converged = 0

    metered=pd.read_excel(os.path.join(paths.DATA_DIR, "HeatEnergyDemand_{}_{}.xlsx".format(scr_gebaeude_id, output_resolution)), index_col=0)
    nr_train_data = round(metered.shape[0]*training_ratio/100)
    start_time_cal, end_time_cal = metered.index[0].strftime('%Y-%m-%d %H:%M:%S'), metered.index[nr_train_data].strftime('%Y-%m-%d %H:%M:%S')
    start_time, end_time = metered.index[0].strftime('%Y-%m-%d %H:%M:%S'), metered.index[-1].strftime('%Y-%m-%d %H:%M:%S')



    ''' (1) Process Controll '''
    df = pd.DataFrame()
    df["Name"] = ["Part done", "Duration time in seconds", "Other info", "Other info", "Other info", "Other info", "Other info", "Other info"]
    df.to_excel(os.path.join(paths.CTRL_DIR, "process_intermittent_check_{}_{}_{}_{}.xlsx".format(scr_gebaeude_id, calib_type, output_resolution, str(training_ratio))))



    ''' (2) DIBS Simulation '''
    from run_DIBS_sim import model_run
    start_calc = time.time()
    HeatingEnergy_sum = model_run(scr_gebaeude_id, climate_file, start_time, end_time, output_resolution, training_ratio)     
    end_calc = time.time()

    if output_resolution == None:
        df["DIBS"] = ["DIBS simulation is working", end_calc-start_calc, "Uncalibrated sim result, kWh: ", "{}".format(HeatingEnergy_sum.tolist()), "-", "-", "-", "-"]
    else: 
        df["DIBS"] = ["DIBS simulation is working", end_calc-start_calc, "Uncalibrated sim result, kWh: ", "{}".format(HeatingEnergy_sum['HeatingEnergy'].values.tolist()), "-", "-", "-", "-"]
    df.to_excel(os.path.join(paths.CTRL_DIR, "process_intermittent_check_{}_{}_{}_{}.xlsx".format(scr_gebaeude_id, calib_type, output_resolution, str(training_ratio))), index=False)



    if RUN_SA_CONVERGENCE == 'Y':             
        num_samples_sa, sa_converged, sa_conv_time = find_converged_sa_sample_size(scr_gebaeude_id, calib_type, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio,  sampling_lower_bound = SA_sampling_lowerbound, sampling_upper_bound= SA_sampling_upperbound)
        df['SA'] = ['SA is done', 'Convergence time: {}'.format(sa_conv_time), "Num samples: {}".format(num_samples_sa), "-", "-", "-", "-", "-"]
        df.to_excel(os.path.join(paths.CTRL_DIR, "process_intermittent_check_{}_{}_{}_{}.xlsx".format(scr_gebaeude_id, calib_type, output_resolution, str(training_ratio))), index=False)
        
    else:
        num_samples_sa = 32
        total_SI, calc_time_SA = run_SA(scr_gebaeude_id, num_samples_sa, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio) 
        df['SA - {} train'.format(training_ratio)] = ['SA is done', calc_time_SA, "Num samples: {}".format(num_samples_sa), "-", "-", "-", "-", "-"]
        df.to_excel(os.path.join(paths.CTRL_DIR, "process_intermittent_check_{}_{}_{}_{}.xlsx".format(scr_gebaeude_id, calib_type, output_resolution, str(training_ratio))), index=False)



    ''' (4) Gaussian Regression'''
    if RUN_GP_CONVERGENCE == 'Y':
        best_result, best_kernel_path, best_samples_df_path, conv_time = perform_gp_convergence(
            scr_gebaeude_id=scr_gebaeude_id,
            climate_file=climate_file,
            output_resolution=output_resolution,
            calib_type=calib_type,
            min_gp_samples= 30,
            max_gp_samples= 400,
            num_bc_param=num_bc_param,
            num_samples_sa=num_samples_sa,
            step_gp=10,
            rmse_threshold=rmse_threshold,
            gp_test_size=gp_test_size,
            training_ratio = training_ratio)
        
        best_rmse_norm = best_result['RMSE_NORM']
        kernel_index = best_result['Kernel_Index']
        kernel = best_result['Kernel']
        num_samples_gp = best_result['Num_Samples_GP']

        df['GP samples'] = ['GP samples is done', conv_time, "num_samples_gp: {}".format(num_samples_gp), "num_bc_param: {}".format(num_bc_param), "num_samples_sa: {}".format(num_samples_sa), "climate_file: {}".format(climate_file), "-", "output_resolution: {}".format(output_resolution)]
        df.to_excel(os.path.join(paths.CTRL_DIR, "process_intermittent_check_{}_{}_{}_{}.xlsx".format(scr_gebaeude_id, calib_type, output_resolution, str(training_ratio))), index=False)


    else:
        num_samples_gp = 80
        kernel, kernel_index = 1 * RationalQuadratic(), 1

        from run_gp_samples import sample_gp
        start = time.time()
        samples_df, calc_time = sample_gp(scr_gebaeude_id, num_bc_param, num_samples_sa, num_samples_gp, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio)
        finish = time.time()

        df['GP samples'] = ['GP samples is done', finish-start, "num_samples_gp: {}".format(num_samples_gp), "num_bc_param: {}".format(num_bc_param), "num_samples_sa: {}".format(num_samples_sa), "climate_file: {}".format(climate_file), "start_time, end_time: {}, {}".format(start_time, end_time), "output_resolution: {}".format(output_resolution)]
        df.to_excel(os.path.join(paths.CTRL_DIR, "process_intermittent_check_{}_{}_{}_{}.xlsx".format(scr_gebaeude_id, calib_type, output_resolution, str(training_ratio))), index=False)
        
        from run_gp_train import train_gp
        start = time.time()
        kernel_trained, mse, mae, rmse, r2, sigma, y_test_mean, mape = train_gp(scr_gebaeude_id, num_bc_param, kernel, kernel_index, num_samples_gp, gp_test_size, output_resolution, training_ratio) 
        finish = time.time()

        df['GP train, training_ratio: {}, num_bc_param: {}'.format(training_ratio, num_bc_param)] = ['GP train is done', finish-start, "bc_param: {}, training_ratio: {}, samples: {}".format(num_bc_param, training_ratio, num_samples_gp), "Kernel: {}".format(kernel_trained), "Kernel index: {}, GP test size: {}".format(kernel_index, gp_test_size), "MAPE: {} / MSE: {} / RMSE: {} / MAE: {}".format(mape, mse, rmse, mae), "r2-score: {}".format(r2), "std of prediction: {}".format(sigma)]
        df.to_excel(os.path.join(paths.CTRL_DIR, "process_intermittent_check_{}_{}_{}_{}.xlsx".format(scr_gebaeude_id, calib_type, output_resolution, str(training_ratio))), index=False)
        


    ''' (5) Bayesian Calibration '''
    start = time.time()
    from run_bc_conv import run_calibration
    threshold = 1.01
    trace, converged, draws, tune, chains = run_calibration(scr_gebaeude_id, num_bc_param, draws, tune, chains, num_samples_gp, kernel_index, output_resolution, training_ratio, start_time_cal, end_time_cal, threshold)
    #run_calibration(scr_gebaeude_id, training_ratio, num_bc_param, draws, tune, chains, num_samples_gp, kernel_index, output_resolution = output_resolution, climate_file, threshold = threshold)
    if converged:
        pm.save_trace(trace, os.path.join(paths.RES_DIR, "calibration/{}/bc_{}_param_{}_obs_{}.trace".format(scr_gebaeude_id, num_bc_param, training_ratio, output_resolution)), overwrite=True)
    else:
        print("Did not converge for training_ratio =", training_ratio)
    finish = time.time()

    df['BC training_ratio: {}, num_bc_param: {}'.format(training_ratio, num_bc_param)] = ['BC is done', finish-start, "num bc param: {}".format(num_bc_param),"{} tune, {} draws, {} chains".format(tune, draws, chains), "building_id: {}".format(scr_gebaeude_id), "output resolution: {}".format(output_resolution), "-", "-"]
    df.to_excel(os.path.join(paths.CTRL_DIR, "process_intermittent_check_{}_{}_{}_{}.xlsx".format(scr_gebaeude_id, calib_type, output_resolution, str(training_ratio))), index=False)


    finish_process = time.time()
    print("The whole process took {} hours to run".format((finish_process-start_process)/3600))


    ''' (6) Running Parallel Simulations with Calibrated Values '''
    if converged:
        # Load calibration data
        parameter_combination = draws * chains 
        inference_data = az.from_netcdf(os.path.join(
            paths.RES_DIR,
            f"calibration/{scr_gebaeude_id}/{num_bc_param}_bc_param/{training_ratio}_obs_{output_resolution}_{draws}_{tune}_{chains}_area.nc"
        ))
        
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
            output_resolution=output_resolution
        )
        
        # Run parallel simulations
        print(f"Setting up multiprocessing with {mp.cpu_count()} processes...")
        start_parallel = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            print("Starting parallel simulations...")
            outputs = pool.map(run_simulation_partial, [row for _, row in df_params.iterrows()])
        end_parallel = time.time()
        
        df_params['HeatingEnergyPredicted'] = outputs
        
        # Save simulation results
        print("Saving results to Excel file...")
        df_params.to_excel(
            f"simulation_result_{scr_gebaeude_id}_numBC{num_bc_param}_Samples_{parameter_combination}_training_{training_ratio}_{output_resolution}_AshraeTrial.xlsx",
            index=False
        )
        
        # Calculate and save metrics
        measured_values = metered['Consumption'].values[1:]
        results = []
        
        for i, row in df_params.iterrows():
            simulated_values = np.array(row['HeatingEnergyPredicted'])
            if output_resolution == "W":
                simulated_values = simulated_values[:-1]
            
            min_length = min(len(simulated_values), len(measured_values))
            simulated_values = simulated_values[:min_length]
            measured_values_adjusted = measured_values[:min_length]
            
            cvrmse, r2 = calculate_cvrmse_r2(measured_values_adjusted, simulated_values)
            results.append({'Simulation': i+1, 'CV(RMSE)': cvrmse, 'R2': r2})
        
        # Save evaluation metrics
        results_df = pd.DataFrame(results)
        results_df.to_excel(
            f"evaluation_metrics_{scr_gebaeude_id}_numBC{num_bc_param}_Samples_{parameter_combination}_training_{training_ratio}_{output_resolution}_AshraeTrial.xlsx",
            index=False
        )
        
        # Calculate and save summary statistics
        summary_stats = results_df.describe()
        best_cvrmse = results_df.loc[results_df['CV(RMSE)'].idxmin()]
        best_r2 = results_df.loc[results_df['R2'].idxmax()]
        
        with open(f"summary_statistics_{scr_gebaeude_id}_numBC{num_bc_param}_Samples_{parameter_combination}_training_{training_ratio}_{output_resolution}_AshraeTrial.txt", 'w') as f:
            f.write("Summary Statistics for Evaluation Metrics:\n\n")
            f.write(summary_stats.to_string())
            f.write("\n\nBest Performing Simulations:\n")
            f.write(f"Lowest CV(RMSE): Simulation {best_cvrmse['Simulation']}, CV(RMSE) = {best_cvrmse['CV(RMSE)']:.2f}\n")
            f.write(f"Highest R2: Simulation {best_r2['Simulation']}, R2 = {best_r2['R2']:.4f}\n")
        
        df['Parallel Simulations'] = [
            'Simulations done', 
            end_parallel - start_parallel,
            f"Parameter combinations: {parameter_combination}",
            f"Best CV(RMSE): {best_cvrmse['CV(RMSE)']:.2f}",
            f"Best R2: {best_r2['R2']:.4f}",
            "-", "-", "-"
        ]
        df.to_excel(os.path.join(paths.CTRL_DIR, f"process_intermittent_check_{scr_gebaeude_id}_{calib_type}_{output_resolution}.xlsx"), index=False)

    finish_process = time.time()
    print(f"The whole process took {(finish_process-start_process)/3600:.2f} hours to run")

# Create final output file
    output_folder = os.path.join(paths.CTRL_DIR, "Output")
    os.makedirs(output_folder, exist_ok=True)
    file_name = f"building{scr_gebaeude_id}_Calibrating{num_bc_param}_TrainSplit_{training_ratio}_calibTyp_{calib_type}_Resolution_{output_resolution}.txt"
    file_path = os.path.join(output_folder, file_name)

    # Write parameters to the file
    with open(file_path, 'w') as f:
        f.write(f"scr_gebaeude_id: {scr_gebaeude_id}\n")
        f.write(f"num_bc_param: {num_bc_param}\n")
        f.write(f"training_ratio: {training_ratio}\n")
        f.write(f"SA_sampling_lowerbound: {SA_sampling_lowerbound}\n")
        f.write(f"SA_sampling_upperbound: {SA_sampling_upperbound}\n")
        f.write(f"RUN_GP_CONVERGENCE: {RUN_GP_CONVERGENCE}\n")
        f.write(f"RUN_SA_CONVERGENCE: {RUN_SA_CONVERGENCE}\n")
        f.write(f"rmseNorm_threshold for GP Convergence: {rmse_threshold}\n")
        f.write(f"gp_test_size: {gp_test_size}\n")
        f.write(f"calib_type: {calib_type}\n")
        f.write(f"climate_file: {climate_file}\n")
        f.write(f"output_resolution: {output_resolution}\n")
        f.write(f"num_samples_sa: {num_samples_sa}\n")
        f.write(f"sa_converged: {sa_converged}\n")
        f.write(f"num_samples_gp: {num_samples_gp}\n")
        f.write(f"kernel: {kernel}\n")
        f.write(f"kernel_index: {kernel_index}\n")
        f.write(f"gp_test_size: {gp_test_size}\n")
        f.write(f"Tune: {tune}\n")
        f.write(f"Draws: {draws}\n")

    print(f"Parameters have been written to {file_path}")

if __name__ == '__main__':
    mp.freeze_support()
    main()