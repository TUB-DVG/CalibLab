import numpy as np
import pandas as pd
import arviz as az
import multiprocessing as mp
from functools import partial
from sklearn.metrics import r2_score
import time
import os
import plotly.graph_objects as go
from multisimulationFunction import run_simulation
import json

try:
    import inputs 
except:
    import src.inputs as inputs
try:
    import paths 
except:
    import src.paths as paths

def calculate_cvrmse_r2(measured, simulated):
    """
    Calculate CV(RMSE) and R2 for a single simulation run.
    
    :param measured: numpy array of measured values
    :param simulated: numpy array of simulated values
    :return: tuple of (CV(RMSE), R2)
    """
    mean_measured = np.mean(measured)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((simulated - measured)**2))
    
    # Calculate CV(RMSE)
    cvrmse = (rmse / mean_measured) * 100
    
    # Calculate R2
    r2 = r2_score(measured, simulated)
    
    return cvrmse, r2



def run_parallel_simulations(scr_gebaeude_id, num_bc_param, draws, tune, chains, 
                           output_resolution, training_ratio, start_time, end_time, 
                           climate_file, ctrl_file):
    """
    Run parallel simulations using calibrated parameters and evaluate results
    
    Returns:
    --------
    tuple
        DataFrames containing simulation results and evaluation metrics
    """
    try:
        # Calculate total parameter combinations
        parameter_combination = draws * chains 
        
        # Create directory structure for results
        results_base_dir = os.path.join(paths.RES_DIR, 'calibrated_model_simulation')
        building_results_dir = os.path.join(results_base_dir, f'{scr_gebaeude_id}_{output_resolution}')
        os.makedirs(building_results_dir, exist_ok=True)
        
        # Load calibration results
        inference_data = az.from_netcdf(os.path.join(
            paths.RES_DIR,
            f'calibration/{scr_gebaeude_id}/{num_bc_param}_bc_param/{training_ratio}_obs_{output_resolution}_{draws}_{tune}_{chains}.nc'))
        
        # Setup for parallel simulation
        variable_names = list(inference_data.posterior.data_vars.keys())[1:6]
        variables_df = inputs.get_building_inputdata(scr_gebaeude_id)
        be_data_original = inputs.setup_be_data_original(scr_gebaeude_id, variables_df)
        
        # Create parameter combinations DataFrame
        df_params = pd.DataFrame()
        for var in variable_names:
            df_params[var] = inference_data.posterior.data_vars[var].values.flatten()
        df_params = df_params.head(parameter_combination)
        
        # Setup parallel simulation function
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
        
        # Add predictions to parameters DataFrame
        df_params['HeatingEnergyPredicted'] = outputs
        
        # Save simulation results
        results_filename = os.path.join(building_results_dir, f'CalibratedSimulationResults_{scr_gebaeude_id}_numBC_{num_bc_param}_samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx')
        df_params.to_excel(results_filename, index=False)
        
        # Calculate metrics
        metered = pd.read_excel(os.path.join(paths.DATA_DIR, f'HeatEnergyDemand_{scr_gebaeude_id}_{output_resolution}.xlsx'))

        measured_values = metered['Consumption'].values[1:]
        results = []
        
        for i, row in df_params.iterrows():
            simulated_values = np.array(row['HeatingEnergyPredicted'])
            
            min_length = min(len(simulated_values), len(measured_values))
            simulated_values = simulated_values[:min_length]
            measured_values_adjusted = measured_values[:min_length]
            
            cvrmse, r2 = calculate_cvrmse_r2(measured_values_adjusted, simulated_values)
            results.append({'Simulation': i+1, 'CV(RMSE)': cvrmse, 'R2': r2})
        
        # Create and save evaluation metrics
        results_df = pd.DataFrame(results)

        metrics_filename = os.path.join(building_results_dir, f'EvaluationMetrics_{scr_gebaeude_id}_numBC_{num_bc_param}_samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx')
        
        results_df.to_excel(metrics_filename, index=False)
        
        # Calculate and save summary statistics
        summary_stats = results_df.describe()
        best_cvrmse = results_df.loc[results_df['CV(RMSE)'].idxmin()]
        best_r2 = results_df.loc[results_df['R2'].idxmax()]
        
        summary_filename = os.path.join(
            building_results_dir,
            f'SummaryStatistics_{scr_gebaeude_id}_numBC_{num_bc_param}_'
            f'samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.txt')
        
        with open(summary_filename, 'w') as f:
            f.write('Summary Statistics for Evaluation Metrics:\n\n')
            f.write(summary_stats.to_string())
            f.write('\n\nBest Performing Simulations:\n')
            f.write(f'Lowest CV(RMSE): Simulation {best_cvrmse["Simulation"]}, CV(RMSE) = {best_cvrmse["CV(RMSE)"]:.2f}\n')
            f.write(f'Highest R2: Simulation {best_r2["Simulation"]}, R2 = {best_r2["R2"]:.4f}\n')
        
        # Update control file
        df = pd.read_excel(ctrl_file)
        df['Parallel Simulations'] = [
            'Simulations done', 
            end_parallel - start_parallel,
            f'Parameter combinations: {parameter_combination}',
            f'Best CV(RMSE): {best_cvrmse["CV(RMSE)"]:.2f}',
            f'Best R2: {best_r2["R2"]:.4f}',
            '-', '-', '-']
        
        df.to_excel(ctrl_file, index=False)
        
        print(f"All results saved. Best CV(RMSE): {best_cvrmse['CV(RMSE)']:.2f}, Best R2: {best_r2['R2']:.4f}")
        
        return df_params, results_df
        
    except Exception as e:
        print(f"Error in parallel simulation: {str(e)}")
        raise



# Example usage:
if __name__ == "__main__":
    df_params, results_df = run_parallel_simulations(
        scr_gebaeude_id=300387001,
        num_bc_param=5,
        draws=1000,
        tune=2000,
        chains=4,
        output_resolution="W",
        training_ratio=0.8,
        start_time="2020-01-01",
        end_time="2021-12-31",
        climate_file="path_to_climate_file",
        ctrl_file="path_to_control_file")
    
    violin_fig.show()