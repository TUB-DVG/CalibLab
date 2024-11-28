import numpy as np
import pandas as pd
import arviz as az
import multiprocessing as mp
from functools import partial
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
def run_parallel_simulations(scr_gebaeude_id, num_bc_param, draws, tune, chains, 
                           output_resolution, training_ratio, start_time, end_time, 
                           climate_file, ctrl_file):
    """
    Run parallel simulations using calibrated parameters and evaluate results
    
    Parameters:
    -----------
    scr_gebaeude_id : str or int
        Building identifier
    num_bc_param : int
        Number of BC parameters
    draws : int
        Number of draws used in calibration
    tune : int
        Number of tuning steps used
    chains : int
        Number of chains used
    output_resolution : str
        Output resolution (e.g., 'W' for weekly)
    training_ratio : float
        Training ratio used
    start_time : str
        Start time for simulation
    end_time : str
        End time for simulation
    climate_file : str
        Path to climate file
    ctrl_file : str
        Path to control file for logging
        
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
            f'calibration/{scr_gebaeude_id}/{num_bc_param}_bc_param/{training_ratio}_obs_{output_resolution}_{draws}_{tune}_{chains}.nc'
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
        
        # Setup parallel simulation function
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
        print(f'Setting up multiprocessing with {mp.cpu_count()} processes...')
        start_parallel = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            print('Starting parallel simulations...')
            outputs = pool.map(run_simulation_partial, [row for _, row in df_params.iterrows()])
        end_parallel = time.time()
        
        # Add predictions to parameters DataFrame
        df_params['HeatingEnergyPredicted'] = outputs
        
        # Save simulation results
        # results_filename = (f'CalibratedSimulationResults_{scr_gebaeude_id}_numBC_{num_bc_param}_'
        #                   f'samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx')
        results_filename = os.path.join(
            building_results_dir,
            f'CalibratedSimulationResults_{scr_gebaeude_id}_numBC_{num_bc_param}_'
            f'samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx'
        )
        df_params.to_excel(results_filename, index=False)
        
        # Calculate metrics
        metered = pd.read_excel(os.path.join(paths.DATA_DIR, 
                                            f'HeatEnergyDemand_{scr_gebaeude_id}_{output_resolution}.xlsx'))
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
        
        # Create and save evaluation metrics
        results_df = pd.DataFrame(results)
        # metrics_filename = (f'EvaluationMetrics_{scr_gebaeude_id}_numBC_{num_bc_param}_'
        #                   f'samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx')
        metrics_filename = os.path.join(
            building_results_dir,
            f'EvaluationMetrics_{scr_gebaeude_id}_numBC_{num_bc_param}_'
            f'samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx'
        )
        results_df.to_excel(metrics_filename, index=False)
        
        # Calculate and save summary statistics
        summary_stats = results_df.describe()
        best_cvrmse = results_df.loc[results_df['CV(RMSE)'].idxmin()]
        best_r2 = results_df.loc[results_df['R2'].idxmax()]
        
        # summary_filename = (f'SummaryStatistics_{scr_gebaeude_id}_numBC_{num_bc_param}_'
        #                   f'samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.txt')
                # Save summary statistics with new path
        summary_filename = os.path.join(
            building_results_dir,
            f'SummaryStatistics_{scr_gebaeude_id}_numBC_{num_bc_param}_'
            f'samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.txt'
        )
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
            '-', '-', '-'
        ]
        df.to_excel(ctrl_file, index=False)
        
        print(f"All results saved. Best CV(RMSE): {best_cvrmse['CV(RMSE)']:.2f}, Best R2: {best_r2['R2']:.4f}")
        return df_params, results_df
        
    except Exception as e:
        print(f"Error in parallel simulation: {str(e)}")
        raise
def create_violin_plot(scr_gebaeude_id, num_bc_param, output_resolution, training_ratio, draws, tune, start_time, end_time, calib_type="TRY"):
    """
    Create yearly violin plot using the correct file naming convention
    
    Parameters:
    -----------
    scr_gebaeude_id : str or int
        Building identifier
    num_bc_param : int
        Number of BC parameters
    samples : int
        Number of samples
    output_resolution : str
        Output resolution (e.g., 'Y' for yearly)
    training_ratio : int/float
        Training ratio (e.g., 60, 75)
    calib_type : str, optional
        Calibration type (default: "AMY")
    """
    # Construct results filename
    parameter_combination = draws*tune
    
    results_base_dir = os.path.join(paths.RES_DIR, 'calibrated_model_simulation')
    building_results_dir = os.path.join(results_base_dir, f'{scr_gebaeude_id}_{output_resolution}')
    
    # Construct results filename
    results_file_name = (f'CalibratedSimulationResults_{scr_gebaeude_id}_numBC_{num_bc_param}_'
                        f'samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx')
    #results_file_name = (f'CalibratedSimulationResults_{scr_gebaeude_id}_numBC_{num_bc_param}_samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx')
   
    print(f"Looking for results file: {results_file_name}")
    
    # Load data
    metered = pd.read_excel(os.path.join(paths.DATA_DIR, 
                                        f"HeatEnergyDemand_{scr_gebaeude_id}_{output_resolution}.xlsx"), 
                           index_col=0)
    
    # Get start and end times from metered data
    start_time = metered.index[0].strftime('%Y-%m-%d %H:%M:%S')
    end_time = metered.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    file_path = os.path.join(building_results_dir, results_file_name)
    # file_path = os.path.join(os.path.dirname(os.getcwd()), 
    #                         f'results/report/{scr_gebaeude_id}_{output_resolution}')
    posterior_sim = pd.read_excel(file_path)
    #posterior_sim = pd.read_excel(os.path.join(file_path, results_file_name))

    # Extract years from metered data
    start_year = metered.index[0].year
    end_year = metered.index[-1].year + 1
    years = list(range(start_year, end_year))
    
    print(f"Creating violin plot for years {start_year} to {end_year-1}")

    # Load DIBS output
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    DIBS_output = pd.read_csv(os.path.join(paths.RES_DIR, 
                                          f"DIBS_sim/{scr_gebaeude_id}/{output_resolution}/{training_ratio}/HeatingEnergy_"
                                          f"{start.year}-{end.month}-{end.day}_{end.year}-{end.month}-{end.day}.csv"), 
                             sep=';', index_col=0)
    DIBS_output.index = pd.to_datetime(DIBS_output.index) + pd.DateOffset(hours=23)

    # Get building area
    area = inputs.get_building_inputdata(scr_gebaeude_id)['ebf'].iloc[0]

    # Process yearly data with json loading
    posterior_sim['HeatingEnergyPredicted'] = posterior_sim['HeatingEnergyPredicted'].apply(json.loads)
    yearly_data = posterior_sim['HeatingEnergyPredicted'].tolist()
    data_per_year = {year: [] for year in years}

    # Fill the data_per_year dictionary
    for values in yearly_data:
        for year_index, year in enumerate(years):
            data_per_year[year].append(values[year_index])

    # Create figure
    fig = go.Figure()

    # Add violin plots
    for year in years:
        fig.add_trace(go.Violin(
            y=data_per_year[year]/area,
            name=str(year),
            box_visible=True,
            meanline_visible=True,
            points=False,
            line_color='#4F81BD',
            marker=dict(size=3),
            hoverinfo='y',
            side='negative',
            spanmode='soft',
            jitter=0.05,
            scalemode='count',
            width=1,
            legendgroup='Kalibriertes Modell',
            showlegend=True,
            legendgrouptitle_text="Kalibriertes Modell"
        ))

    # Add metered data
    metered_values = metered[metered.index.year.isin(years)].values.flatten()
    metered_years = metered[metered.index.year.isin(years)].index.year
    fig.add_trace(go.Scatter(
        x=metered_years,
        y=metered_values/area,
        mode='markers+lines',
        name='Gemessen',
        marker=dict(color='black', size=12, symbol='circle'),
        line=dict(width=2)
    ))

    # Add DIBS output
    dibs_years = DIBS_output.index.year
    dibs_values = DIBS_output['HeatingEnergy'].values
    fig.add_trace(go.Scatter(
        x=dibs_years,
        y=dibs_values/area,
        mode='markers+lines',
        name='DIBS prior Kalibrierung',
        marker=dict(color='red', size=12, symbol='cross'),
        line=dict(width=2)
    ))

    # Update layout
    fontstyle = dict(family="Arial", size=40)
    fig.update_layout(
        title=dict(
            text="Vorhersage Wärmeenergieverbrauch mit kalibriertem Modell",
            font=fontstyle,
            x=0.5,
            y=0.95
        ),
        xaxis_title=dict(text="Jahr", font=fontstyle),
        yaxis_title=dict(text='<b>Wärmeenergieverbrauch</b> (kWh/m²)', font=fontstyle),
        violingap=0.1,
        violingroupgap=0,
        plot_bgcolor='white',
        font=fontstyle,
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            linecolor='black',
            tickfont=fontstyle,
            titlefont=fontstyle
        ),
        yaxis=dict(
            range=[28, 129],
            showgrid=True,
            zeroline=False,
            linecolor='black',
            tickfont=fontstyle,
            titlefont=fontstyle
        )
    )

    return fig

# Example usage:


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
        ctrl_file="path_to_control_file"
    )
    
    violin_fig = create_violin_plot(scr_gebaeude_id=300387001 , num_bc_param = 5, output_resolution = 'Y', 
                                training_ratio = 0.8, draws= 1000, tune=2000, start_time = "2020-01-01", end_time = "2021-12-31", calib_type= 'TRY') 
    violin_fig.show()