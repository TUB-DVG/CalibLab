import os
import pandas as pd
import numpy as np
import arviz as az
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import pymc3 as pm
import seaborn as sns
import pickle
try:
    import inputs 
except:
    import src.inputs as inputs
try:
    import paths 
except:
    import src.paths as paths


plots_folder_path = os.path.join(paths.PLOTS_DIR)
size_legend = 14
size_ticks = 12
size_labels = 14
size_title = 18
figsize = (8, 4)
font_family = 'Calibri'




def Plot_Model(scr_gebaeude_id, output_resolution, training_ratio):
    ''' OPEN DATA '''
    # Read in the metered data
    metered = pd.read_excel(os.path.join(paths.DATA_DIR, "HeatEnergyDemand_{}_{}.xlsx".format(scr_gebaeude_id, output_resolution)), index_col=0)
    start_time, end_time = metered.index[1].strftime('%Y-%m-%d %H:%M:%S'), metered.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    # Read in the simulated data
    HeatingEnergy_sum = pd.read_csv(os.path.join(paths.RES_DIR, "DIBS_sim/{}/{}/{}/HeatingEnergy_{}-{}-{}_{}-{}-{}.csv".format(scr_gebaeude_id, output_resolution, training_ratio, start.year, start.month, start.day, end.year, end.month, end.day)), sep=';', index_col=0)
    HeatingEnergy_sum.index = pd.to_datetime(HeatingEnergy_sum.index) + pd.DateOffset(hours=23)

    # Merge metered and simulated data
    merged_df = pd.merge(metered, HeatingEnergy_sum, left_index=True, right_index=True, how='inner')
    area = inputs.get_building_inputdata(scr_gebaeude_id)['ebf'].iloc[0]
    merged_df = merged_df.div(area)

    ''' PLOT '''
    # Create the plot
    plt.figure(figsize=figsize)

    # Plot the metered and simulated energy consumption
    plt.plot(merged_df.index, merged_df.iloc[:, 0], color='black', label='Metered', marker='o', markersize=6, linestyle='-', linewidth=2)
    plt.plot(merged_df.index, merged_df.iloc[:, 1], color='red', label='Simulated', marker='o', markersize=6, linestyle='-', linewidth=2)

    # Set title and labels
    plt.title(f'Metered and Simulated Heat Energy Consumption', fontsize=size_title, family=font_family)
    plt.xlabel('Year', fontsize=size_labels)
    plt.ylabel('Heat energy demand (kWh/m²)', fontsize=size_labels)

    # # Format the x-axis to show years
    merged_df.index = pd.to_datetime(merged_df.index)
    years = merged_df.index.year
    plt.xticks(merged_df.index, years)
    plt.xticks(rotation=45, fontsize=size_ticks)

    # Customize the y-axis
    plt.yticks(fontsize=size_ticks)

    # Add a legend
    plt.legend(fontsize=size_legend, loc='upper right', frameon=True, framealpha=0.7, edgecolor='black')

    # Customize grid and axis lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().spines['top'].set_linewidth(1)
    plt.gca().spines['right'].set_linewidth(1)
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['bottom'].set_linewidth(1)

    # Show the plot
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(paths.PLOTS_DIR, f"{scr_gebaeude_id}/1_ModelPriorCalibration_{scr_gebaeude_id}_{output_resolution}.png")
    plt.savefig(save_path, format='png')

    # Show the plot
    plt.show()
    plt.close()





def Plot_SA(scr_gebaeude_id, calib_type, output_resolution, training_ratio, sample_sizes, converged_sample_size, total_Si):

    # Plot total sensitivity index after convergence
    total_Si.rename(index=inputs.name_mapping(), inplace=True)
    total_Si['ST'].plot(kind='line', figsize=figsize)
    # Add title and labels
    plt.title('Total Sensitivity Index Value for Each Parameter After Convergence'.format(converged_sample_size), fontsize=size_title, family=font_family)
    plt.xlabel('Parameters',fontsize=size_labels, family=font_family)
    plt.ylabel('Total Sensitivity Index', fontsize=size_labels, family=font_family)
    # Rotate x-axis labels for better readability if needed
    plt.xticks(ticks=range(len(total_Si)), labels=total_Si.index, rotation=50, ha='right', fontsize=size_ticks, family=font_family)
    plt.yticks(fontsize=size_ticks, family=font_family)
    # Show the plot
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(paths.PLOTS_DIR, f"{scr_gebaeude_id}/2a_ST_values{scr_gebaeude_id}_{output_resolution}.png"))
    plt.show()
    plt.close()

    

    # Plot parameter ranking for all samples
    idx = sample_sizes.index(converged_sample_size)
    # Dictionary to store the ranks for each parameter
    ranks = {}
    parameters = []

    for sample_no in sample_sizes[:idx+1]:
        folder_name = "SA/SA_convergence_{}_{}_{}_{}_sample_no_{}".format(
            calib_type, scr_gebaeude_id, output_resolution, training_ratio, sample_no)
        file_path = os.path.join(paths.DATA_DIR, folder_name, f'TotalSi_{scr_gebaeude_id}_{sample_no}.xlsx')
        
        df = pd.read_excel(file_path)
        
        # If parameters list is empty, populate it from first file
        if not parameters:
            parameters = df.iloc[:, 0].tolist()
        
        # Sort parameters based on ST values to get rankings (descending order for highest ST = rank 1)
        df_sorted = df.sort_values(by='ST', ascending=False)
        ranks[sample_no] = df_sorted.iloc[:, 0].tolist()

    name_mapping = inputs.name_mapping()

    colorPalette = [ "#1d1d1d", "#ebce2b", "#702c8c", "#db6917", "#96cde6", "#ba1c30", "#c0bd7f", "#7f7e80", "#5fa641",
                        "#d485b2", "#4277b6", "#df8461", "#463397", "#e1a11a", "#91218c", "#e8e948", "#7e1510", "#92ae31", 
                        "#6f340d", "#d32b1e", "#2b3514"]

    # Setup plot
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font_family]
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate multiplied sample numbers for x-axis
    multiplied_sample_numbers = [x * 44 for x in sample_sizes[:idx+1]]

    # Plot for each parameter
    for param_idx, parameter in enumerate(parameters):
        y = [ranks[sample_no].index(parameter) + 1 for sample_no in sample_sizes[:idx+1]]
        
        ax.plot(multiplied_sample_numbers, y, 
               marker='o', color=colorPalette[param_idx], 
               linewidth=1.5, markersize=8)
        
        ax.text(multiplied_sample_numbers[-1] + 20, y[-1], 
               f' {name_mapping[parameter]}', 
               va='center', ha='left', 
               color=colorPalette[param_idx], 
               fontsize=14, fontfamily=font_family)

    ax.set_title(f'Parameter Ranking Based on Sobol´s Total Sensitivity Index', 
                pad=20, fontsize=size_title, fontweight='bold', fontfamily=font_family)
    ax.set_xlabel('Number of Samples', 
                 fontsize=size_labels, fontweight='medium', fontfamily=font_family, labelpad=10)
    ax.set_ylabel('Parameter Ranking', 
                 fontsize=size_labels, fontweight='medium', fontfamily=font_family, labelpad=10)
    
    ax.set_xticks(multiplied_sample_numbers)
    ax.set_yticks(range(1, len(parameters) + 1))
    ax.tick_params(axis='both', which='major', labelsize=size_ticks)
    
    ax.set_xlim(600, multiplied_sample_numbers[-1] + 1200)
    ax.set_ylim(21.5, 0.5)
    
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.grid(False)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    # Save and display the final plot
    plt.savefig(os.path.join(paths.PLOTS_DIR, f'{scr_gebaeude_id}/2b_SA_convergence_{scr_gebaeude_id}_{output_resolution}.png'))
    plt.show()
    plt.close()





def Plot_GP(best_result, scr_gebaeude_id, output_resolution, num_bc_param, training_ratio):

    dir_path = os.path.join(paths.CTRL_DIR, f'{scr_gebaeude_id}')
    results_df = pd.read_csv(os.path.join(dir_path, f'GP_Conv_all_results_{scr_gebaeude_id}_{output_resolution}_{num_bc_param}_{training_ratio}.csv')) #, index_col=0)

    # For convergence plot
    filtered_df = results_df[results_df['Kernel_Index'] == best_result['Kernel_Index']]
    filtered_df = filtered_df[filtered_df['Num_Samples_GP'] <= best_result['Num_Samples_GP']]
    filtered_df['Total_Time'] = filtered_df['Sampling_Time'] + filtered_df['Training_Time']
    filtered_df = filtered_df.sort_values(by='Num_Samples_GP', ascending=True)


    # Plot configuration
    COLORS = {
        'rmse': '#E63946', 'r2': '#1D3557', 'text': '#2F2F2F',
        'axis': '#666666', 'background': '#FFFFFF'}

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.85, 0.15],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        vertical_spacing=0.15)

    # Add RMSE trace
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Num_Samples_GP'],
            y=filtered_df['RMSE_NORM']*100,
            name="Normalized RMSE",
            mode='lines+markers',
            line=dict(width=2, color=COLORS['rmse']),
            marker=dict(color=COLORS['rmse'], symbol='diamond', size=8, line=dict(width=1, color='white')),
            hovertemplate="<b>Normalized RMSE:</b> %{y:.3f}%<br>Samples: %{x}<extra></extra>"
        ),
        row=1, col=1,
        secondary_y=False
    )

    # Add R² trace
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Num_Samples_GP'],
            y=filtered_df['R2'],
            name="R²",
            mode='lines+markers',
            line=dict(width=2, color=COLORS['r2']),
            marker=dict(color=COLORS['r2'], symbol='circle', size=8, line=dict(width=1, color='white')),
            hovertemplate="<b>R² Score:</b> %{y:.4f}<br>Samples: %{x}<extra></extra>"
        ),
        row=1, col=1,
        secondary_y=True
    )

    # Add computation time heatmap
    comp_time = filtered_df['Total_Time'].values.reshape(1, -1)
    fig.add_trace(
        go.Heatmap(
            z=comp_time,
            x=filtered_df['Num_Samples_GP'],
            y=[''],
            colorscale=[
                [0, '#184E77'], [0.25, '#1E6091'],
                [0.5, '#1A759F'], [0.75, '#168AAD'],
                [1, '#34A0A4']
            ],
            colorbar=dict(
                title=dict(text='Computational time<br>            (s)', font=dict(size=15, color=COLORS['text'])),
                thickness=15, len=0.3, xanchor='left', x=1.05, y=0.15,
                yanchor='bottom', tickfont=dict(size=15),
                orientation='v', bgcolor='rgba(255,255,255,0.9)',
                bordercolor=COLORS['axis'], borderwidth=1
            ),
            showscale=True,
            hovertemplate="<b>Computation Time:</b> %{z:.1f}s<br>Samples: %{x}<extra></extra>"
        ),
        row=2, col=1
    )

    fig.add_annotation(
    x=0.89,  # Position from 0 to 1 in x
    y=0.75,  # Position from 0 to 1 in y
    xref='paper',
    yref='paper',
    text='Convergence at {} samples <br> Normalized RMSE: {:.3f} % <br> R²-Score: {:.2f}'.format(best_result['Num_Samples_GP'], best_result['RMSE_NORM'] * 100, best_result["R2"]),
    showarrow=False,
    font=dict(size=15),
    bgcolor='white',
    bordercolor='black',
    borderwidth=1,
    borderpad=10,
    align='left'
    )

    # Update layout
    x_min = filtered_df['Num_Samples_GP'].min()
    x_max = filtered_df['Num_Samples_GP'].max()
    tick_vals = list(range(int(x_min), int(x_max)+1, 10))

    fig.update_layout(
        template='plotly_white',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        title=dict(
            text='<b>Convergence Analysis of Gaussian Process Regression',
            x=0.5, y=0.95, xanchor='center', yanchor='top',
            font=dict(family=font_family, size=24, color=COLORS['text'])
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05,
            xanchor="center", x=0.5, font=dict(size=20),
            bordercolor=COLORS['axis'], borderwidth=1,
            bgcolor='rgba(255,255,255,0.9)'
        ),
        margin=dict(l=80, r=150, t=150, b=40),
        width=1000, height=700,
        hovermode='x unified'
    )

    # Update axes
    for row in [1, 2]:
        fig.update_xaxes(
            range=[x_min-1, x_max+1], tickmode='array', tickvals=tick_vals,
            showticklabels=True, showgrid=False, zeroline=False,
            showline=True, linewidth=1, linecolor=COLORS['axis'],
            ticks="outside", tickwidth=1, tickcolor=COLORS['axis'],
            tickfont=dict(size=20), row=row, col=1
        )

    fig.update_xaxes(
        title=dict(
            text="Number of Samples for Training",
            font=dict(size=20, color=COLORS['text']),
            standoff=15
        ),
        row=2, col=1
    )

    fig.update_yaxes(
        title=dict(text="Normalized RMSE (%)", font=dict(size=20, color=COLORS['rmse'])),
        showgrid=False, zeroline=False, showline=True,
        linewidth=1, linecolor=COLORS['rmse'],
        ticks="outside", tickwidth=1, tickcolor=COLORS['rmse'],
        tickfont=dict(size=20, color=COLORS['rmse']),
        row=1, col=1, secondary_y=False
    )

    fig.update_yaxes(
        title=dict(text="R²", font=dict(size=20, color=COLORS['r2'])),
        showgrid=False, zeroline=False, showline=True,
        linewidth=1, linecolor=COLORS['r2'],
        ticks="outside", tickwidth=1, tickcolor=COLORS['r2'],
        tickfont=dict(size=20, color=COLORS['r2']),
        tickformat=".2f",
        row=1, col=1, secondary_y=True
    )

    fig.update_yaxes(
        showticklabels=False,
        showline=False,
        row=2, col=1
    )

    # Save and show plot
    plot_path = os.path.join(paths.PLOTS_DIR, f'{scr_gebaeude_id}/3a_GP_Conv_{scr_gebaeude_id}_{output_resolution}_{training_ratio}.html')
    # fig.show() 
    fig.write_html(plot_path)





def Plot_MetaModel(scr_gebaeude_id, num_bc_param, output_resolution, training_ratio, trained_meta_model, samples_meta_model, end_time_cal):


    ''' OPEN DATA '''
    file_path = os.path.join(paths.DATA_DIR, f'GP/{scr_gebaeude_id}/{output_resolution}/{num_bc_param}_bc_param')

    # Trained best meta-model
    file_name = os.path.join(file_path, trained_meta_model)

    file_gp = open(file_name, 'rb')
    objects = pickle.load(file_gp)
    file_gp.close()

    gaussian_process = objects[0]
    scaler_X = objects[1]

    # Samples for best meta-model
    file_gp_samples = open(os.path.join(file_path, samples_meta_model), "rb")
    samples_gp = pickle.load(file_gp_samples)
    file_gp_samples.close()
    param_names_list = samples_gp.drop("heating_energy", axis=1).columns.tolist()

    # Getting the prior parameter mean values
    variables_df = inputs.get_building_inputdata(scr_gebaeude_id)
    be_data_original = inputs.setup_be_data_original(scr_gebaeude_id, variables_df)
    prior_params_mean=[]
    for name in param_names_list:
        prior_params_mean.append(be_data_original[name].iloc[0])

    # DIBS
    metered=pd.read_excel(os.path.join(paths.DATA_DIR, "HeatEnergyDemand_{}_{}.xlsx".format(scr_gebaeude_id, output_resolution)), index_col=0)
    start_time, end_time = metered.index[0].strftime('%Y-%m-%d %H:%M:%S'), metered.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    DIBS_output = pd.read_csv(os.path.join(paths.RES_DIR, "DIBS_sim/{}/{}/{}/HeatingEnergy_{}-{}-{}_{}-{}-{}.csv".format(scr_gebaeude_id, output_resolution, training_ratio, start.year, end.month, end.day, end.year, end.month, end.day)), sep=';', index_col=0)
    DIBS_output.index = pd.to_datetime(DIBS_output.index) + pd.DateOffset(hours=23)

    # Meta-model
    X_scaled = scaler_X.transform(np.array(prior_params_mean).reshape(1,-1))
    Y = gaussian_process.predict(X_scaled)
    meta_model_output = Y[0] 

    # TRY-version requires this
    if output_resolution == None:
        str_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').year
        fnsh_time = datetime.strptime(end_time_cal, '%Y-%m-%d %H:%M:%S').year

        Climate_Factor = inputs.CF_factors(scr_gebaeude_id, str_time, fnsh_time)
        meta_model_output = np.divide(meta_model_output, Climate_Factor)



    df = DIBS_output.head(len(meta_model_output))
    df['metamodel'] = meta_model_output.tolist()

    area = inputs.get_building_inputdata(scr_gebaeude_id)['ebf'].iloc[0]
    df = df.div(area)

    df.index = pd.to_datetime(df.index)
    df.index = df.index.year
    df.index.name = 'year'


    ''' PLOT '''
    fig, ax = plt.subplots(figsize=(10, 7))  # Specify width and height in inches

    # Plot each line with respective style
    ax.plot(df.index, df['HeatingEnergy'], label='DIBS', color='#E6A532', linestyle='-', marker='o', markersize=25)
    ax.plot(df.index, df['metamodel'], label='Meta-model', color='#466EB4', linestyle='-', marker='*', markersize=20)

    # Title and labels
    ax.set_title('DIBS & Meta-model outputs', fontsize=size_title, fontfamily=font_family)
    ax.set_xlabel('Year', fontsize=size_labels, fontfamily=font_family)
    ax.set_ylabel('Heat energy demand (kWh/m2)', fontsize=size_labels, fontfamily=font_family)

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=size_ticks)
    ax.xaxis.set_tick_params(rotation=0)

    # Grid lines
    ax.grid(False)

    # Customizing legend
    legend = ax.legend(title='', fontsize=size_legend, title_fontsize=size_legend, loc='best')

    # Customize axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Save or show plot
    plt.tight_layout()
    plot_path = os.path.join(paths.PLOTS_DIR, f'{scr_gebaeude_id}/3b_MetaModel_{scr_gebaeude_id}_{output_resolution}.png')
    plt.savefig(plot_path, dpi=100)
    plt.show()
    plt.close()





def Plot_BC(scr_gebaeude_id, num_bc_param, output_resolution, training_ratio, draws, tune, chains):
    """
    Create plots from saved trace file with dynamic variable name mapping
    """
    # Complete name mapping dictionary
    name_mapping = inputs.name_mapping()

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
    plots_dir = os.path.join(paths.PLOTS_DIR, str(scr_gebaeude_id))
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
    
    # Create renamed version for plotting
    inference_data_renamed = normalized_data.copy()
    new_posterior = normalized_data.posterior.rename(trace_name_mapping)
    inference_data_renamed.posterior = new_posterior
    
    # Generate plots
    n_vars = len(inference_data_renamed.posterior.data_vars)
    
    # Trace plot
    plt.figure(figsize=(n_vars * 3, 3))
    az.plot_trace(inference_data_renamed)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'4a_BC_Trace_plot_{output_resolution}_{draws}_{tune}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # # Posterior plot
    # plt.figure(figsize=(n_vars * 4, 4))
    # az.plot_posterior(inference_data_renamed)
    # plt.tight_layout()
    # plot_path = os.path.join(plots_dir, f'BC_Posterior_{output_resolution}_{draws}_{tune}.png')
    # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close()
    
    # Summary statistics
    summary = az.summary(normalized_data)
    summary_path = os.path.join(plots_dir, f'4b_BC_Summary_{output_resolution}_{draws}_{tune}.csv')
    summary.to_csv(summary_path)

    return summary





def Plot_PosteriorParam(scr_gebaeude_id, num_bc_param, output_resolution, num_samples_gp, training_ratio, draws, tune, chains):
        
    ''' READ DATA '''
    param_names_list, prior_params_mean, prior_params_lower, prior_params_upper, prior_params_sigma = inputs.get_prior_param_values(scr_gebaeude_id, num_bc_param, num_samples_gp, output_resolution, training_ratio)
    prior_prob = pd.read_excel(os.path.join(paths.DATA_DIR, 'prior_probability_definition.xlsx'), index_col=0)
    name_mapping = inputs.name_mapping()

    # Prior parameter distributions
    with pm.Model() as model:
        list_var_distr = []
        for var in param_names_list:
            if prior_prob.loc['{}'.format(var), 'bc'] == 'Normal':
                list_var_distr.append(pm.TruncatedNormal(var, mu=prior_params_mean[var], sigma=prior_params_sigma[var], lower=prior_params_lower[var], upper=prior_params_upper[var]))

            if prior_prob.loc['{}'.format(var), 'bc'] == 'Uniform':
                list_var_distr.append(pm.Uniform(var, lower=prior_params_lower[var], upper=prior_params_upper[var]))

    ''' PLOT '''
    with model:
        file_name = os.path.join(paths.RES_DIR, f'calibration/{scr_gebaeude_id}/{num_bc_param}_bc_param/{training_ratio}_obs_{output_resolution}_{draws}_{tune}_{chains}.nc')
        inference_data = az.from_netcdf(file_name)
        draws = inference_data.sample_stats.dims['draw']
        chains = inference_data.sample_stats.dims['chain']

        trace_prior = pm.sample(draws=draws, chains=chains, cores=16, progressbar=False)
        posterior = inference_data.posterior.stack(sample=['chain', 'draw']) 

        for var in param_names_list:
            plt.figure(figsize=(6, 4))
            sns.histplot(trace_prior[var], bins=20, color='#4F81BD', label='Prior', stat='density', edgecolor=None)
            # Plot the KDE with a thicker line
            sns.kdeplot(trace_prior[var], color='#4F81BD', linewidth=6)

            sns.histplot(posterior[var], bins=20, color='#E45854', label='Posterior', stat='density', edgecolor=None)
            # Plot the KDE with a thicker line
            sns.kdeplot(posterior[var], color='#E45854', linewidth=6)

            plt.axvline(np.mean(trace_prior[var]), color='#4F81BD', linestyle='--', linewidth=3, label='Prior Mean')
            plt.axvline(np.mean(posterior[var]), color='#E45854', linestyle='--', linewidth=3, label='Posterior Mean')

            plt.xlabel(name_mapping.get(var, var), fontsize=16, family=font_family)
            plt.ylabel('Density', fontsize=16, family=font_family)
            plt.xticks(fontsize=14, family=font_family)
            plt.yticks(fontsize=14,family=font_family)
            plt.title('Prior and Posterior Distributions of {}'.format(name_mapping.get(var, var)), fontsize=16, family=font_family)
            plt.legend()

            #remove frame from each side of plot
            sns.despine()

            plot_path = os.path.join(paths.PLOTS_DIR, f'{scr_gebaeude_id}/5_PosteriorDistrParam_{var}_{scr_gebaeude_id}_{output_resolution}.png')
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.show()
            plt.close()





def Plot_CalibratedModel(scr_gebaeude_id, num_bc_param, output_resolution, training_ratio, draws, tune, start_time, end_time):

    # Construct results filename
    parameter_combination = draws*tune
    
    results_base_dir = os.path.join(paths.RES_DIR, 'calibrated_model_simulation')
    building_results_dir = os.path.join(results_base_dir, f'{scr_gebaeude_id}_{output_resolution}')
    results_file_name = (f'CalibratedSimulationResults_{scr_gebaeude_id}_numBC_{num_bc_param}_samples_{parameter_combination}_training_{training_ratio}_{output_resolution}.xlsx')
    
    # Load data
    metered = pd.read_excel(os.path.join(paths.DATA_DIR, f"HeatEnergyDemand_{scr_gebaeude_id}_{output_resolution}.xlsx"), index_col=0)
    
    # Get start and end times from metered data
    start_time = metered.index[0].strftime('%Y-%m-%d %H:%M:%S')
    end_time = metered.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    file_path = os.path.join(building_results_dir, results_file_name)
    posterior_sim = pd.read_excel(file_path)

    # Extract years from metered data
    start_year = metered.index[0].year
    end_year = metered.index[-1].year + 1
    years = list(range(start_year, end_year))
    
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
            legendgroup='Calibrated model',
            showlegend=True,
            legendgrouptitle_text="Calibrated model"))

    # Add metered data
    metered_values = metered[metered.index.year.isin(years)].values.flatten()
    metered_years = metered[metered.index.year.isin(years)].index.year
    fig.add_trace(go.Scatter(
        x=metered_years,
        y=metered_values/area,
        mode='markers+lines',
        name='Metered data',
        marker=dict(color='black', size=20, symbol='circle'),
        line=dict(width=2)))

    # Add DIBS output
    dibs_years = DIBS_output.index.year
    dibs_values = DIBS_output['HeatingEnergy'].values
    fig.add_trace(go.Scatter(
        x=dibs_years,
        y=dibs_values/area,
        mode='markers+lines',
        name='DIBS prior calibration',
        marker=dict(color='red', size=20, symbol='cross'),
        line=dict(width=2)))

    # Update layout
    fig.update_layout(
        title=dict(
            text="Predicted Heat Energy Consumption with Calibrated Model",
            font=dict(family=font_family, size=26),
            x=0.5,
            y=0.95
        ),
        xaxis_title=dict(text="Year", font=dict(family=font_family, size=20)),
        yaxis_title=dict(text='<b>Heat Energy Consumption</b> (kWh/m²)', font=dict(family=font_family, size=20)),
        violingap=0.1,
        violingroupgap=0,
        plot_bgcolor='white',
        font=dict(family=font_family, size=20),
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            linecolor='black',
            tickfont=dict(family=font_family, size=20),
            titlefont=dict(family=font_family, size=22)
        ),
        yaxis=dict(
            range=[55, 120],
            showgrid=True,
            zeroline=False,
            linecolor='black',
            tickfont=dict(family=font_family, size=20),
            titlefont=dict(family=font_family, size=22)
        ),
        width=1000, 
        height=700)
    
    fig.add_shape(type="rect",
    x0=0, y0=55, x1=len(years)*(training_ratio/100)-1, y1=120,
    line=dict(color="rgba(128, 128, 128, 0.25)", width=2),
    fillcolor="rgba(128, 128, 128, 0.25)")

    # Save and show plot
    plot_path = os.path.join(paths.PLOTS_DIR, f'{scr_gebaeude_id}/6_CalibModel_{scr_gebaeude_id}_{output_resolution}_{training_ratio}.html')
    fig.write_html(plot_path)

    return fig

