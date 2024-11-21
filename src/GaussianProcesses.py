import time
import os
import pandas as pd
import pickle
import multiprocessing
from multiprocessing import Pool
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, ExpSineSquared, ConstantKernel as C
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from run_GP_samples import sample_gp
from run_GP_train import train_gp
try:
    import paths 
except:
    import src.paths as paths



def train_gp_wrapper(args):

    kernel_index, kernel_name, kernel, num_samples_gp, scr_gebaeude_id, num_bc_param, num_samples_sa, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio, gp_test_size = args
    start_time_sampling = time.time()
    
    samples_df = sample_gp(scr_gebaeude_id, num_bc_param, num_samples_sa, num_samples_gp, climate_file, 
                           start_time_cal, end_time_cal, output_resolution, training_ratio)
    
    end_time_sampling = time.time()

    start_time_training = time.time()

    gaussian_process, mse, mae, rmse, r2, sigma, y_test_mean, mape, rmse_normalised = train_gp(scr_gebaeude_id, num_bc_param, kernel, 
                                                                                            kernel_index, num_samples_gp, gp_test_size, 
                                                                                            output_resolution, training_ratio)

    end_time_training = time.time()

    return {
        'Kernel_Index': kernel_index,
        'Kernel_Name': kernel_name,
        'Kernel':kernel,
        'Num_Samples_GP': num_samples_gp, 
        'RMSE': rmse, 
        'Sampling_Time': end_time_sampling - start_time_sampling,
        'Training_Time': end_time_training - start_time_training, 
        'R2': r2, 
        'MAPE': mape, 
        'Y_TEST_MEAN': y_test_mean,
        'Training_ratio': training_ratio,
        'RMSE_NORM': rmse_normalised,
        'GP_Object': gaussian_process,
        'Samples_DF': samples_df }



def perform_gp_convergence(scr_gebaeude_id, climate_file, output_resolution, calib_type, start_time_cal, end_time_cal, min_gp_samples, max_gp_samples, num_bc_param, num_samples_sa, step_gp, rmse_threshold, gp_test_size = 0.2, training_ratio = 0.75):
    start_time=time.time()

    # Kernels to be tested are listed here
    kernels = [
    (1, 'RationalQuadratic', C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-5, 1e5), alpha_bounds=(1e-5, 1e5))),
    (3, 'RBF + ExpSine', C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)) + C(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds=(1e-5, 1e5), periodicity_bounds=(1e-5, 1e5))),
    (4, 'RBF', C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)))]

    total_cores = multiprocessing.cpu_count()
    cores_to_use = max(1, total_cores - 2)
    print(f'Total CPU cores: {total_cores}')
    print(f'Cores to be used for processing: {cores_to_use}')

    tasks = [(kernel_index, kernel_name, kernel, num_samples_gp, scr_gebaeude_id, num_bc_param, num_samples_sa, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio, gp_test_size) 
             for kernel_index, kernel_name, kernel in kernels 
             for num_samples_gp in range(min_gp_samples, max_gp_samples, step_gp)]
    
    results = []
    threshold_found = False
    best_result = None

    with Pool(processes=cores_to_use) as pool:
        for result in pool.imap_unordered(train_gp_wrapper, tasks):
            results.append(result)
            
            if result['RMSE_NORM'] < rmse_threshold and result['R2'] > 0.99:
                threshold_found = True
                best_result = result
                print(f'\nRMSE threshold of {rmse_threshold} and R² > 0.99 reached!')
                #print(f'Stopping search. Found RMSE {result['RMSE_NORM']} and R² {result['R2']} for kernel {result['Kernel']} with {result['Num_Samples_GP']} samples.')
                pool.terminate()
                break

    results_df = pd.DataFrame(results)
    
    if not threshold_found:
        print('\nWarning: No result met both RMSE and R² criteria. Selecting best available result.')
        best_result = min(results, key=lambda x: x['RMSE_NORM'])

    best_rmse_norm = best_result['RMSE_NORM']
    best_kernel_index = best_result['Kernel_Index']
    best_kernel = best_result['Kernel']
    best_num_samples = best_result['Num_Samples_GP']
    best_training_ratio = best_result['Training_ratio']

    # Create a copy of results without the large objects for saving to Excel/CSV
    results_for_saving = [{k: v for k, v in r.items() if k not in ['Samples_DF', 'GP_Object']} for r in results]
    results_df = pd.DataFrame(results_for_saving)
    
    # results_df_save = results_df.drop(columns=['GP_Object'])
    results_df.to_excel(os.path.join(paths.CTRL_DIR, f'GP/GP_Convergence_all_results_{scr_gebaeude_id}_{output_resolution}_{num_bc_param}_{training_ratio}_new.xlsx'), index=False)
    results_df.to_csv(os.path.join(paths.CTRL_DIR, f'GP/GP_Convergence_all_results_{scr_gebaeude_id}_{output_resolution}_{num_bc_param}_{training_ratio}.csv'), index=False)
    
    best_kernel_path = os.path.join(paths.CTRL_DIR, 'GP', f'best_kernel_{scr_gebaeude_id}_{output_resolution}_{training_ratio}.pkl')
    with open(best_kernel_path, 'wb') as f:
        pickle.dump(best_result['GP_Object'], f)

    best_samples_df_path = os.path.join(paths.CTRL_DIR, 'GP', f'best_samples_df_{scr_gebaeude_id}_{output_resolution}_{training_ratio}.pkl')
    with open(best_samples_df_path, 'wb') as f:
        pickle.dump(best_result['Samples_DF'], f)

    filename = os.path.join(paths.CTRL_DIR, 'GP', f'{scr_gebaeude_id}_{output_resolution}_{best_training_ratio}_gp_convergence.txt')
    with open(filename, 'w') as f:
        f.write('GP Convergence Results\n')
        f.write('======================\n\n')
        f.write(f'Building ID: {scr_gebaeude_id}\n')
        f.write(f'  Kernel Index: {best_kernel_index}\n')
        f.write(f'Calibration Type: {calib_type}\n')
        f.write(f'Number of Observations for Training: {best_training_ratio}\n')
        f.write(f'Total CPU cores: {total_cores}\n')
        f.write(f'Cores used for processing: {cores_to_use}\n\n')
        f.write('Best GP Configuration:\n')
        f.write(f'  Kernel: {best_kernel}\n')
        f.write(f'  Number of Samples: {best_num_samples}\n')
        #f.write(f'  R²: {best_result['R2']:.4f}\n\n')
        f.write(f'  R²: {best_result["R2"]:.4f}\n\n')
        f.write(f'  RMSE_NORM: {best_rmse_norm:.4f}\n\n')
        if threshold_found:
            f.write(f'Normalised RMSE threshold of {rmse_threshold} and R² > 0.98 reached. Search stopped early.\n')
        else:
            f.write(f'Note: RMSE threshold of {rmse_threshold} or R² > 0.98 not reached.\n')
            f.write('Suggestions:\n')
            f.write('  1. Try increasing the number of samples (adjust max_gp_samples)\n')
            f.write('  2. Experiment with different kernel types\n')

    end_time = time.time()
    
    ''' New code Start'''
       # Create convergence plot
    filtered_df = results_df[results_df['Kernel_Index'] == best_kernel_index]
    filtered_df = filtered_df[filtered_df['Num_Samples_GP'] <= best_num_samples]
    filtered_df['Total_Time'] = filtered_df['Sampling_Time'] + filtered_df['Training_Time']
    filtered_df = filtered_df.sort_values(by='Num_Samples_GP', ascending=True)

    # Plot configuration
    COLORS = {
        'rmse': '#E63946', 'r2': '#1D3557', 'text': '#2F2F2F',
        'axis': '#666666', 'background': '#FFFFFF'
    }

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.85, 0.15],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        vertical_spacing=0.15
    )

    # Add RMSE trace
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Num_Samples_GP'],
            y=filtered_df['RMSE_NORM']*100,
            name="Normalisierte RMSE",
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
                title=dict(text='Berechnungszeit<br>          (s)', font=dict(size=15, color=COLORS['text'])),
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
    text=f'Konvergenz bei {best_num_samples} Proben<br>' +
         f'Normalisierte RMSE: {best_rmse_norm * 100:.3f}%<br>' +
         f'R²-Wert: {best_result["R2"]:.4f}',
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
            text='<b>Konvergenzanalyse Gaussian Process Regressors',
            x=0.5, y=0.95, xanchor='center', yanchor='top',
            font=dict(family='Arial', size=20, color=COLORS['text'])
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05,
            xanchor="center", x=0.5, font=dict(size=20),
            bordercolor=COLORS['axis'], borderwidth=1,
            bgcolor='rgba(255,255,255,0.9)'
        ),
        margin=dict(l=80, r=150, t=150, b=40),
        width=1200, height=800,
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
            text="Anzahl der Versuche Training",
            font=dict(size=20, color=COLORS['text']),
            standoff=15
        ),
        row=2, col=1
    )

    fig.update_yaxes(
        title=dict(text="Normalisierte RMSE (%)", font=dict(size=20, color=COLORS['rmse'])),
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
        row=1, col=1, secondary_y=True
    )

    fig.update_yaxes(
        showticklabels=False,
        showline=False,
        row=2, col=1
    )

    # Save and show plot
    plot_path = os.path.join(paths.CTRL_DIR, 'GP', 
                            f'convergence_plot_{scr_gebaeude_id}_{output_resolution}_{training_ratio}.html')
    fig.write_html(plot_path)
    fig.show() 
    ''' New Code End'''
    return best_result, best_kernel_path, best_samples_df_path, end_time-start_time

if __name__ == '__main__':
    multiprocessing.freeze_support()
