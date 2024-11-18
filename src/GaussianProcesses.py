import time
import os
import pandas as pd
import pickle
import multiprocessing
from multiprocessing import Pool
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, ExpSineSquared, ConstantKernel as C
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



def perform_gp_convergence(scr_gebaeude_id, climate_file, output_resolution, calib_type, min_gp_samples, max_gp_samples, num_bc_param, num_samples_sa, step_gp, rmse_threshold, gp_test_size = 0.2, training_ratio = 0.75):
    start_time=time.time()

    metered = pd.read_excel(os.path.join(paths.DATA_DIR, f'HeatEnergyDemand_{scr_gebaeude_id}_{output_resolution}.xlsx'), index_col=0)
    nr_train_data = round(metered.shape[0] * training_ratio / 100)
    start_time_cal, end_time_cal = metered.index[0].strftime('%Y-%m-%d %H:%M:%S'), metered.index[nr_train_data].strftime('%Y-%m-%d %H:%M:%S')

    
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
        f.write(f'  R²: {best_result['R2']:.4f}\n\n')
        f.write(f'  RMSE_NORM: {best_rmse_norm:.4f}\n\n')
        if threshold_found:
            f.write(f'Normalised RMSE threshold of {rmse_threshold} and R² > 0.98 reached. Search stopped early.\n')
        else:
            f.write(f'Note: RMSE threshold of {rmse_threshold} or R² > 0.98 not reached.\n')
            f.write('Suggestions:\n')
            f.write('  1. Try increasing the number of samples (adjust max_gp_samples)\n')
            f.write('  2. Experiment with different kernel types\n')

    end_time = time.time()
    
    return best_result, best_kernel_path, best_samples_df_path, end_time-start_time

if __name__ == '__main__':
    multiprocessing.freeze_support()
