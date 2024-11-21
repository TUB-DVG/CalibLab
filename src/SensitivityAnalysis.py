import os
import time
import pandas as pd
import numpy as np
from run_DIBS import run_model
from SALib.sample import saltelli
from SALib.analyze import sobol
import inputs
import DIBS.data_preprocessing.breitenerhebung.dataPreprocessingBE as preprocessing
import DIBS.iso_simulator.annualSimulation.annualSimulation as sim
import matplotlib.pyplot as plt
import multiprocessing

try:
    import paths 
except:
    import src.paths as paths

def cal_accu_dev(ts_ori, ts_var):
    ''' calculate accumulated deviation of two time series (df) '''
    assert len(ts_ori) == len(ts_var)
    return abs(ts_ori - ts_var).sum()

def plot_rankings_from_files(scr_gebaeude_id, calib_type, output_resolution, training_ratio, sample_sizes, converged_sample_size):
    dir_path = os.path.join(paths.DATA_DIR, "SA", str(scr_gebaeude_id), str(output_resolution))
    idx = sample_sizes.index(converged_sample_size)
    # Dictionary to store the ranks for each parameter
    ranks = {}
    parameters = []

    for sample_no in sample_sizes[:idx+1]:
        folder_name = "SA_convergence_{}_{}_{}_{}_sample_no_{}".format(
            calib_type, scr_gebaeude_id, output_resolution, training_ratio, sample_no)
        file_path = os.path.join(paths.DATA_DIR, folder_name, f'TotalSi_{scr_gebaeude_id}_{sample_no}.xlsx')
        
        df = pd.read_excel(file_path)
        
        # If parameters list is empty, populate it from first file
        if not parameters:
            parameters = df.iloc[:, 0].tolist()
        
        # Sort parameters based on ST values to get rankings (descending order for highest ST = rank 1)
        df_sorted = df.sort_values(by='ST', ascending=False)
        ranks[sample_no] = df_sorted.iloc[:, 0].tolist()

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
        'k_L': 'Faktor der Lichtbereitstellung'
    }

    colorPalette = [
        "#1d1d1d", "#ebce2b", "#702c8c", "#db6917", "#96cde6", "#ba1c30", "#c0bd7f", "#7f7e80", "#5fa641",
        "#d485b2", "#4277b6", "#df8461", "#463397", "#e1a11a", "#91218c", "#e8e948", "#7e1510", "#92ae31", 
        "#6f340d", "#d32b1e", "#2b3514"
    ]

    # Setup plot
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    fig, ax = plt.subplots(figsize=(24, 12))

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
               fontsize=20, fontfamily='Arial')

    ax.set_title(f'Parameterreihenfolge der Gesamtsensitivitätsindizes nach Sobol', 
                pad=20, fontsize=25, fontweight='bold', fontfamily='Arial')
    ax.set_xlabel('Anzahl der Versuche', 
                 fontsize=25, fontweight='medium', fontfamily='Arial', labelpad=10)
    ax.set_ylabel('Parameterreihenfolge basierend auf dem Gesamtsensitivitätsindex', 
                 fontsize=22, fontweight='medium', fontfamily='Arial', labelpad=10)
    
    ax.set_xticks(multiplied_sample_numbers)
    ax.set_yticks(range(1, len(parameters) + 1))
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    ax.set_xlim(600, multiplied_sample_numbers[-1] + 1200)
    ax.set_ylim(21.5, 0.5)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    
    ax.grid(False)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    # Save and display the final plot
    plt.savefig(os.path.join(dir_path, 'SA_convergence_final.png'))
    plt.show()
    plt.close()

def simulate(i, X, scr_gebaeude_id, fixed_variables_bd, calib_type, climate_file, start_time, end_time, output_resolution, ts_ori):
    be_data_original = inputs.setup_be_data_original_SA(scr_gebaeude_id, fixed_variables_bd, X)
    building_data = preprocessing.data_preprocessing(be_data_original)
    if calib_type == "AMY":
        hourlyResults = sim.cal_heating_energy_bd(building_data.iloc[0], climate_file, start_time, end_time, output_resolution)
        return i, cal_accu_dev(ts_ori, hourlyResults)["HeatingEnergy"]
    if calib_type == "TRY":
        return i, sim.cal_heating_energy_bd(building_data.iloc[0])
    
def find_converged_sa_sample_size(scr_gebaeude_id, calib_type, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio, SA_sampling_lowerbound, SA_sampling_upperbound):


    dir_path = os.path.join(paths.DATA_DIR, "SA", str(scr_gebaeude_id), str(output_resolution))

    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        # Another process in the multiprocessing may have created the directory already; just pass
        pass

        
    sa_start = time.time()

    ts_ori = run_model(scr_gebaeude_id, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio)

    # Set the range of sample sizes as powers of 2
    sample_sizes = [2**n for n in range(SA_sampling_lowerbound, SA_sampling_upperbound)]  # Start n with 5, end with 7
    
    intervals = inputs.create_intervals(scr_gebaeude_id)

    problem = {
        'num_vars': 21,
        'names': ["q25_1", "aw_fl", "qd1", "facade_area", 
                  "geb_f_flaeche_n_iwu", "d_fl_be", "nrf_2", "ebf", 
                  "n_og", "geb_f_hoehe_mittel_iwu", "glass_solar_transmittance", "qd8", 
                  "u_fen", "u_aw", "d_u_ges", "u_ug",
                  "heat_recovery_efficiency", "thermal_capacitance", "heating_coefficient",
                  "p_j_lx", "k_L"],
        'bounds': [intervals["q25_1"], intervals["aw_fl"], intervals["qd1"], intervals["facade_area"], 
                    intervals["geb_f_flaeche_n_iwu"], intervals["d_fl_be"], intervals["nrf_2"], intervals["ebf"], 
                    intervals["n_og"], intervals["geb_f_hoehe_mittel_iwu"], intervals["glass_solar_transmittance"], intervals["qd8"], 
                    intervals["u_fen"], intervals["u_aw"], intervals["d_u_ges"], intervals["u_ug"],
                    intervals["heat_recovery_efficiency"], intervals["thermal_capacitance"], intervals["heating_coefficient"], 
                    intervals["p_j_lx"], intervals["k_L"]]}

    # Arrays to store the index estimates
    S1_estimates = np.zeros([problem['num_vars'], len(sample_sizes)])
    ST_estimates = np.zeros([problem['num_vars'], len(sample_sizes)])
    S1_ranks = np.zeros([problem['num_vars'], len(sample_sizes)])
    ST_ranks = np.zeros([problem['num_vars'], len(sample_sizes)])
    #all_rankings = {}  # newly added
    fixed_variables = pd.read_excel(os.path.join(paths.DATA_DIR, 'fixed_variables.xlsx'))
    fixed_variables_bd = fixed_variables[fixed_variables['scr_gebaeude_id'] == scr_gebaeude_id].iloc[0]

    df_calc_time = pd.DataFrame(columns=['Num_Samples_SA', 'Calc_time'])

    num_cores = max(1, multiprocessing.cpu_count() - 2)  # Use all but 2 cores
    print(num_cores)
    converged_sample_size = None
    #sa_converged = 0  # Initialize convergence flag
    #sa_conv_time = 0 # newly added
    for idx, SA_num_samples in enumerate(sample_sizes):
        start_time1 = time.time()
        
        param_samples = saltelli.sample(problem, SA_num_samples, calc_second_order=True)
        print('Generated {} parameter combinations for {} samples. Cores allocated for simulation are {}'.format(param_samples.shape[0], SA_num_samples, num_cores))

        with multiprocessing.Pool(num_cores) as pool:
            results = pool.starmap(simulate, [(i, X, scr_gebaeude_id, fixed_variables_bd, calib_type, climate_file, start_time_cal, end_time_cal, output_resolution, ts_ori) for i, X in enumerate(param_samples)])

        Y = np.zeros([param_samples.shape[0]])
        for i, value in results:
            Y[i] = value

        sobol_results = sobol.analyze(problem, Y, print_to_console=False)

        end_time1 = time.time()
        df_calc_time = pd.concat([df_calc_time, pd.DataFrame([{'Num_Samples_SA': SA_num_samples, 'Calc_time': end_time1 - start_time1}])], ignore_index=True)

        new_folder_name = "SA_convergence_{}_{}_{}_{}_sample_no_{}".format(calib_type, scr_gebaeude_id, output_resolution, training_ratio, SA_num_samples)
        new_folder_path = os.path.join(paths.DATA_DIR, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        DF_Y = pd.DataFrame(Y)
        DF_Y.to_excel(os.path.join(new_folder_path, "Y_trial1{}_{}.xlsx".format(scr_gebaeude_id, SA_num_samples)))

        total_Si, first_Si, second_Si = sobol_results.to_df()

        total_Si.plot()
        plt.title('SA Convergence for {} Samples'.format(SA_num_samples))
        plt.savefig(os.path.join(new_folder_path, 'SA_convergence_{}_samples.png'.format(SA_num_samples)))

        total_Si.to_excel(os.path.join(new_folder_path, "TotalSi_{}_{}.xlsx".format(scr_gebaeude_id, SA_num_samples)))
        first_Si.to_excel(os.path.join(new_folder_path, "FirstSi_{}_{}.xlsx".format(scr_gebaeude_id, SA_num_samples)))
        second_Si.to_excel(os.path.join(new_folder_path, "SecondSi_{}_{}.xlsx".format(scr_gebaeude_id, SA_num_samples)))

        print("Files_written", SA_num_samples)
        # Store estimates
        S1_estimates[:, idx] = sobol_results['S1']
        ST_estimates[:, idx] = sobol_results['ST']
        print("S1 estimates written", SA_num_samples)
        # Calculate parameter rankings
        orderS1 = np.argsort(S1_estimates[:, idx])
        orderST = np.argsort(-ST_estimates[:, idx]) 
        S1_ranks[:, idx] = orderS1.argsort() + 1
        ST_ranks[:, idx] = orderST.argsort() + 1
        print("Sampling done ", SA_num_samples)
        
        print(f"Sample Size: {SA_num_samples}")
        print(f"Top 5 ST Parameters: {np.argsort(ST_ranks[:, idx])[:5] + 1}")
        # Check for convergence
        if idx > 0:
            prev_top5_ST = np.argsort(ST_ranks[:, idx-1])[:5] + 1
            curr_top5_ST = np.argsort(ST_ranks[:, idx])[:5] + 1

            if np.array_equal(prev_top5_ST, curr_top5_ST):
                converged_sample_size = SA_num_samples
                sa_end = time.time()
                sa_conv_time=sa_end-sa_start
                print("Time:", sa_conv_time)

                if output_resolution == None:
                    total_Si.to_excel(os.path.join(paths.DATA_DIR, "SA/{}/{}/TotalSi_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, converged_sample_size)))  
                if output_resolution != None:
                    total_Si.to_excel(os.path.join(paths.DATA_DIR, "SA/{}/{}/TotalSi_{}_obs_train_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, training_ratio, converged_sample_size)))

                break
            
           
    df_calc_time.to_excel(os.path.join(paths.CTRL_DIR, f'SA_Convergence_all_results_{calib_type}_{scr_gebaeude_id}_{output_resolution}_{training_ratio}.xlsx'), index=False)

    if converged_sample_size:
        print(f"The sample size for which the ranking of the top five parameters doesn't change compared to the previous sample size is {converged_sample_size}.")
        sa_converged = 1

    else:
        print("The rankings of the top five parameters did not stabilize within the given sample sizes.")
        converged_sample_size = 2**SA_sampling_upperbound
        sa_converged = 0
    
    #sa_converged =1 
    #converged_sample_size  = 64  
    if sa_converged:
        plot_rankings_from_files(scr_gebaeude_id, calib_type, output_resolution, training_ratio, sample_sizes, converged_sample_size)
    
    return converged_sample_size, sa_converged, sa_conv_time

