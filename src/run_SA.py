import os
# os.chdir(os.path.join(os.getcwd(), 'src'))
import pandas as pd
import time
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import inputs
import DIBS.data_preprocessing.breitenerhebung.dataPreprocessingBE as preprocessing
import DIBS.iso_simulator.annualSimulation.annualSimulation as sim
from run_DIBS import model_run
try:
    import paths 
except:
    import src.paths as paths



def cal_accu_dev(ts_ori, ts_var):
    ''' calculate accummulate deviation of two time series (df) '''
    assert len(ts_ori) == len(ts_var)
    return abs(ts_ori - ts_var).sum()



def run_SA(scr_gebaeude_id, num_samples_sa, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio):

    # if not os.path.exists(os.path.join(paths.DATA_DIR, "SA")):
    #     os.mkdir(os.path.join(paths.DATA_DIR, "SA"))
    # if not os.path.exists(os.path.join(paths.DATA_DIR, "SA", "{}".format(scr_gebaeude_id))):
    #     os.mkdir(os.path.join(paths.DATA_DIR, "SA", "{}".format(scr_gebaeude_id)))
    # if not os.path.exists(os.path.join(paths.DATA_DIR, "SA", "{}".format(scr_gebaeude_id), "{}".format(output_resolution))):
    #     os.mkdir(os.path.join(paths.DATA_DIR, "SA", "{}".format(scr_gebaeude_id), "{}".format(output_resolution)))
        
    dir_path = os.path.join(paths.DATA_DIR, "SA", "{}".format(scr_gebaeude_id), "{}".format(output_resolution))

    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        # Another process in the multiprocessing may have created the directory already; just pass
        pass


    start_time_calc = time.time()
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

    param_samples = saltelli.sample(problem, num_samples_sa, calc_second_order=True)
    
    # df_samples = pd.DataFrame(data=param_samples[0:,0:],
    #                             index = [i for i in range(param_samples.shape[0])],
    #                             columns = ['f'+str(i) for i in range(param_samples.shape[1])])   
    
    # df_samples.to_excel(os.path.join(paths.DATA_DIR, "SA/SA_conv/{}/{}/SA_{}_obs_train_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, num_obs_train, num_samples_sa)))

    fixed_variables = pd.read_excel(os.path.join(paths.DATA_DIR, 'fixed_variables.xlsx'))
    fixed_variables_bd = fixed_variables[fixed_variables['scr_gebaeude_id'] == scr_gebaeude_id].iloc[0]


# TRY - DOUBLE CHECK IF THIS PART IS CORRECT
    if output_resolution == None:
        print('Generated {} parameter combinations.'.format(param_samples.shape[0]))
        Y = []
        for i, X in enumerate(param_samples):
            be_data_original = inputs.setup_be_data_original_SA(scr_gebaeude_id, fixed_variables_bd, X)
            building_data = preprocessing.data_preprocessing(be_data_original)
            Y.append(sim.cal_heating_energy_bd(building_data.iloc[0], climate_file, start_time_cal, end_time_cal, output_resolution)["HeatingEnergy"])
            print("PERCENTAGE DONE OF SA: ", i/param_samples.shape[0]*100)
        results = sobol.analyze(problem, np.array(Y), print_to_console=True)
        total_Si, first_Si, second_Si = results.to_df()
        total_Si.to_excel(os.path.join(paths.DATA_DIR, "SA/{}/{}/TotalSi_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, num_samples_sa)))
        first_Si.to_excel(os.path.join(paths.DATA_DIR, "SA/{}/{}/First_Si_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, num_samples_sa)))
        second_Si.to_excel(os.path.join(paths.DATA_DIR, "SA/{}/{}/Second_Si_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, num_samples_sa)))
        print("SA is done for building {} - {} ".format(scr_gebaeude_id, output_resolution))

# AMY: Sim results using original inputs. This will be the base. The SA is calculated relativaly to this timeseries.
    else:
        print('Generated {} parameter combinations for {} parameters.'.format(param_samples.shape[0], param_samples.shape[1]))
        ts_ori = model_run(scr_gebaeude_id, climate_file, start_time_cal, end_time_cal, output_resolution)  
        Y = np.zeros([param_samples.shape[0]])
        for i, X in enumerate(param_samples):
            be_data_original = inputs.setup_be_data_original_SA(scr_gebaeude_id, fixed_variables_bd, X)
            building_data = preprocessing.data_preprocessing(be_data_original)
            hourlyResults = sim.cal_heating_energy_bd(building_data.iloc[0], climate_file, start_time_cal, end_time_cal, output_resolution)
            Y[i] = cal_accu_dev(ts_ori, hourlyResults)["HeatingEnergy"]
            print("PERCENTAGE DONE OF SA: ", i/param_samples.shape[0]*100)
        results = sobol.analyze(problem, Y, print_to_console=True)
        total_Si, first_Si, second_Si = results.to_df()
        total_Si.to_excel(os.path.join(paths.DATA_DIR, "SA/{}/{}/TotalSi_{}_obs_train_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, training_ratio, num_samples_sa)))
        first_Si.to_excel(os.path.join(paths.DATA_DIR, "SA/{}/{}/First_Si_{}_obs_train_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, training_ratio, num_samples_sa)))
        second_Si.to_excel(os.path.join(paths.DATA_DIR, "SA/{}/{}/Second_Si_{}_obs_train_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, training_ratio, num_samples_sa)))
        print("SA is done for building {} - {} - {} obs_train".format(scr_gebaeude_id, output_resolution, training_ratio))

    finish_time = time.time()

    return total_Si, finish_time-start_time_calc

