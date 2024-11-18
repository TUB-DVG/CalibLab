import pandas as pd
import os
import numpy as np
import pickle
try:
    import paths 
except:
    import src.paths as paths

''' Parameter values from the DATES-database '''
def get_building_inputdata(scr_gebaeude_id):

    variables_df = pd.read_excel(os.path.join(paths.DATA_DIR, 'variables_df.xlsx')) 
    variables_df = variables_df.loc[variables_df["scr_gebaeude_id"] == scr_gebaeude_id]

    return variables_df



''' Parameter values required by the DIBS Preprocessing [DataFrame] '''
def setup_be_data_original(scr_gebaeude_id, variables_df):
    fixed_variables = pd.read_excel(os.path.join(paths.DATA_DIR, 'fixed_variables.xlsx'))
    fixed_variables_bd = fixed_variables[fixed_variables['scr_gebaeude_id'] == scr_gebaeude_id].iloc[0]
    
    be_data_original = {"scr_gebaeude_id": scr_gebaeude_id, 
                        "bak_grob": fixed_variables_bd['bak_grob'], 
                        "plz": fixed_variables_bd['plz'], 
                        "hk_geb": fixed_variables_bd['hk_geb'], 
                        "uk_geb": fixed_variables_bd['uk_geb'], 
                        "w_erz_art_et": fixed_variables_bd['w_erz_art_et'], 
                        "k_erz_art_rk": fixed_variables_bd['k_erz_art_rk'], 
                        "qg13": fixed_variables_bd['qg13'], 
                        "qi11": fixed_variables_bd['qi11'], 
                        "qg21": fixed_variables_bd['qg21'], 
                        "n_ug": fixed_variables_bd['n_ug'], 
                        "qh1": fixed_variables_bd['qh1'],   
                        "night_flushing_flow": fixed_variables_bd['night_flushing_flow'], 
                        "unteraw_fl": fixed_variables_bd['unteraw_fl'], 
                        "k": fixed_variables_bd['k'], 

                        "q25_1": variables_df["q25_1"].iloc[0],
                        "aw_fl": variables_df["aw_fl"].iloc[0],
                        "qd1": variables_df["qd1"].iloc[0],
                        "facade_area": variables_df["facade_area"].iloc[0],
                        "geb_f_flaeche_n_iwu": variables_df["geb_f_flaeche_n_iwu"].iloc[0],
                        "d_fl_be": variables_df["d_fl_be"].iloc[0],
                        "nrf_2": variables_df["nrf_2"].iloc[0],
                        "ebf": variables_df["ebf"].iloc[0],
                        "n_og": variables_df["n_og"].iloc[0],
                        "geb_f_hoehe_mittel_iwu": variables_df["geb_f_hoehe_mittel_iwu"].iloc[0],    
                        "u_fen": variables_df["u_fen"].iloc[0],
                        "u_aw": variables_df["u_aw"].iloc[0],
                        "d_u_ges": variables_df["d_u_ges"].iloc[0],
                        "u_ug": variables_df["u_ug"].iloc[0],

                        "heat_recovery_efficiency": fixed_variables_bd["heat_recovery_efficiency"], 
                        "thermal_capacitance": fixed_variables_bd["thermal_capacitance"], 
                        "heating_coefficient": fixed_variables_bd['heating_coefficient'],
                        "glass_solar_transmittance": fixed_variables_bd["glass_solar_transmittance"],
                        "qd8": fixed_variables_bd["qd8"],
                        "p_j_lx": fixed_variables_bd['p_j_lx'],
                        "k_L": fixed_variables_bd['k_L']
                        } 

    return pd.DataFrame(be_data_original, index=[0])



''' Parameter values required by the DIBS Preprocessing [List] [SA]'''
def setup_be_data_original_SA(scr_gebaeude_id, fixed_variables_bd, variables_list):
  
    be_data_original = {"scr_gebaeude_id": scr_gebaeude_id, 
                        "bak_grob": fixed_variables_bd['bak_grob'], 
                        "plz": fixed_variables_bd['plz'],
                        "hk_geb": fixed_variables_bd['hk_geb'], 
                        "uk_geb": fixed_variables_bd['uk_geb'], 
                        "w_erz_art_et": fixed_variables_bd['w_erz_art_et'], 
                        "k_erz_art_rk": fixed_variables_bd['k_erz_art_rk'], 
                        "qg13": fixed_variables_bd['qg13'], 
                        "qi11": fixed_variables_bd['qi11'], 
                        "qg21": fixed_variables_bd['qg21'], 
                        "n_ug": fixed_variables_bd['n_ug'], 
                        "qh1": fixed_variables_bd['qh1'],   
                        "night_flushing_flow": fixed_variables_bd['night_flushing_flow'], 
                        "unteraw_fl": fixed_variables_bd['unteraw_fl'], 
                        "k": fixed_variables_bd['k'], 

                        "q25_1": variables_list[0],
                        "aw_fl": variables_list[1],
                        "qd1": variables_list[2],
                        "facade_area": variables_list[3],
                        "geb_f_flaeche_n_iwu": variables_list[4],
                        "d_fl_be": variables_list[5],
                        "nrf_2": variables_list[6],
                        "ebf": variables_list[7],
                        "n_og": variables_list[8],
                        "geb_f_hoehe_mittel_iwu": variables_list[9],    
                        "glass_solar_transmittance": variables_list[10],
                        "qd8": variables_list[11],
                        "u_fen": variables_list[12],
                        "u_aw": variables_list[13],
                        "d_u_ges": variables_list[14],
                        "u_ug": variables_list[15],
                        "heat_recovery_efficiency": variables_list[16], 
                        "thermal_capacitance": variables_list[17], 
                        "heating_coefficient": variables_list[18],
                        "p_j_lx": variables_list[19],
                        "k_L": variables_list[20]
                        } 

    be_data_original = pd.DataFrame(be_data_original, index=[0])
    return be_data_original



''' Parameter intervals [SA, GP]'''
def create_intervals(scr_gebaeude_id):
    
    def get_p_j_lx(k):
        # Create DataFrame of Table 5 DIN V 18599-4:2018-09, S. 24
        p_j_lx_dict ={0.6: [0.045, 0.122],
                        0.7: [0.041, 0.105],
                        0.8: [0.037, 0.09],
                        0.91: [0.035, 0.08],
                        1: [0.033, 0.071],
                        1.2: [0.0298, 0.0606],
                        1.25: [0.029, 0.058],
                        1.5: [0.027, 0.05],
                        2: [0.025, 0.044],
                        2.4: [0.0242, 0.04],
                        2.5: [0.024, 0.039],
                        3: [0.023, 0.037],
                        4: [0.022, 0.035],
                        5: [0.021, 0.033]}
        p_j_lx =  p_j_lx_dict[k] 
        return p_j_lx

    def get_heating_coefficient(w_erz_art_et):
        # Create DataFrame of Table 5 DIN V 18599-4:2018-09, S. 24
        heating_coefficient_dict = {"OilBoiler": [1.004, 1.114],
                                "GasBoiler": [1.028, 1.142],
                                "LGasBoiler": [1.019, 1.129],
                                "BiogasOilBoiler": [1.016, 1.0995],
                                "BioGasBoiler": [1.054, 1.057],
                                "SoilidFuelBioler": [1.054, 1.123],
                                "ElectricHeating": [1, 1],
                                "DistrictHeating": [1.002, 1.002],
                                "SolidFuelLiquidFuelFurnace": [1.428571429, 1.428571429],
                                "GasCHP": [2.04081632, 2.040816327]}
                                #"NoHeating": [0, 0]}
        heating_coefficient =  heating_coefficient_dict[w_erz_art_et]
        return heating_coefficient

    building_inputdata = get_building_inputdata(scr_gebaeude_id)
    fixed_variables = pd.read_excel(os.path.join(paths.DATA_DIR, 'fixed_variables.xlsx'))
    fixed_variables_bd = fixed_variables[fixed_variables['scr_gebaeude_id'] == scr_gebaeude_id].iloc[0]

    w_erz_art_et = fixed_variables_bd['w_erz_art_et']
    k =  fixed_variables_bd['k']

    group_A = [0.9, 1.1]
    group_B = [0.75, 1.25]
    group_C = [0.5, 1.5]

    if fixed_variables_bd["heat_recovery_efficiency"] == 0:
        heat_recovery_efficiency_int = [0, 0.001]
    else:
        heat_recovery_efficiency_int = np.multiply(fixed_variables_bd["heat_recovery_efficiency"], group_A).tolist()


    intervals = {"q25_1": np.multiply(building_inputdata["q25_1"].iloc[0], group_B).tolist(),
                
                "aw_fl": np.multiply(building_inputdata["aw_fl"].iloc[0], group_A).tolist(),
                
                "qd1" : np.multiply(building_inputdata["qd1"].iloc[0], group_A).tolist(),
                
                "facade_area" : np.multiply(building_inputdata["facade_area"].iloc[0], group_A).tolist(),
                
                "geb_f_flaeche_n_iwu": np.multiply(building_inputdata["geb_f_flaeche_n_iwu"].iloc[0], group_C).tolist(), # Changed  from group_A
                
                "d_fl_be": np.multiply(building_inputdata["d_fl_be"].iloc[0], group_A).tolist(),
                
                "nrf_2": np.multiply(building_inputdata["nrf_2"].iloc[0], group_A).tolist(),
                
                "ebf": np.multiply(building_inputdata["ebf"].iloc[0], group_A).tolist(),
                
                "n_og": np.multiply(building_inputdata["n_og"].iloc[0], group_A).tolist(),
                
                "geb_f_hoehe_mittel_iwu": np.multiply(building_inputdata["geb_f_hoehe_mittel_iwu"].iloc[0], group_A).tolist(),
                
                "u_fen": np.multiply(building_inputdata["u_fen"].iloc[0], group_C).tolist(),
                
                "u_aw": np.multiply(building_inputdata["u_aw"].iloc[0], group_C).tolist(),
                
                "d_u_ges": np.multiply(building_inputdata["d_u_ges"].iloc[0], group_C).tolist(),
                
                "u_ug": np.multiply(building_inputdata["u_ug"].iloc[0], group_C).tolist(),
                
                "heat_recovery_efficiency": heat_recovery_efficiency_int,
                
                "heating_coefficient": np.multiply(get_heating_coefficient(w_erz_art_et), group_A).tolist(),
                
                "k_L": [0.8, 1.5], #[1.09, 5.5],
                
                "p_j_lx": get_p_j_lx(k),
                
                "glass_solar_transmittance": [0.53, 0.87],
                
                "qd8": [0.22, 1],
                
                "thermal_capacitance": [110000, 260000]
                }
                
    return intervals



''' Calibration parameters (Most influential ones from SA) '''
def get_SA_parameters(scr_gebaeude_id, num_bc_param, num_samples_sa, output_resolution, training_ratio):
    # Fetch the names of the most sensitive parameters from the SA
    if output_resolution == None:
        total_Si = pd.read_excel(os.path.join(paths.DATA_DIR, "SA/{}/{}/TotalSi_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, num_samples_sa)))  
    if output_resolution != None:
        total_Si = pd.read_excel(os.path.join(paths.DATA_DIR, "SA/{}/{}/TotalSi_{}_obs_train_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, training_ratio, num_samples_sa)))

    total_Si = total_Si.rename(columns={"Unnamed: 0": "parameters"})
    total_Si = total_Si.sort_values(by="ST", ascending=False)
    total_Si = total_Si.reset_index(drop=True)

    SA_param_names = list(total_Si["parameters"][:num_bc_param])

    return total_Si, SA_param_names



''' Open trained Meta-model '''
def gp_metamodel(scr_gebaeude_id, num_bc_param, num_samples_gp, output_resolution, training_ratio, kernel_index):

    if output_resolution==None:
        file_gp = open(os.path.join(paths.DATA_DIR, "GP/{}/{}/{}_bc_param/Trained_GP_{}_{}_kernel.pkl".format(scr_gebaeude_id, output_resolution, num_bc_param, num_samples_gp, kernel_index)), "rb")
    else:
        file_gp = open(os.path.join(paths.DATA_DIR, "GP/{}/{}/{}_bc_param/Trained_GP_{}_{}_obs_train_{}_kernel.pkl".format(scr_gebaeude_id, output_resolution, num_bc_param, num_samples_gp, training_ratio, kernel_index)), "rb")
    
    objects = pickle.load(file_gp)
    file_gp.close()
    
    gaussian_process = objects[0]
    scaler_X = objects[1]
    X_train = objects[2]
    X_test = objects[3]
    y_train = objects[4]
    y_test = objects[5]
    y_pred = objects[6]
    sigma = objects[7]
    mse = objects[8]
    mae = objects[9]
    rmse = objects[10]
    r2 = objects[11]
    mape = objects[12]

    return gaussian_process, scaler_X, X_train, X_test, y_train, y_test, y_pred, sigma, mse, mae, rmse, r2, mape


  
''' Observed data [BC] '''
def get_observations(scr_gebaeude_id, climate_file):

    if climate_file == "AMY_2010_2022.epw":
        observations = pd.read_excel(os.path.join(paths.DATA_DIR, 'building_yearly_Q2.xlsx')) 

    else:
        observations = pd.read_excel(os.path.join(paths.DATA_DIR, 'building_yearly_Q2_normalized.xlsx')) 

    return observations[scr_gebaeude_id].to_numpy()



''' Define: Prior probability distributions [BC]'''
def get_prior_param_values_old(scr_gebaeude_id, num_bc_param, num_samples_gp, output_resolution):
    # Get the be_data_original in order to have the prior parameter values
    variables_df = get_building_inputdata(scr_gebaeude_id)
    be_data_original = setup_be_data_original(scr_gebaeude_id, variables_df)

    file_gp_samples = open(os.path.join(paths.DATA_DIR, "GP/{}/{}_bc_param/Indata_GP_{}_{}.pkl".format(scr_gebaeude_id, num_bc_param, num_samples_gp, output_resolution)), "rb")
    data_gp = pickle.load(file_gp_samples)
    file_gp_samples.close()
    param_names_list = data_gp.drop("heating_energy", axis=1).columns.tolist()

    # Get the prior parameter mean values
    prior_params_mean = dict(filter(lambda item: item[0] in param_names_list, be_data_original.items()))
    intervals = create_intervals(scr_gebaeude_id)

    prior_params_lower = {}
    for i in param_names_list:
        prior_params_lower["{}".format(i)] = intervals["{}".format(i)][0]

    prior_params_upper = {}
    for i in param_names_list:
        prior_params_upper["{}".format(i)] = intervals["{}".format(i)][1]

    prior_params_sigma = {}
    for i in param_names_list:
        prior_params_sigma["{}".format(i)] = (intervals["{}".format(i)][1] - intervals["{}".format(i)][0]) / 3.29

    return param_names_list, prior_params_mean, prior_params_lower, prior_params_upper, prior_params_sigma



''' Define: Prior probability distributions [BC]'''
def get_prior_param_values(scr_gebaeude_id, num_bc_param, num_samples_gp, output_resolution, training_ratio):
    # Get the be_data_original in order to have the prior parameter values
    variables_df = get_building_inputdata(scr_gebaeude_id)
    be_data_original = setup_be_data_original(scr_gebaeude_id, variables_df)
    prior_prob = pd.read_excel(os.path.join(paths.DATA_DIR, 'prior_probability_definition.xlsx'), index_col=0)

    file_gp_samples = open(os.path.join(paths.DATA_DIR, "GP/{}/{}/{}_bc_param/Indata_GP_{}_{}_samples.pkl".format(scr_gebaeude_id, output_resolution, num_bc_param, training_ratio, num_samples_gp)), "rb")
    data_gp = pickle.load(file_gp_samples)
    file_gp_samples.close()
    param_names_list = data_gp.drop("heating_energy", axis=1).columns.tolist()

    # Get the prior parameter mean values
    prior_params_mean = dict(filter(lambda item: item[0] in param_names_list, be_data_original.items()))
    intervals = create_intervals(scr_gebaeude_id)

    prior_params_lower = {}
    for i in param_names_list:
        prior_params_lower["{}".format(i)] = intervals["{}".format(i)][0]

    prior_params_upper = {}
    for i in param_names_list:
        prior_params_upper["{}".format(i)] = intervals["{}".format(i)][1]

    prior_params_sigma = {}
    for i in param_names_list:
        prior_params_sigma["{}".format(i)] = prior_prob.loc['{}'.format(i), 'std'] * (intervals["{}".format(i)][1] - intervals["{}".format(i)][0]) + intervals["{}".format(i)][0]
        
    return param_names_list, prior_params_mean, prior_params_lower, prior_params_upper, prior_params_sigma

    

def CF_factors(scr_gebaeude_id, start_time, end_time):
        fixed_variables = pd.read_excel(os.path.join(paths.DATA_DIR, 'fixed_variables.xlsx'))
        plz = int(fixed_variables[fixed_variables['scr_gebaeude_id'] == scr_gebaeude_id].iloc[0]['plz'])

        Climate_Factor_file = pd.read_excel(os.path.join(paths.DATA_DIR, 'Jahrliche_Klimafaktoren.xlsx'))
        Climate_Factor = Climate_Factor_file.loc[Climate_Factor_file['Postleitzahl'] == plz, start_time:end_time].values.tolist()[0]

        return Climate_Factor

