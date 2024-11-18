import pandas as pd
import numpy as np
import time
import os
import pickle
from datetime import datetime
import DIBS.data_preprocessing.breitenerhebung.dataPreprocessingBE as preprocessing
import DIBS.iso_simulator.annualSimulation.annualSimulation as sim
try:
    import inputs 
except:
    import src.inputs as inputs
try:
    import paths 
except:
    import src.paths as paths



def sample_gp(scr_gebaeude_id, num_bc_param, num_samples_sa, num_samples_gp, climate_file, start_time_cal, end_time_cal, output_resolution, training_ratio):

    dir_path = os.path.join(paths.DATA_DIR, "GP", str(scr_gebaeude_id), "{}".format(output_resolution), "{}_bc_param".format(num_bc_param))
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        # Another process in the multiprocessing may have created the directory already; just pass
        pass


    # Get the names of the most sensitive parameters from the SA
    total_SI, SA_param_names = inputs.get_SA_parameters(scr_gebaeude_id, num_bc_param, num_samples_sa, output_resolution, training_ratio)

    # Create the be_data_original, which goes to the DIBS simulation
    variables_df = inputs.get_building_inputdata(scr_gebaeude_id)
    be_data_original = inputs.setup_be_data_original(scr_gebaeude_id, variables_df)

    def model_run(param_comb, be_data_original, climate_file, start_time_cal, end_time_cal):

        # Overwrite the parameters that are considered in the GP
        for i, param in enumerate(param_comb):
            be_data_original[SA_param_names[i]] = param
        building_data = preprocessing.data_preprocessing(be_data_original)

        if output_resolution == None:
            start_time_cal, end_time_cal = None, None
            HeatingEnergy_sum = sim.cal_heating_energy_bd(building_data.iloc[0], climate_file, start_time_cal, end_time_cal, output_resolution)["HeatingEnergy"]

        else:
            HeatingEnergy_sum = sim.cal_heating_energy_bd(building_data.iloc[0], climate_file, start_time_cal, end_time_cal, output_resolution)["HeatingEnergy"].values.tolist()
            
        return HeatingEnergy_sum



    from SALib.sample import latin
    # Geting the intervals for the specific variables
    intervals = inputs.create_intervals(scr_gebaeude_id)

    problem = {
        'num_vars': num_bc_param,
        'names': SA_param_names,
        
        'bounds': [intervals[SA_param_names[i]] for i in range(num_bc_param)]
        }    
                                        

    param_samples = latin.sample(problem, num_samples_gp)
    print('Generated {} parameter combinations.'.format(param_samples.shape[0]))

    df = pd.DataFrame(data = param_samples, columns = SA_param_names)

    # Get Y = HeatingEnergy_sum for each combination
    Y = []
    for i, param_comb in enumerate(param_samples):
        # param_comb is a list of the calibration parameter values
        Y.append(model_run(param_comb, be_data_original, climate_file, start_time_cal, end_time_cal))
        print("PERCENTAGE DONE OF GP SAMPLING {} %: ".format(i/num_samples_gp*100))
    df["heating_energy"] = Y  
    
    df.to_excel(os.path.join(paths.DATA_DIR, "GP/{}/{}/{}_bc_param/Indata_GP_{}_{}_samples.xlsx".format(scr_gebaeude_id, output_resolution, num_bc_param, training_ratio, num_samples_gp)))
    file_gp_samples = open(os.path.join(paths.DATA_DIR, "GP/{}/{}/{}_bc_param/Indata_GP_{}_{}_samples.pkl".format(scr_gebaeude_id, output_resolution, num_bc_param, training_ratio, num_samples_gp)), "wb")
    pickle.dump(df, file_gp_samples)
    file_gp_samples.close()


    return df

