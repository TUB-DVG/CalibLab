from datetime import datetime
import numpy as np
import pandas as pd
import os
# os.chdir(os.path.join(os.getcwd(), 'src'))
import inputs
import time
import DIBS_sim_modified.data_preprocessing.breitenerhebung.dataPreprocessingBE as preprocessing
import DIBS_sim_modified.iso_simulator.annualSimulation.annualSimulation as sim
from DIBS_sim_modified.iso_simulator.annualSimulation.annualSimulation import iterate_namedlist
try:
    import paths 
except:
    import src.paths as paths
try:
    import inputs 
except:
    import src.inputs as inputs


''' For Potsdam, TRY: climate_file, start_time, end_time, output_resolution = None, None, None, None '''

def model_run(scr_gebaeude_id, climate_file, start_time, end_time, output_resolution, training_ratio):
 
    dir_path = os.path.join(paths.RES_DIR, "DIBS_sim", str(scr_gebaeude_id), str(output_resolution), str(training_ratio))

    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        # Another process in the multiprocessing may have created the directory already; just pass
        pass


    variables_df = inputs.get_building_inputdata(scr_gebaeude_id)
    be_data_original = inputs.setup_be_data_original(scr_gebaeude_id, variables_df)
    building_data = preprocessing.data_preprocessing(be_data_original)
    HeatingEnergy_sum = sim.cal_heating_energy_bd(building_data.iloc[0], climate_file, start_time, end_time, output_resolution) 


    if output_resolution != None :
        start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

        HeatingEnergy_sum.to_csv(os.path.join(paths.RES_DIR, "DIBS_sim/{}/{}/{}/HeatingEnergy_{}-{}-{}_{}-{}-{}.csv".format(scr_gebaeude_id, output_resolution, training_ratio, start.year, start.month, start.day, end.year, end.month, end.day)), sep=';')


    else:
        str_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').year
        fnsh_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').year

        Climate_Factor = inputs.CF_factors(scr_gebaeude_id, str_time, fnsh_time)
        HeatingEnergy_sum = np.divide(HeatingEnergy_sum.values, Climate_Factor)

        dates = pd.date_range(start=f'{str_time}-12-31', periods=len(Climate_Factor), freq='Y')
        df = pd.DataFrame({'': dates.strftime('%Y-%m-%d'), 'HeatingEnergy': HeatingEnergy_sum})

        start = datetime.strptime(f'{str_time}-12-31', '%Y-%m-%d')
        end = datetime.strptime(f'{fnsh_time}-12-31', '%Y-%m-%d')
        df.to_csv(os.path.join(paths.RES_DIR, "DIBS_sim/{}/{}/{}/HeatingEnergy_{}-{}-{}_{}-{}-{}.csv".format(scr_gebaeude_id, output_resolution, training_ratio, start.year, start.month, start.day, end.year, end.month, end.day)), sep=';', index=False)

    return HeatingEnergy_sum







