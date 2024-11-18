
import DIBS.data_preprocessing.breitenerhebung.dataPreprocessingBE as preprocessing
import DIBS.iso_simulator.annualSimulation.annualSimulation as sim
import numpy as np
from datetime import datetime
try:
    import inputs 
except:
    import src.inputs as inputs
try:
    import paths 
except:
    import src.paths as paths


def run_simulation(row, scr_gebaeude_id, be_data_original, variable_names, climate_file, start_time, end_time, output_resolution):
    
    be_data = be_data_original.copy()
    for nr, param in enumerate(variable_names):
        be_data[param] = row[nr]
    building_data = preprocessing.data_preprocessing(be_data)

    if output_resolution == None:
        HeatingEnergy_sum = sim.cal_heating_energy_bd(building_data.iloc[0], climate_file, start_time, end_time, output_resolution)["HeatingEnergy"]
        str_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').year
        fnsh_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').year

        Climate_Factor = inputs.CF_factors(scr_gebaeude_id, str_time, fnsh_time)
        HeatingEnergy_sum = np.divide(HeatingEnergy_sum, Climate_Factor).tolist()

    else:
        HeatingEnergy_sum = sim.cal_heating_energy_bd(building_data.iloc[0], climate_file, start_time, end_time, output_resolution)["HeatingEnergy"].values.tolist()

    return HeatingEnergy_sum