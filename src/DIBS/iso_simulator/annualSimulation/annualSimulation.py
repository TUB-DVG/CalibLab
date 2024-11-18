""" adapted from DIBS - changes:
(1) extract codes relevant to heating energy calculation (2) split the script into smaller functions
Create and last updated by Siling Chen - 05.05.2023
"""
# Import packages
import sys
import os

# mainPath = os.path.abspath(os.path.join(os.path.dirname(__file__), './DIBS/iso_simulator'))
# sys.path.insert(0, mainPath)

mainPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, mainPath)

# Import more packages
import numpy as np
import pandas as pd
try:
    import paths 
except:
    import src.paths as paths
# Import modules
from namedlist import namedlist
from building_physics import Building
import supply_system
import emission_system
from radiation import Location
from radiation import Window
from auxiliary import scheduleReader
from auxiliary import normReader
from auxiliary import TEKReader
from datetime import datetime
import time

# Create namedlist of building_data for further iterations
def iterate_namedlist(building_data):
    Row = namedlist('Gebaeude', building_data.columns)
    for row in building_data.itertuples():
        yield Row(*row[1:])
def initialise_building(i_gebaeudeparameter):
    # Initialise an instance of the building
    BuildingInstance = Building(scr_gebaeude_id = i_gebaeudeparameter.scr_gebaeude_id,
                                plz = i_gebaeudeparameter.plz,
                                hk_geb = i_gebaeudeparameter.hk_geb,
                                uk_geb = i_gebaeudeparameter.uk_geb,
                                max_occupancy = i_gebaeudeparameter.max_occupancy,   # Sensitivity analysis
                                wall_area_og = i_gebaeudeparameter.wall_area_og,
                                wall_area_ug = i_gebaeudeparameter.wall_area_ug,
                                window_area_north = i_gebaeudeparameter.window_area_north,   # Sensitivity analysis
                                window_area_east = i_gebaeudeparameter.window_area_east,   # Sensitivity analysis
                                window_area_south = i_gebaeudeparameter.window_area_south,   # Sensitivity analysis   
                                window_area_west = i_gebaeudeparameter.window_area_west,     # Sensitivity analysis 
                                roof_area = i_gebaeudeparameter.roof_area,
                                net_room_area = i_gebaeudeparameter.net_room_area,
                                base_area = i_gebaeudeparameter.base_area,
                                energy_ref_area = i_gebaeudeparameter.energy_ref_area,
                                building_height = i_gebaeudeparameter.building_height,      
                                lighting_load = i_gebaeudeparameter.lighting_load, 
                                lighting_control = i_gebaeudeparameter.lighting_control,  
                                lighting_utilisation_factor = i_gebaeudeparameter.lighting_utilisation_factor,   
                                lighting_maintenance_factor = i_gebaeudeparameter.lighting_maintenance_factor,  
                                glass_solar_transmittance = i_gebaeudeparameter.glass_solar_transmittance,  
                                glass_solar_shading_transmittance = i_gebaeudeparameter.glass_solar_shading_transmittance,   
                                glass_light_transmittance = i_gebaeudeparameter.glass_light_transmittance,   
                                u_windows = i_gebaeudeparameter.u_windows,   
                                u_walls = i_gebaeudeparameter.u_walls,   
                                u_roof = i_gebaeudeparameter.u_roof,   
                                u_base = i_gebaeudeparameter.u_base,   
                                temp_adj_base = i_gebaeudeparameter.temp_adj_base,
                                temp_adj_walls_ug = i_gebaeudeparameter.temp_adj_walls_ug,
                                ach_inf = i_gebaeudeparameter.ach_inf,
                                ach_win = i_gebaeudeparameter.ach_win,
                                ach_vent = i_gebaeudeparameter.ach_vent,
                                heat_recovery_efficiency = i_gebaeudeparameter.heat_recovery_efficiency,
                                thermal_capacitance = i_gebaeudeparameter.thermal_capacitance,
                                t_start = i_gebaeudeparameter.t_start,
                                t_set_heating = i_gebaeudeparameter.t_set_heating,
                                t_set_cooling = i_gebaeudeparameter.t_set_cooling,
                                night_flushing_flow = i_gebaeudeparameter.night_flushing_flow,
                                max_cooling_energy_per_floor_area = i_gebaeudeparameter.max_cooling_energy_per_floor_area,
                                max_heating_energy_per_floor_area = i_gebaeudeparameter.max_heating_energy_per_floor_area,
                                heating_supply_system = getattr(supply_system, i_gebaeudeparameter.heating_supply_system),  
                                cooling_supply_system = getattr(supply_system, i_gebaeudeparameter.cooling_supply_system),
                                heating_emission_system = getattr(emission_system, i_gebaeudeparameter.heating_emission_system),
                                cooling_emission_system = getattr(emission_system, i_gebaeudeparameter.cooling_emission_system),
                                heating_coefficient = i_gebaeudeparameter.heating_coefficient)
    return BuildingInstance


def cal_heating_energy_bd(i_gebaeudeparameter, climate_file=None, start_time=None, end_time=None, output_resolution=None):
    # i_gebaeudeparameter is a dataframe with all the input parameters to simAnnual (preprocessed)
    ''' if climate file is not given, use the DEU_BE_Potsdam.AP.103790_TMYx.2004-2018.epw 
        if start_time AND end_time are given, trim the DIBS simulation results accordingly
        if output_resolution is given: resample the trimmed DIBS results acordingly; if not, the hourly simulation results will be returned
    '''

    HeatingEnergy = []
    BuildingInstance = initialise_building(i_gebaeudeparameter)

    if (i_gebaeudeparameter.energy_ref_area == -8) | (i_gebaeudeparameter.heating_supply_system == 'NoHeating'):
        print('Building ' + str(i_gebaeudeparameter.scr_gebaeude_id) + ' not heated')
        return #continue
    if output_resolution == None:
        climate_file = 'DEU_BE_Potsdam.AP.103790_TMYx.2004-2018.epw'
        start_time, end_time = None, None # reset start_tim, end_time 
    building_location = Location(epwfile_path=os.path.join(paths.AUX_DIR, 'weather_data', climate_file))
    plz_data = pd.read_csv(os.path.join(paths.AUX_DIR, 'weather_data/plzcodes.csv'), encoding='latin',
                            dtype={'zipcode': int})





    # Pick latitude and longitude from plz_data and put values into a list
    coordinates_plz = plz_data.loc[plz_data['zipcode'] == BuildingInstance.plz, ['latitude', 'longitude']].iloc[0].tolist()


    # coordinate for the weather station (Potsdam; got from the .epw file). Necessary for calc_sun_position()
    latitude_station = coordinates_plz[0]
    longitude_station = coordinates_plz[1]
    # Define windows for each compass direction
    SouthWindow = Window(azimuth_tilt=0, alititude_tilt=90,
                         glass_solar_transmittance=BuildingInstance.glass_solar_transmittance,
                         glass_solar_shading_transmittance=BuildingInstance.glass_solar_shading_transmittance,
                         glass_light_transmittance=BuildingInstance.glass_light_transmittance,
                         area=BuildingInstance.window_area_south)
    EastWindow = Window(azimuth_tilt=90, alititude_tilt=90,
                        glass_solar_transmittance=BuildingInstance.glass_solar_transmittance,
                        glass_solar_shading_transmittance=BuildingInstance.glass_solar_shading_transmittance,
                        glass_light_transmittance=BuildingInstance.glass_light_transmittance,
                        area=BuildingInstance.window_area_east)
    WestWindow = Window(azimuth_tilt=180, alititude_tilt=90,
                        glass_solar_transmittance=BuildingInstance.glass_solar_transmittance,
                        glass_solar_shading_transmittance=BuildingInstance.glass_solar_shading_transmittance,
                        glass_light_transmittance=BuildingInstance.glass_light_transmittance,
                        area=BuildingInstance.window_area_west)
    NorthWindow = Window(azimuth_tilt=270, alititude_tilt=90,
                         glass_solar_transmittance=BuildingInstance.glass_solar_transmittance,
                         glass_solar_shading_transmittance=BuildingInstance.glass_solar_shading_transmittance,
                         glass_light_transmittance=BuildingInstance.glass_light_transmittance,
                         area=BuildingInstance.window_area_north)
    # Get information from DIN V 18599-10 or SIA 2024 for gain_per_person and appliance_gains depending on
    # hk_geb, uk_geb
    # Assignments see Excel/CSV-File in /auxiliary/norm_profiles/profiles_DIN18599_SIA2024
    din = 'din18599'
    sia = 'sia2024'
    mid_values = 'mid'
    profile_from_norm = din  # Choose here where to pick data from
    gains_from_group_values = mid_values  # Choose here here between low, mid or max values
    gain_per_person, appliance_gains, typ_norm = normReader.getGains(BuildingInstance.hk_geb, BuildingInstance.uk_geb,
                                                                     profile_from_norm, gains_from_group_values)
    # Get usage time of the specific building from DIN V 18599-10 or SIA2024
    usage_from_norm = sia
    usage_start, usage_end = normReader.getUsagetime(BuildingInstance.hk_geb, BuildingInstance.uk_geb, usage_from_norm)
    # Read specific occupancy schedule
    # Assignments see Excel/CSV-File in /auxiliary/occupancy_schedules/
    occupancy_schedule, schedule_name = scheduleReader.getSchedule(BuildingInstance.hk_geb, BuildingInstance.uk_geb)
    

    t_m_prev = BuildingInstance.t_start

    building_location.weather_data.index = pd.to_datetime(building_location.weather_data[['year', 'month', 'day', 'hour', 'minute']].astype(str),
                   format='%Y-%m-%d %H:%M')
    # select weather data for simulation
    if start_time != None and end_time != None:
        start_time_ori = start_time

        if not start_time.endswith('-01-01 00:00:00'):
            start_time_ori = start_time
            start_time = '%s-01-01 00:00:00' % pd.to_datetime(start_time).year # reset the start_time to be Jan of the start year
            # print('Start time %s that was originally given, is now set to %s' % (start_time_ori, start_time))
        weather_data_sim = building_location.weather_data.loc[start_time: end_time]
    else:
        weather_data_sim = building_location.weather_data # if start_ and end_time are not simultaneously given, take the whole ts

    # ## Inner Loop: Loop through all hours within the input weather data
    # # set index (time) for weather data ts
    # building_location.weather_data.index = pd.to_datetime(building_location.weather_data[['year', 'month', 'day', 'hour', 'minute']].astype(str),
    #                format='%Y-%m-%d %H:%M')
    # # select weather data for simulation
    # if start_time != None and end_time != None:
    #     weather_data_sim = building_location.weather_data.loc[start_time: end_time]
    # else:
    #     weather_data_sim = building_location.weather_data # if start_ and end_time are not simultaneously given, take the whole ts
    for hour in range(len(weather_data_sim)):
        # Initialize t_set_heating at the beginning of each time step, due to BuildingInstance.t_set_heating = 0 if night flushing is active
        # (Also see below)
        BuildingInstance.t_set_heating = i_gebaeudeparameter.t_set_heating
        # Extract the outdoor temperature in building_location for that hour from weather_data
        t_out = weather_data_sim['drybulb_C'][hour]
        # Call calc_sun_position(). Depending on latitude, longitude, year and hour - Independent from epw weather_data
        Altitude, Azimuth = building_location.calc_sun_position(
            latitude_deg=latitude_station, longitude_deg=longitude_station,
            year=weather_data_sim['year'][hour], hoy=hour)
        # Calculate H_ve_adj, See building_physics for details
        BuildingInstance.h_ve_adj = BuildingInstance.calc_h_ve_adj(hour, t_out, usage_start, usage_end)
        # Set t_set_heating = 0 for the time step, otherwise the heating system heats up during night flushing is on
        # BuildingInstance.t_set_heating = 0
        # Define t_air for calc_solar_gains(). Starting condition (hour==0) necessary for first time step
        if hour == 0:
            t_air = BuildingInstance.t_set_heating
        else:
            t_air = BuildingInstance.t_air
        # Calculate solar gains and illuminance through each window
        SouthWindow.calc_solar_gains(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                     normal_direct_radiation=weather_data_sim[
                                         'dirnorrad_Whm2'][hour],
                                     horizontal_diffuse_radiation=weather_data_sim['difhorrad_Whm2'][
                                         hour],
                                     t_air=t_air, hour=hour)
        SouthWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                     normal_direct_illuminance=weather_data_sim[
                                         'dirnorillum_lux'][hour],
                                     horizontal_diffuse_illuminance=weather_data_sim['difhorillum_lux'][
                                         hour])
        EastWindow.calc_solar_gains(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                    normal_direct_radiation=weather_data_sim[
                                        'dirnorrad_Whm2'][hour],
                                    horizontal_diffuse_radiation=weather_data_sim['difhorrad_Whm2'][hour],
                                    t_air=t_air, hour=hour)
        EastWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                    normal_direct_illuminance=weather_data_sim[
                                        'dirnorillum_lux'][hour],
                                    horizontal_diffuse_illuminance=weather_data_sim['difhorillum_lux'][
                                        hour])
        WestWindow.calc_solar_gains(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                    normal_direct_radiation=weather_data_sim[
                                        'dirnorrad_Whm2'][hour],
                                    horizontal_diffuse_radiation=weather_data_sim['difhorrad_Whm2'][hour],
                                    t_air=t_air, hour=hour)
        WestWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                    normal_direct_illuminance=weather_data_sim[
                                        'dirnorillum_lux'][hour],
                                    horizontal_diffuse_illuminance=weather_data_sim['difhorillum_lux'][
                                        hour])
        NorthWindow.calc_solar_gains(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                     normal_direct_radiation=weather_data_sim[
                                         'dirnorrad_Whm2'][hour],
                                     horizontal_diffuse_radiation=weather_data_sim['difhorrad_Whm2'][
                                         hour],
                                     t_air=t_air, hour=hour)
        NorthWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                     normal_direct_illuminance=weather_data_sim[
                                         'dirnorillum_lux'][hour],
                                     horizontal_diffuse_illuminance=weather_data_sim['difhorillum_lux'][
                                         hour])
        # Occupancy for the time step
        occupancy = occupancy_schedule.loc[hour, 'People'] * BuildingInstance.max_occupancy
        # Calculate the lighting of the building for the time step
        BuildingInstance.solve_building_lighting(illuminance=
                                                 SouthWindow.transmitted_illuminance +
                                                 EastWindow.transmitted_illuminance +
                                                 WestWindow.transmitted_illuminance +
                                                 NorthWindow.transmitted_illuminance,
                                                 occupancy=occupancy)
        # Calculate gains from occupancy and appliances
        internal_gains = occupancy * gain_per_person + \
                         appliance_gains * occupancy_schedule.loc[
                             hour, 'Appliances'] * BuildingInstance.energy_ref_area + \
                         BuildingInstance.lighting_demand
        # Calculate energy demand for the time step
        BuildingInstance.solve_building_energy(internal_gains=internal_gains,
                                               solar_gains=
                                               SouthWindow.solar_gains +
                                               EastWindow.solar_gains +
                                               WestWindow.solar_gains +
                                               NorthWindow.solar_gains,
                                               t_out=t_out, t_m_prev=t_m_prev)
        # Set the previous temperature for the next time step
        t_m_prev = BuildingInstance.t_m_next
        HeatingEnergy.append(BuildingInstance.heating_energy/1000)

        # DataFrame with hourly results of specific building
    hourlyResults = pd.DataFrame(index=weather_data_sim.index)
    hourlyResults['HeatingEnergy'] = HeatingEnergy

    if start_time != None and end_time != None:
        hourlyResults = hourlyResults.loc[start_time_ori: end_time]
    
    if output_resolution != None:
        final_output = hourlyResults.resample(output_resolution).sum()
        return final_output
    else:
        return hourlyResults.sum()

    # # hier endet die Inner Loop
    # # DataFrame with hourly results of specific building
    # if hourly == True:
    #     hourlyResults = pd.DataFrame(index=weather_data_sim.index)
    #     hourlyResults['HeatingEnergy'] = HeatingEnergy
    #     return hourlyResults
    # else:
    #     hourlyResults = pd.DataFrame({
    #                         'HeatingEnergy': HeatingEnergy
    #                     })
    #     return hourlyResults.HeatingEnergy.sum()
    
def save_dfs_dict(dictex, folder=None):
    if folder == None:
        folder = 'results'
    if not os.path.exists(os.path.join(paths.RES_DIR, folder)):
        os.makedirs(os.path.join(paths.RES_DIR, folder))
    for key, val in dictex.items():
        val.to_excel(os.path.join(paths.RES_DIR, folder, '{}.xlsx'.format(str(key))))
    print("All saved results can be found in folder", os.path.join(paths.DIBS_DIR, folder))
    return
# Create dictionary to store final DataFrames of the buildings


if __name__ == '__main__':
    pass
