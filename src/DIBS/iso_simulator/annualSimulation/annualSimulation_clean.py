""" adapted from DIBS - changes:
(1) extract codes relevant to heating energy calculation (2) split the script into smaller functions
Create and last updated by Siling Chen - 05.05.2023
"""
# Import packages
import sys
import os

mainPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, mainPath)

# Import more packages
import numpy as np
import pandas as pd
import paths
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

import time

# Create namedlist of building_data for further iterations
def iterate_namedlist(building_data):
    Row = namedlist('Gebaeude', building_data.columns)
    for row in building_data.itertuples():
        yield Row(*row[1:])
def initialise_building(i_gebaeudeparameter):
    # Initialise an instance of the building
    i_gebaeudeparameter = i_gebaeudeparameter.iloc[0]
    BuildingInstance = Building(scr_gebaeude_id=i_gebaeudeparameter.scr_gebaeude_id,
                                plz=i_gebaeudeparameter.plz,
                                hk_geb=i_gebaeudeparameter.hk_geb,
                                uk_geb=i_gebaeudeparameter.uk_geb,
                                max_occupancy=i_gebaeudeparameter.max_occupancy,
                                wall_area_og=i_gebaeudeparameter.wall_area_og,
                                wall_area_ug=i_gebaeudeparameter.wall_area_ug,
                                window_area_north=i_gebaeudeparameter.window_area_north,
                                window_area_east=i_gebaeudeparameter.window_area_east,
                                window_area_south=i_gebaeudeparameter.window_area_south,
                                window_area_west=i_gebaeudeparameter.window_area_west,
                                roof_area=i_gebaeudeparameter.roof_area,
                                net_room_area=i_gebaeudeparameter.net_room_area,
                                base_area=i_gebaeudeparameter.base_area,
                                energy_ref_area=i_gebaeudeparameter.energy_ref_area,
                                building_height=i_gebaeudeparameter.building_height,
                                lighting_load=i_gebaeudeparameter.lighting_load,
                                lighting_control=i_gebaeudeparameter.lighting_control,
                                lighting_utilisation_factor=i_gebaeudeparameter.lighting_utilisation_factor,
                                lighting_maintenance_factor=i_gebaeudeparameter.lighting_maintenance_factor,
                                glass_solar_transmittance=i_gebaeudeparameter.glass_solar_transmittance,
                                glass_solar_shading_transmittance=i_gebaeudeparameter.glass_solar_shading_transmittance,
                                glass_light_transmittance=i_gebaeudeparameter.glass_light_transmittance,
                                u_windows=i_gebaeudeparameter.u_windows,
                                u_walls=i_gebaeudeparameter.u_walls,
                                u_roof=i_gebaeudeparameter.u_roof,
                                u_base=i_gebaeudeparameter.u_base,
                                temp_adj_base=i_gebaeudeparameter.temp_adj_base,
                                temp_adj_walls_ug=i_gebaeudeparameter.temp_adj_walls_ug,
                                ach_inf=i_gebaeudeparameter.ach_inf,
                                ach_win=i_gebaeudeparameter.ach_win,
                                ach_vent=i_gebaeudeparameter.ach_vent,
                                heat_recovery_efficiency=i_gebaeudeparameter.heat_recovery_efficiency,
                                thermal_capacitance=i_gebaeudeparameter.thermal_capacitance,
                                t_start=i_gebaeudeparameter.t_start,
                                t_set_heating=i_gebaeudeparameter.t_set_heating,
                                t_set_cooling=i_gebaeudeparameter.t_set_cooling,
                                night_flushing_flow=i_gebaeudeparameter.night_flushing_flow,
                                max_cooling_energy_per_floor_area=i_gebaeudeparameter.max_cooling_energy_per_floor_area,
                                max_heating_energy_per_floor_area=i_gebaeudeparameter.max_heating_energy_per_floor_area,
                                heating_supply_system=getattr(supply_system, i_gebaeudeparameter.heating_supply_system),
                                cooling_supply_system=getattr(supply_system, i_gebaeudeparameter.cooling_supply_system),
                                heating_emission_system=getattr(emission_system,
                                                                i_gebaeudeparameter.heating_emission_system),
                                cooling_emission_system=getattr(emission_system,
                                                                i_gebaeudeparameter.cooling_emission_system),
                                heating_coefficient=i_gebaeudeparameter.heating_coefficient)
    
    print("i_gebaeudeparameter.energy_ref_area: ", i_gebaeudeparameter.energy_ref_area)
                                
                                
    return BuildingInstance

def cal_heating_energy_bd(i_gebaeudeparameter, getEPWFile_list = ('DEU_BE_Potsdam.AP.103790_TMYx.2004-2018.epw', [52.38300,13.06600])):
    HeatingEnergy = []
    BuildingInstance = initialise_building(i_gebaeudeparameter)
    # If there's no heated area (energy_ref_area == -8) or no heating supply system (heating_supply_system == 'NoHeating')
    # no heating demand can be calculated. In this case skip calculation and proceed with next building.
    if (i_gebaeudeparameter.energy_ref_area == 0) | (i_gebaeudeparameter.heating_supply_system == 'NoHeating'):
        print('Building ' + str(i_gebaeudeparameter.scr_gebaeude_id) + ' not heated')
        return #continue
    if getEPWFile_list == None:
        # Initialize the buildings location with a weather file from the nearest weather station depending on the plz
        getEPWFile_list = Location.getEPWFile(BuildingInstance.plz)
    epw_filename = getEPWFile_list[0]
    building_location = Location(epwfile_path=os.path.join(paths.AUX_DIR, 'weather_data', epw_filename))
    # Extract coordinates of that weather station. Necessary for calc_sun_position()
    latitude_station = getEPWFile_list[1][0]
    longitude_station = getEPWFile_list[1][1]
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
    ## Inner Loop: Loop through all 8760 hours of the year
    for hour in range(8760):
        # Initialize t_set_heating at the beginning of each time step, due to BuildingInstance.t_set_heating = 0 if night flushing is active
        # (Also see below)
        BuildingInstance.t_set_heating = i_gebaeudeparameter.t_set_heating
        # Extract the outdoor temperature in building_location for that hour from weather_data
        t_out = building_location.weather_data['drybulb_C'][hour]
        # Call calc_sun_position(). Depending on latitude, longitude, year and hour - Independent from epw weather_data
        Altitude, Azimuth = building_location.calc_sun_position(
            latitude_deg=latitude_station, longitude_deg=longitude_station,
            year=building_location.weather_data['year'][hour], hoy=hour)
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
                                     normal_direct_radiation=building_location.weather_data[
                                         'dirnorrad_Whm2'][hour],
                                     horizontal_diffuse_radiation=building_location.weather_data['difhorrad_Whm2'][
                                         hour],
                                     t_air=t_air, hour=hour)
        SouthWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                     normal_direct_illuminance=building_location.weather_data[
                                         'dirnorillum_lux'][hour],
                                     horizontal_diffuse_illuminance=building_location.weather_data['difhorillum_lux'][
                                         hour])
        EastWindow.calc_solar_gains(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                    normal_direct_radiation=building_location.weather_data[
                                        'dirnorrad_Whm2'][hour],
                                    horizontal_diffuse_radiation=building_location.weather_data['difhorrad_Whm2'][hour],
                                    t_air=t_air, hour=hour)
        EastWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                    normal_direct_illuminance=building_location.weather_data[
                                        'dirnorillum_lux'][hour],
                                    horizontal_diffuse_illuminance=building_location.weather_data['difhorillum_lux'][
                                        hour])
        WestWindow.calc_solar_gains(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                    normal_direct_radiation=building_location.weather_data[
                                        'dirnorrad_Whm2'][hour],
                                    horizontal_diffuse_radiation=building_location.weather_data['difhorrad_Whm2'][hour],
                                    t_air=t_air, hour=hour)
        WestWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                    normal_direct_illuminance=building_location.weather_data[
                                        'dirnorillum_lux'][hour],
                                    horizontal_diffuse_illuminance=building_location.weather_data['difhorillum_lux'][
                                        hour])
        NorthWindow.calc_solar_gains(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                     normal_direct_radiation=building_location.weather_data[
                                         'dirnorrad_Whm2'][hour],
                                     horizontal_diffuse_radiation=building_location.weather_data['difhorrad_Whm2'][
                                         hour],
                                     t_air=t_air, hour=hour)
        NorthWindow.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                     normal_direct_illuminance=building_location.weather_data[
                                         'dirnorillum_lux'][hour],
                                     horizontal_diffuse_illuminance=building_location.weather_data['difhorillum_lux'][
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
        HeatingEnergy.append(BuildingInstance.heating_energy)
    # hier endet die Inner Loop
    # DataFrame with hourly results of specific building
    hourlyResults = pd.DataFrame({
        'HeatingEnergy': HeatingEnergy
    })
    return hourlyResults

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
