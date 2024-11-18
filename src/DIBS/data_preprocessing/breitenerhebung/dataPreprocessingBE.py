# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:05:52 2023

@author: Kata
"""

import pandas as pd
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path

import platform 
# To Do check os and then write import 
if platform.system() == "Windows":
    import paths 
else: 
    try:
        import paths 
    except:
        import src.paths as paths

pd.options.mode.chained_assignment = None

def data_preprocessing(be_data_original):
    path = os.path.join(paths.DIBS_DIR, 'data_preprocessing/breitenerhebung/BE_data')
      
    # Import Data from DIN V 18599-10:2018-09, DIN V 18599-4:2018-09
    data_18599_10_4 = pd.read_csv(path+ "/profile_18599_10_data.csv", sep = ';', encoding= 'unicode_escape', decimal=",")
    profile_zuweisung_18599_10 = pd.read_csv(path+ "/profile_18599_10_zuweisung.csv", sep = ';', encoding= 'unicode_escape', decimal=",")
    
    # Gebäude-ID (scr_gebaeude_id)
    ##############################################################################
    # Create DataFrame building_data with column 'scr_gebaeude_id' including all buildings from be_data_original
    building_data = be_data_original[['scr_gebaeude_id']]
    
    # PLZ (plz)
    ##############################################################################
    # Map 'scr_plz' from be_data_original to building_data and name column 'plz'
    building_data['plz'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['plz'])

    
    # Gebäudefunktion Hauptkategorie/Unterkategorie (hk_geb, uk_geb)
    ##############################################################################
    # Map 'HK_Geb' and 'UK_Geb' to be_data_original
    building_data['hk_geb'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['hk_geb']).astype(str)
    building_data['uk_geb'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['uk_geb']).astype(str)
    
    
    # Map Profile from DIN V 18599-10:2018-09 according to profile_zuweisung_18599_10 (See: normData.xlsx)
    # Used for further calculations
    building_data['typ_18599'] = building_data['uk_geb'].map(profile_zuweisung_18599_10.set_index('uk_geb')['typ_18599'])

    # Maximale Personenbelegung (max_occupancy)
    ##############################################################################
    # Map 'q25_1'(max. Personenbelegung) from be_data_original to building_data as column 'max_occupancy'
    building_data['max_occupancy'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['q25_1'])
    # from R as Integer
    
    
    # Oberirdische Außenwandfläche (wall_area_og)
    ##############################################################################
    # Map 'aw_fl' from be_data_original to building_data as column 'wall_area_og'
    building_data['wall_area_og'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['aw_fl'])
    # from R as Double
    
    # Unterirdische Außenwandfläche (wall_area_ug)
    ##############################################################################
    # Map 'unteraw_fl' from be_data_original to building_data as column 'wall_area_ug'
    building_data['wall_area_ug'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['unteraw_fl'])
    # from R as Double
    
    
    # Fensterflächen (window_area_north, window_area_east, window_area_south, window_area_west)
    ##############################################################################
    # Map windows share and building area for each direction to building_data    
    building_data['Fen_ant'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['qd1'])
    
    # Map facade area from building_data
    building_data['facade_area'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['facade_area'])
    
    # Map northern facade area from building data
    building_data['geb_f_flaeche_n_iwu'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['geb_f_flaeche_n_iwu'])
    building_data['geb_f_flaeche_s_iwu'] = building_data['geb_f_flaeche_n_iwu']
    
    # Calculate east and west facade area
    building_data['geb_f_flaeche_w_iwu'] = building_data['facade_area']/2 - building_data['geb_f_flaeche_n_iwu']
    building_data['geb_f_flaeche_o_iwu'] = building_data['geb_f_flaeche_w_iwu']
    
    
    # Calculate window area for each directions                                                         
    building_data['window_area_north'] = (building_data['Fen_ant']) * building_data['geb_f_flaeche_n_iwu'] 
    building_data['window_area_east'] = (building_data['Fen_ant']) * building_data['geb_f_flaeche_o_iwu'] 
    building_data['window_area_south'] = (building_data['Fen_ant']) * building_data['geb_f_flaeche_s_iwu'] 
    building_data['window_area_west'] = (building_data['Fen_ant']) * building_data['geb_f_flaeche_w_iwu'] 
    
    
    # Dachfläche (roof_area)
    ##############################################################################
    # Map 'D_fl_be' from be_data_original to building_data as column 'roof_area'
    building_data['roof_area'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['d_fl_be']) 
    # from R as Double
    
    # Netto-Raumfläche (net_room_area)
    ##############################################################################
    # Map 'nrf_2' from be_data_original to building_data as column 'net_room_area'
    building_data['net_room_area'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['nrf_2']) 
    # from R as Double
    
    
    # Energiebezugsfläche (energy_ref_area)
    ##############################################################################
    # Map 'ebf' from be_data_original to building_data as column 'energy_ref_area'
    building_data['energy_ref_area'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['ebf']) 
    # from R as Double
    
        
    # Fläche des unteren Gebäudeabschlusses (base_area)
    ##############################################################################
    # Map 'Mittlere Anzahl oberidrische Geschosse' from be_data_original to building_data as column 'Mittlere Anzahl oberidrische Geschosse'
    building_data['n_OG'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['n_og']) 
    # from R as Double
    
    # Calculate base_area
    building_data['base_area'] = building_data['energy_ref_area'] / (building_data['n_OG'] * 0.87)
    
    
    # Mittlere Gebäudehöhe (building_height)
    ##############################################################################
    # Map 'geb_f_hoehe_mittel_iwu' from be_data_original to building_data as column 'building_height'
    building_data['building_height'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['geb_f_hoehe_mittel_iwu'])
    # from R as Double
    
    
    # spezifische Beleuchtungsleistung (lighting_load)
    ##############################################################################
    # Create Subset for calculation of lighting_load 
    subset_lighting_load = building_data[['scr_gebaeude_id', 'typ_18599']]

    subset_lighting_load['k_L'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['k_L'])    # Interval: 0.465 - 5.5
    # Map 'E_m', 'k_A', 'k_VB', 'k_WF', 'k' from data_18599_10_4 as new columns to subset_lighting_load
    subset_lighting_load['E_m'] = subset_lighting_load['typ_18599'].map(data_18599_10_4.set_index('typ_18599')['E_m'])
    subset_lighting_load['k_A'] = subset_lighting_load['typ_18599'].map(data_18599_10_4.set_index('typ_18599')['k_A'])
    subset_lighting_load['k_VB'] = subset_lighting_load['typ_18599'].map(data_18599_10_4.set_index('typ_18599')['k_VB'])
    subset_lighting_load['k_WF'] = subset_lighting_load['typ_18599'].map(data_18599_10_4.set_index('typ_18599')['k_WF'])
    subset_lighting_load['p_j_lx'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['p_j_lx'])
    # Calculate 'p_j'
    # p_j = p_j_lx * E_m * k_WF * k_A * k_L * k_VB [See: DIN V 18599-4:2018-09, S. 25]
    subset_lighting_load['p_j'] = subset_lighting_load['p_j_lx'] * subset_lighting_load['E_m'] * \
                                    subset_lighting_load['k_WF'] * subset_lighting_load['k_L'] * subset_lighting_load['k_VB']
    
    # Map 'p_j' from subset_lighting_load as column 'lighting_load' to building_data
    building_data['lighting_load'] = building_data['scr_gebaeude_id'].map(subset_lighting_load.set_index('scr_gebaeude_id')['p_j'])
    
    # Lichtausnutzungsgrad der Verglasung (lighting_control)
    ##############################################################################
    # Map 'E_m' from data_18599_10_4 to building_data
    building_data['lighting_control'] = building_data['typ_18599'].map(data_18599_10_4.set_index('typ_18599')['E_m'])
    
    
    # Wartungsfaktor der Verglasung (lighting_utilisation_factor) 
    ##############################################################################
    # Assumption [See Szokolay (1980): Environmental Science Handbook for Architects and Builders, p. 104ff.]
    building_data['lighting_utilisation_factor'] = 0.45
    
    # Wartungsfaktor der Fensterflächen (lighting_maintenance_factor) 
    ##############################################################################
    # See Szokolay (1980): Environmental Science Handbook for Architects and Builders, p. 109
    def set_lighting_maintenance_factor(row):
        if row['hk_geb'] == 'Produktions-, Werkstatt-, Lager- oder Betriebsgebäude':
            lighting_maintenance_factor = 0.8
        else:
            lighting_maintenance_factor = 0.9
        return lighting_maintenance_factor
    building_data['lighting_maintenance_factor'] = building_data.apply(set_lighting_maintenance_factor, axis = 1)    

    # Energiedurchlassgrad der Verglasung (glass_solar_transmittance)
    ##############################################################################
    # Map 'fen_glasart_1' from be_data_original as column 'glass_solar_transmittance' to building_data
    building_data['glass_solar_transmittance'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['glass_solar_transmittance'])
    
    # Fen_glasart_1
    # k_1 = 0.7 if no further information is available [See DIN V 18599-4:2018-09, p.39]
    building_data['k_1'] =  0.7
    
    # k_3 = 0.85 [See DIN V 18599-4:2018-09, p.39]
    building_data['k_3'] =  0.85
    
    def assign_tau_D65SNA(row):
        if row['glass_solar_transmittance'] < 0.615:  # PH-Fenster
            tau_D65SNA = 0.705
        elif row['glass_solar_transmittance'] < 0.74:   # 3-S-Glas Fenster
            tau_D65SNA = 0.75  
        elif row['glass_solar_transmittance'] < 0.825:  # 2-S-Glas Fenster
            tau_D65SNA = 0.82
        else: 
            tau_D65SNA = 0.9    # 1-S-Glas Fenster
        return tau_D65SNA
    
            
    building_data['tau_D65SNA'] = building_data.apply(assign_tau_D65SNA, axis = 1)   
        
    building_data['glass_light_transmittance'] = building_data['k_1'] * building_data['lighting_maintenance_factor'] * building_data['k_3'] * building_data['tau_D65SNA']
   
    

    # Energiedurchlassgrad der Verglasung bei aktiviertem Sonnenschutz (glass_solar_shading_transmittance)
    ##############################################################################
    # Map 'qd8' from be_data_original as column 'qD8' to building_data
    building_data['qD8'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['qd8'])
    building_data['glass_solar_shading_transmittance'] = building_data['glass_solar_transmittance'] * building_data['qD8']



    
    # U-Wert Fenster (u_windows)
    ##############################################################################
    building_data['u_windows'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['u_fen'])
    # from R as Double
    
    # U-Wert Außenwände (u_walls)
    ##############################################################################
    building_data['u_walls'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['u_aw'])
    # from R as Double
    
    
    # U-Wert Dach (u_roof)
    ##############################################################################
    building_data['u_roof'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['d_u_ges'])
    # from R as Double
    
    
    # U-Wert Bodenplatte/Kellerdecke (u_base)
    ##############################################################################
    building_data['u_base'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['u_ug'])
    # from R as Double
    
    
    # Temperaturanpassungsfaktor unterer Gebäudeabschluss (temp_adj_base)
    ##############################################################################
    building_data['n_UG'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['n_ug'])
    # from R as Double
    
    # Find corresponding case according to DIN V 4108-6:2003-06
    def fall_temp_adj_base(row):
        if row['n_UG'] == 0:                                                           # If there's no basement, Case 12
            return_value = 12                                                                       
        else:
            return_value = 16                                                          # Case 16, not heated
        return return_value
    
    building_data['case_temp_adj_base'] = building_data.apply(fall_temp_adj_base, axis = 1)   
    
    arrays_fx = [['<5', '<5', '5 bis 10', '5 bis 10',  '>10', '>10'], ['<=1', '>1', '<=1', '>1', '<=1', '>1']]
    tuples_fx = list(zip(*arrays_fx))
    
    index_fx = pd.MultiIndex.from_tuples(tuples_fx, names = ['B', 'R'])
    fx = pd.DataFrame(([0.3, 0.45, 0.25, 0.4, 0.2, 0.35],
                        [0.4, 0.6, 0.4, 0.6, 0.4, 0.6],
                        [0.45, 0.6, 0.4, 0.5, 0.25, 0.35],
                        [0.55, 0.55, 0.5, 0.5, 0.45, 0.45],
                        [0.7, 0.7, 0.65, 0.65, 0.55, 0.55]), index=['10', '11', '12', '15', '16'], columns=index_fx)
    fx = fx.unstack().reset_index()
    fx = fx.rename(columns = {'level_2': 'case_temp_adj', 0: 'temp_adj_factors'})
    fx['case_temp_adj'] = fx['case_temp_adj'].astype(int)
    
    building_data['building_length_n'] = building_data['geb_f_flaeche_n_iwu'] / building_data['building_height']
    building_data['building_length_s'] = building_data['geb_f_flaeche_s_iwu'] / building_data['building_height']
    building_data['building_length_o'] = building_data['geb_f_flaeche_o_iwu'] / building_data['building_height']
    building_data['building_length_w'] = building_data['geb_f_flaeche_w_iwu'] / building_data['building_height']
    
    building_data['B_raw'] = (2 * building_data[['building_length_n', 'building_length_s']].values.max(1)) + (2 * building_data[['building_length_o', 'building_length_w']].values.max(1))
    
    def clean_B(row):
        if row['B_raw'] < 5:
            value = '<5'
        elif 5 <= row['B_raw'] <= 10:
            value = '5 bis 10'   
        else:
            value = '>10'
        return value    
    building_data['B'] = building_data.apply(clean_B, axis = 1) 
    
    building_data['R_raw'] = 1 / building_data['u_base']
    def clean_R(row):
        if row['R_raw'] <= 1:
            value = '<=1'
        else:
            value = '>1'
        return value  
    building_data['R'] = building_data.apply(clean_R, axis = 1)
    
    building_data = pd.merge(building_data, fx, left_on = ['B', 'R', 'case_temp_adj_base'], right_on = ['B', 'R', 'case_temp_adj'], how = 'left' )
    
    
    # Temperaturanpassungsfaktor unterirdische Außenwandflächen (temp_adj_walls_ug)
    ##############################################################################
    def case_temp_adj_walls_ug(row):
        if row['n_UG'] > 0:
            return_value = 11
        else: 
            return_value = 0
        return return_value
     
    building_data['case_temp_adj_walls_ug'] = building_data.apply(case_temp_adj_walls_ug, axis = 1)  
       
    building_data = pd.merge(building_data, fx, left_on = ['B', 'R', 'case_temp_adj_walls_ug'], right_on = ['B', 'R', 'case_temp_adj'], how = 'left' )
    building_data = building_data.drop(['case_temp_adj_x', 'case_temp_adj_y', 'n_UG'], axis = 1)     
    building_data = building_data.rename(columns = {'temp_adj_factors_x': 'temp_adj_base', 
                                              'temp_adj_factors_y': 'temp_adj_walls_ug'})
    
    # Fill NaNs with 0 
    building_data['temp_adj_walls_ug'] = building_data['temp_adj_walls_ug'].replace(np.nan, 0)
    
    
    # Luftwechselrate Infiltration (ach_inf)
    ##############################################################################
    # Map minimum flow rate from data_18599_10_4 to building_data
    building_data['V_min_18599'] = building_data['typ_18599'].map(data_18599_10_4.set_index('typ_18599')['Außenluftvolumenstrom'])
    
    # Map bak_grob
    # 1: Altbau bis einschl. 1978
    # 2: 1979 - 2009 (1. Wärmeschutzverordnung vor 2010)
    # 3: Neubau ab 2010
    building_data['bak_grob'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['bak_grob'])
    
    # Min flow from DIN V 18599-10 in m³/hm², multiply by m²/m³ for air change rate
    building_data['ach_min'] = (building_data['V_min_18599'] * building_data['net_room_area']) / (building_data['net_room_area'] * (building_data['building_height']/building_data['n_OG']))
    building_data['qH1'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['qh1'])

    def assign_n_50_standard_av(row):
        if row['bak_grob'] == 3:
            if row['qH1'] == 'Ja, zentrale Anlage(n) vorhanden':
                n_50_standard_av = 1
            else:
                n_50_standard_av = 2
        elif row['bak_grob'] == 2:
            n_50_standard_av = 4
        else:
            n_50_standard_av = 6
        return n_50_standard_av  
       
    building_data['n_50_standard_av'] = building_data.apply(assign_n_50_standard_av, axis = 1)  
          
    building_data['standard av-verhältnis'] = 0.9
    
    # AV-Verhältnis Gebäude = Thermische Hüllfläche des Gebäudes / beheiztes Bruttogebäudevolumen
    
    building_data['av-verhältnis'] = (building_data['facade_area'] + building_data['roof_area'] + (building_data['base_area'])) / (building_data['base_area'] * building_data['building_height'])
    
    # Luftdichtheit n50
    building_data['n_50'] = building_data['n_50_standard_av'] * building_data['av-verhältnis'] / building_data['standard av-verhältnis']
    
    # Abschätzung der Infiltrationsluftwechselrate nach ISO 13789 bzw. der früheren EN 832
    # ach_inf = n_50 * e * fATD
    # mit e = 0.07 (DIN V 18599-2, S. 58)
    # mit fATD = 1 (keine Außenluftdurchlässe: Annahme, da keine Informationen vorhanden)
    building_data['ach_inf'] = building_data['n_50'] * 0.07
    
    
    # Luftwechselrate Fenster (ach_win)
    ##############################################################################
    def calc_ach_win(row):
        if row['qH1'] in ('Nein, Fensterlüftung', 'Nein, nur dezentrale Anlage(n) vorhanden', 'Weiß nicht'):
            ach_win = max(0.1, (row['ach_min'] - row['ach_inf']))
        else:
            ach_win = 0.1
        return ach_win  
    building_data['ach_win'] = building_data.apply(calc_ach_win, axis = 1) 
    
    
    # Luftwechselrate RLT (ach_vent)
    ##############################################################################
    def calc_ach_vent(row):
        if row['qH1'] == 'Ja, zentrale Anlage(n) vorhanden':
            ach_vent = max(0.1, (row['ach_min'] - row['ach_inf']))
        else:
            ach_vent = 0.1
        return ach_vent  
    building_data['ach_vent'] = building_data.apply(calc_ach_vent, axis = 1) 

    # Wärmerückgewinnung - originally from qh3_1
    building_data['heat_recovery_efficiency'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['heat_recovery_efficiency'])

    # Wärmespeicherfähigkeit (thermal_capacitance)
    ##############################################################################
    
    building_data['thermal_capacitance'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['thermal_capacitance'])
    
    
    # Sollwert Heizung (t_set_heating)
    ##############################################################################
    building_data['t_set_heating'] = building_data['typ_18599'].map(data_18599_10_4.set_index('typ_18599')['Raum-Solltemperatur Heizung'])
    
    
    # Anfangstemperatur im Gebäude (t_start)
    ##############################################################################
    building_data['t_start'] = building_data['t_set_heating']
    
    
    # Sollwert Kühlung (t_set_cooling)
    ##############################################################################
    building_data['t_set_cooling'] = building_data['typ_18599'].map(data_18599_10_4.set_index('typ_18599')['Raum-Solltemperatur Kühlung'])
    
    
    # Nachtlüftung (night_flushing_flow) 
    # Before freie_kuehlung
    ##############################################################################
    building_data['night_flushing_flow'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['night_flushing_flow'])
  
    
    
    # Max. Heizlast (max_heating_energy_per_floor_area)
    ##############################################################################
    # Set to inf (no further information available)
    building_data['max_heating_energy_per_floor_area'] = np.inf
    
    
    # Max. Kühllast (max_cooling_energy_per_floor_area)
    ##############################################################################
    # Set to -inf (no further information available)
    building_data['max_cooling_energy_per_floor_area'] = -np.inf
    
    
    # Art der Heizanlage (heating_supply_system)
    ##############################################################################
    building_data['heating_supply_system'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['w_erz_art_et']).astype(str)
    
    building_data['heating_coefficient'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['heating_coefficient'])
        
    
    # Art der Kühlanlage (cooling_supply_system)
    ##############################################################################
    building_data['cooling_supply_system'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['k_erz_art_rk']).astype(str)
    
    
    # Art der Wärmeübergabe (heating_emission_system)
    ##############################################################################
    building_data['heating_emission_system'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['qg13']).astype(str)

    # If there's no heating_supply_system, there can't be any heating
    building_data['heating_emission_system'] = np.where(building_data['heating_supply_system'] == 'NoHeating', 'NoHeating', building_data['heating_emission_system'])

    # Art der Kälteübergabe (cooling_emission_system)
    ##############################################################################
    building_data['cooling_emission_system'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['qi11']).astype(str)
    
    # Art der Warmwassererzeugung (dhw_system) (BRE: qg21)
    ##############################################################################
    building_data['dhw_system'] = building_data['scr_gebaeude_id'].map(be_data_original.set_index('scr_gebaeude_id')['qg21']).astype(str)

    ##############################################################################
    # Delete unnecessary columns
    building_data.drop(['typ_18599', 'Fen_ant', 'geb_f_flaeche_n_iwu', 'geb_f_flaeche_o_iwu', 'geb_f_flaeche_s_iwu', 'geb_f_flaeche_w_iwu', 'building_length_n', 'building_length_s', 'building_length_o', 'building_length_w', 'n_OG', 'qD8', 'k_1', 'k_3', 'tau_D65SNA', 'case_temp_adj_base', 'B_raw', 'B', 'R_raw', 'R', 'case_temp_adj_walls_ug', 'V_min_18599', 'bak_grob', 'ach_min', 'qH1', 'n_50_standard_av', 'standard av-verhältnis', 'facade_area', 'av-verhältnis', 'n_50'], axis = 1, inplace = True)  
    

    #building_data.to_csv("intermittent_results/indata_annualsim/Outdata_Preprocessing_{}.csv".format(building_data["scr_gebaeude_id"].values[0]), index = False, sep = ';') 
    #building_data.to_excel("intermittent_results/indata_annualsim/Outdata_Preprocessing_SA_{}.xlsx".format(building_data["scr_gebaeude_id"].values[0])) 


    return building_data
    
