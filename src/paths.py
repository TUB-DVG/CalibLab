import os
from pathlib import Path

# BASE PATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = str(Path(ROOT_DIR).parents[0])
#DIBS_SIM_DIR = os.path.join(ROOT_DIR, 'DIBS_sim_modified/iso_simulator/annualSimulation')
DIBS_DATAPRE_DIR = str(os.path.join(PROJECT_DIR, 'src','DIBS','data_preprosessing'))
#DIBS_DATAPRE_DIR = str(PROJECT_DIR+'/src/DIBS_sim_modified/data_preprosessing')

DIBS_DATAPRE_DIR = DIBS_DATAPRE_DIR.replace('\\','/')

RES_DIR = os.path.join(PROJECT_DIR, 'results')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
IMAGE_DIR = os.path.join(PROJECT_DIR, 'images')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')
SAMP_DIR = os.path.join(PROJECT_DIR, 'samples')
CTRL_DIR = os.path.join(PROJECT_DIR, 'control')


DIBS_DIR =  os.path.join(ROOT_DIR, 'DIBS')
AUX_DIR = os.path.join(ROOT_DIR, 'DIBS/iso_simulator/auxiliary')


