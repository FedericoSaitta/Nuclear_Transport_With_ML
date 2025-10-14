import openmc
import openmc.deplete
import matplotlib.pyplot as plt
import os
import time
from loguru import logger
import pandas as pd
from datetime import datetime
import numpy as np
import sys

from reactor_sim import create_materials, set_material_volumes, create_geometry, create_settings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../util")))

import plot_helper

## Functions to setup the openMC environment ##
def setup_logging(script_dir):
  log_file = os.path.join(script_dir, 'log.log')
  logger.remove()
  logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

def setup_paths(script_dir, name):
  openmc_exec_path = os.path.abspath(os.path.join(script_dir, "../external/openmc/build/bin/openmc"))
  results_dir = os.path.abspath(os.path.join(script_dir, "results/" + name))
  os.makedirs(results_dir, exist_ok=True)
  
  os.environ['OPENMC_EXEC'] = openmc_exec_path
  openmc.config['cross_sections'] = os.path.join(script_dir, "../data/cross_sections.xml")
  chain_file = os.path.join(script_dir, "../data/simple_chain.xml")
  chain = openmc.deplete.Chain.from_xml(chain_file) # Load chain once
  
  return results_dir, chain


def generate_data(config_dict):
  script_dir = os.path.dirname(os.path.abspath(__file__))

  setup_logging(script_dir)
  results_dir, chain = setup_paths(script_dir, 'Data_Gen/') # Return chain object already opened
  logger.info(f"Using seed: {config_dict['seed']}")
  np.random.seed(config_dict['seed'])

  fuel, clad, water = create_materials(config_dict)

  radii = [0.42, 0.45]
  pitch = 0.62

  # OpenMC objects are passed by reference
  set_material_volumes(fuel, clad, water, radii=radii, pitch=pitch)
  materials = openmc.Materials([fuel, clad, water])
  geometry = create_geometry(materials, radii=radii, pitch=pitch)

  settings = create_settings(config_dict)

  # Export geometry and settings (these don't change)
  geometry.export_to_xml(path=results_dir)
  settings.export_to_xml(path=results_dir)

  os.chdir(results_dir) # Change working directory so OpenMC finds xml files

  fuel_mass_g = config_dict['fuel_density'] * fuel.volume # Fuel density times volume (in g /cm^3)

  n = config_dict['num_depl_steps']
  time_steps = [config_dict['delta_t']] * n # 5 months each
  power = list(np.random.uniform(config_dict['power'][0], config_dict['power'][1], n))# W
  fuel_temps = list(np.random.uniform(config_dict['t_fuel'][0], config_dict['t_fuel'][1], n))# k
  mod_temps = list(np.random.uniform(config_dict['t_mod'][0], config_dict['t_mod'][1], n))# k
  clad_temps = list(np.random.uniform(config_dict['t_clad'][0], config_dict['t_clad'][1], n)) # k
  
  for i in range(n):
    logger.info(f"Step {i+1}/{n}: T_fuel={fuel_temps[i]}K, T_mod={mod_temps[i]}K, Power={power[i]}W")
    
    # Update temperatures for all the materials
    fuel.temperature = fuel_temps[i]
    water.temperature = mod_temps[i]
    clad.temperature = clad_temps[i]
    materials.export_to_xml(path=results_dir)

    model = openmc.model.Model(geometry, materials, settings)
    
    if i == 0: operator = openmc.deplete.CoupledOperator(model, chain)
    else: # For subsequent steps create new operator based on prev results
      operator = openmc.deplete.CoupledOperator(model, chain,  prev_results=openmc.deplete.Results("depletion_results.h5"))
    
    # Depletion Integration step
    integrator = openmc.deplete.PredictorIntegrator(operator, [time_steps[i]], [power[i]*fuel_mass_g], timestep_units='s')
    integrator.integrate()

  # All results get appended to same depletion file
  results = openmc.deplete.Results("depletion_results.h5")

  # Get all nuclides tracked by looking at first result step
  nuclides = results[0].index_nuc.keys() 

  time, k = results.get_keff()
  time /= (24 * 3600) # Get time in seconds

  # Create label with current time and home directory
  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  home_dir = os.path.expanduser("~")
  label = f"run_{current_time}_{home_dir}"

  int_p = list(np.cumsum(np.array(power)))

  data = {'run_label': [label]*len(time), 'time_days': time, 'k_eff': k[:, 0], 'k_eff_std': k[:, 1], 
          'power_W_g': power + ['NaN'], 'int_p_W': int_p + ['NaN'], 'fuel_temp_K': fuel_temps + ['NaN'], 
          'mod_temp_K': mod_temps + ['NaN'], 'clad_temp_K': clad_temps + ['NaN']}

  # Get concentration for each nuclide
  for nuclide in nuclides:
    _, concentration = results.get_atoms("1", nuclide, nuc_units="atom/b-cm")
    data[nuclide] = concentration

  # Create DataFrame and save to CSV
  df = pd.DataFrame(data)
  file_path = os.path.join(script_dir, "nuclide_concentrations.csv")
  file_exists = os.path.isfile(file_path)
  df.to_csv(file_path, mode='a', index=False, header=not file_exists)

  plot_helper.plot_generated_data(nuclides, data, save_folder=script_dir)


if (__name__ == "__main__"):

  import random
  random.seed()  # Seed from system time + process ID
  seed = random.randint(1, 2**31 - 1)

  ### DATA GENERATION CONFIG ###
  ### 'random var': [min, max]
  config_dict = {'seed': 911651453, 'num_depl_steps': 100, 'delta_t': 10*24*3600, # 10 days
                'enrichment': 3.1, # In %
                'fuel_density': 10.4, # In g/cm^3
                'particles': 2_000, 'inactive': 5, 'batches': 20, 'temp_method': 'interpolation',
                
                'power': [0, 40], # In watts / g
                't_fuel': [600, 1200], # In Kelvin
                't_mod': [550, 600], # In Kelvin
                't_clad': [570, 670], # In Kelvin
                }
  import time

  start_time = time.perf_counter()

  try:
    generate_data(config_dict)
  except Exception as e:
    logger.exception(f"Error during data generation: {e}")
  finally:
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    logger.info(f"Data generation completed in {elapsed:.2f} seconds.")