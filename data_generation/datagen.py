import os
os.environ["OMP_NUM_THREADS"] = "2"

HOUR_IN_SECONDS = 3600
DAY_IN_SECONDS = 24 * HOUR_IN_SECONDS

import openmc
import openmc.deplete
import time
from loguru import logger
from datetime import datetime
import numpy as np
import pandas as pd
import sys
import random
import multiprocessing as mp

from reactor_sim import create_materials, set_material_volumes, create_geometry, create_settings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../util")))
import plot_helper


def setup_logging(script_dir):
  log_file = os.path.join(script_dir, 'log.log')
  logger.remove()
  logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")


def setup_paths(script_dir, worker_id):
  openmc_exec_path = os.path.abspath(os.path.join(script_dir, "../external/openmc/build/bin/openmc"))
  results_dir = os.path.abspath(os.path.join(script_dir, "results", f"worker_{worker_id}"))
  os.makedirs(results_dir, exist_ok=True)
  
  openmc.config['cross_sections'] = os.path.join(script_dir, "../data/cross_sections.xml")
  os.environ['OPENMC_EXEC'] = openmc_exec_path
  os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
  os.environ['OPENMC_CROSS_SECTIONS'] = str(openmc.config['cross_sections'])
  
  chain_file = os.path.join(script_dir, "../data/chain_casl_pwr.xml")
  chain = openmc.deplete.Chain.from_xml(chain_file)
  
  return results_dir, chain


def setup_reactor_model(config, results_dir):
  fuel, clad, water = create_materials(config)
  set_material_volumes(fuel, clad, water, radii=config['geometry_radii'], pitch=config['geometry_pitch'])
  
  materials = openmc.Materials([fuel, clad, water])
  geometry = create_geometry(materials, radii=config['geometry_radii'], pitch=config['geometry_pitch'])
  settings = create_settings(config)
  settings.verbosity = 1
  settings.output = {'tallies': False}
  
  geometry.export_to_xml(path=results_dir)
  settings.export_to_xml(path=results_dir)
  
  return fuel, clad, water, materials, geometry, settings


def generate_random_conditions(config):
  num_steps = config['num_depl_steps']
  return {
    'time_steps': [config['delta_t']] * num_steps,
    'power': list(np.random.uniform(config['power'][0], config['power'][1], num_steps)),
    'fuel_temps': list(np.random.uniform(config['t_fuel'][0], config['t_fuel'][1], num_steps)),
    'mod_temps': list(np.random.uniform(config['t_mod'][0], config['t_mod'][1], num_steps)),
    'clad_temps': list(np.random.uniform(config['t_clad'][0], config['t_clad'][1], num_steps)),
    'mod_densities': list(np.random.uniform(config['rho_mod'][0], config['rho_mod'][1], num_steps)),
    'boron_ppm': list(np.random.uniform(config['boron_ppm'][0], config['boron_ppm'][1], num_steps)) 
  }


def run_depletion_step(model, chain, time_step, power_watts, prev_results_file=None):
    if prev_results_file and os.path.exists(prev_results_file):
        prev_results = openmc.deplete.Results(prev_results_file)
        operator = openmc.deplete.CoupledOperator(model, chain, prev_results=prev_results)
    else:
        operator = openmc.deplete.CoupledOperator(model, chain)
    
    integrator = openmc.deplete.PredictorIntegrator(
        operator, [time_step], [power_watts], timestep_units='s'
    )
    integrator.integrate()


def run_depletion_simulation(fuel, clad, water, materials, geometry, settings, chain, 
                             conditions, fuel_mass_g, worker_id, results_dir):
  num_steps = len(conditions['time_steps'])
  
  for i in range(num_steps):
    logger.info(f"Worker {worker_id}** Step {i+1}/{num_steps}")
    
    fuel.temperature = conditions['fuel_temps'][i]
    water.temperature = conditions['mod_temps'][i]
    clad.temperature = conditions['clad_temps'][i]
    water.set_density('g/cm3', conditions['mod_densities'][i])
    # Remove old boron and add new
    water.remove_element('B')
    water.add_element('B', conditions['boron_ppm'][i] * 1e-6, 'wo')

    materials.export_to_xml(path=results_dir)
    
    model = openmc.model.Model(geometry, materials, settings)
    power_watts = conditions['power'][i] * fuel_mass_g
    prev_results_file = "depletion_results.h5" if i > 0 else None
    
    run_depletion_step(model, chain, conditions['time_steps'][i], power_watts, prev_results_file)


def extract_results_data(results, conditions):
  time, k = results.get_keff()
  time /= DAY_IN_SECONDS

  burnup = []
  for i, res in enumerate(results):
    if i == 0: burnup.append(0.0)
    else:
      bu = results.get_burnup(i, units='MWd/kg') # Burnup in MWd/kgU
      burnup.append(bu)
  
  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  home_dir = os.path.expanduser("~")
  label = f"run_{current_time}_{home_dir}"
  integrated_power = list(np.cumsum(np.array(conditions['power'])))
  
  data = {
    'run_label': [label] * len(time),
    'time_days': time,
    'k_eff': k[:, 0],
    'k_eff_std': k[:, 1],
    'power_W_g': conditions['power'] + ['NaN'],
    'int_p_W': integrated_power + ['NaN'],
    'burnup_MWd_kg': burnup,
    'fuel_temp_K': conditions['fuel_temps'] + ['NaN'],
    'mod_temp_K': conditions['mod_temps'] + ['NaN'],
    'clad_temp_K': conditions['clad_temps'] + ['NaN'], 
    'mod_density_g_cm3': conditions['mod_densities'] + ['NaN'],
    'boron_ppm': conditions['boron_ppm'] + ['NaN']
  }
  
  nuclides = results[0].index_nuc.keys()
  for nuclide in nuclides:
    _, concentration = results.get_atoms("1", nuclide, nuc_units="atom/b-cm")
    data[nuclide] = concentration
  
  return data, nuclides


def save_results(data, script_dir, worker_id):
  os.makedirs(os.path.join(script_dir, "data"), exist_ok=True)
  df = pd.DataFrame(data)
  file_path = os.path.join(script_dir, "data", f"worker_{worker_id}_nuclide_concentrations.csv")
  file_exists = os.path.isfile(file_path)
  df.to_csv(file_path, mode='a', index=False, header=not file_exists)


def generate_data(config):
  worker_id = config['worker_id']
  script_dir = os.path.dirname(os.path.abspath(__file__))
  
  setup_logging(script_dir)
  results_dir, chain = setup_paths(script_dir, worker_id)
  
  logger.info(f"Worker {worker_id}** Using seed: {config['seed']}")
  np.random.seed(config['seed'])
  
  fuel, clad, water, materials, geometry, settings = setup_reactor_model(config, results_dir)
  os.chdir(results_dir)
  
  fuel_mass_g = config['fuel_density'] * fuel.volume
  conditions = generate_random_conditions(config)
  
  run_depletion_simulation(fuel, clad, water, materials, geometry, settings, chain,
                          conditions, fuel_mass_g, worker_id, results_dir)
  
  results = openmc.deplete.Results("depletion_results.h5")
  data, nuclides = extract_results_data(results, conditions)
  save_results(data, script_dir, worker_id)

  # Ensure to disable plotting for runs with huge number of tracked isotopes as these will overflow diagram
  # plot_helper.plot_generated_data(nuclides, data, save_folder=script_dir, worker_id=worker_id)


def create_worker_configs(base_config, num_workers):
  random.seed()
  configs = []
  for worker_id in range(1, num_workers + 1):
    config = base_config.copy()
    config['worker_id'] = worker_id
    config['seed'] = random.randint(1, 2**31 - 1)
    configs.append(config)
  return configs


def run_parallel_simulations(configs):
  processes = []
  
  try:
    for config in configs:
      process = mp.Process(target=generate_data, args=(config,))
      process.start()
      processes.append(process)
    
    for process in processes:
      process.join()
        
  except Exception as e:
    print(f"Error during parallel data generation: {e}")
    for process in processes:
      if process.is_alive():
        process.terminate()
    raise


if __name__ == "__main__":
  base_config = {
    'num_depl_steps': 10,
    'delta_t': 10 * DAY_IN_SECONDS,
    'enrichment': 3.1,
    'fuel_density': 10.4,
    'particles': 5_000,
    'inactive': 10,
    'batches': 50,
    'temp_method': 'interpolation',
    'power': [0, 40],
    't_fuel': [600, 1200],
    't_mod': [550, 600],
    't_clad': [570, 670],
    'rho_mod': [0.74, 1.00],
    'boron_ppm': [0, 1000],
    'geometry_radii': [0.39218, 0.45720],
    'geometry_pitch': 1.25984,            
  }
  
  DATA_GEN_RUNS = 1 # Number of time to repeat the program
  NUM_WORKERS = 20   # Number of parallel program instances

  for i in range(DATA_GEN_RUNS):
    configs = create_worker_configs(base_config, NUM_WORKERS)
    
    start_time = time.perf_counter()
    run_parallel_simulations(configs)
    elapsed = time.perf_counter() - start_time
    
    print(f"RUN: {i+1}/{DATA_GEN_RUNS}, All workers completed in {elapsed:.2f} seconds.")
