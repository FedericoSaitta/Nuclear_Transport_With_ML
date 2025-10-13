import openmc
import openmc.deplete
import matplotlib.pyplot as plt
import os
import time
from loguru import logger
import pandas as pd
from datetime import datetime
import numpy as np

from reactor_sim import create_materials, set_material_volumes, create_geometry, create_settings

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



def generate_data(seed):
  script_dir = os.path.dirname(os.path.abspath(__file__))

  setup_logging(script_dir)
  results_dir, chain = setup_paths(script_dir, 'Data_Gen/') # Return chain object already opened
  logger.info(f"Using seed: {seed}")

  fuel, clad, water = create_materials(enrichment=4.25, fuel_density=10.4)

  radii = [0.42, 0.45]
  pitch = 0.62

  # OpenMC objects are passed by reference
  set_material_volumes(fuel, clad, water, radii=radii, pitch=pitch)
  materials = openmc.Materials([fuel, clad, water])
  geometry = create_geometry(materials, radii=radii, pitch=pitch)

  settings = create_settings(seed, particles=10, inactive=5, batches=20, temperature=294.0, temp_method='interpolation')

  # Export geometry and settings (these don't change)
  geometry.export_to_xml(path=results_dir)
  settings.export_to_xml(path=results_dir)

  os.chdir(results_dir) # Change working directory so OpenMC finds xml files

  power = [10, 50, 100, 200, 400]  # Watts
  time_steps = [30*24*3600]*5  # 5 months each
  fuel_temps = [900, 1000, 1100, 1200, 1300]  # K
  mod_temps  = [580, 590, 600, 610, 620]  # K
  
  for i in range(len(time_steps)):
    logger.info(f"Step {i+1}/{len(time_steps)}: T_fuel={fuel_temps[i]}K, T_mod={mod_temps[i]}K, Power={power[i]}W")
    
    # Update temperatures
    fuel.temperature = fuel_temps[i]
    water.temperature = mod_temps[i]
    materials.export_to_xml(path=results_dir)

    model = openmc.model.Model(geometry, materials, settings)
    
    if i == 0: operator = openmc.deplete.CoupledOperator(model, chain)
    else: # For subsequent steps create new operator based on prev results
      operator = openmc.deplete.CoupledOperator(model, chain,  prev_results=openmc.deplete.Results("depletion_results.h5"))
    
    # Depletion Integration step
    integrator = openmc.deplete.PredictorIntegrator(operator, [time_steps[i]], [power[i]], timestep_units='s')
    integrator.integrate()

  # All results get appended to same depletion file
  results = openmc.deplete.Results("depletion_results.h5")

  # Get all nuclides tracked by looking at first result step
  nuclides = results[0].index_nuc.keys() 

  time, k = results.get_keff()
  time /= (24 * 60 * 60)

  # Categorize nuclides
  actinides = [nuc for nuc in nuclides if nuc.startswith(('U', 'Pu', 'Np', 'Am', 'Cm'))]
  fission_products = [nuc for nuc in nuclides if nuc not in actinides]


  # Create label with current time and home directory
  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  home_dir = os.path.expanduser("~")
  label = f"run_{current_time}_{home_dir}"

  int_p = list(np.cumsum(np.array(power)))

  data = {'run_label': [label]*len(time), 'time_days': time, 'k_eff': k[:, 0], 'k_eff_std': k[:, 1], 
          'power_W': power + ['NaN'], 'int_p_W': int_p + ['NaN'], 'fuel_temp_K': fuel_temps + ['NaN'], 'mod_temp_K': mod_temps + ['NaN']}

  # Get concentration for each nuclide
  for nuclide in nuclides:
    _, concentration = results.get_atoms("1", nuclide, nuc_units="atom/b-cm")
    data[nuclide] = concentration

  # Create DataFrame and save to CSV
  df = pd.DataFrame(data)
  file_path = os.path.join(script_dir, "nuclide_concentrations.csv")
  file_exists = os.path.isfile(file_path)
  df.to_csv(file_path, mode='a', index=False, header=not file_exists)


  # Plot actinides
  if actinides:
    plt.figure(figsize=(12, 6))
    for nuclide in actinides:
      _, concentration = results.get_atoms("1", nuclide, nuc_units="atom/b-cm")
      plt.plot(time, concentration, marker='o', linewidth=2, label=nuclide)
    
    plt.xlabel("Time [d]")
    plt.ylabel("Number density [atom/b-cm]")
    plt.title("Actinide Number Density Evolution")
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig("actinides_number_density.png", dpi=300)

  # Plot fission products, these are all 0 at t=0
  if fission_products:
    plt.figure(figsize=(12, 6))
    for nuclide in fission_products:
      _, concentration = results.get_atoms("1", nuclide, nuc_units="atom/b-cm")
      plt.plot(time[1:], concentration[1:], marker='o', linewidth=2, label=nuclide)

    plt.xlabel("Time [d]")
    plt.ylabel("Number density [atom/b-cm]")
    plt.title("Fission Product Number Density Evolution")
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig("fission_products_number_density.png", dpi=300)


if (__name__ == "__main__"):

  import random
  random.seed()  # Seed from system time + process ID
  seed = random.randint(1, 2**31 - 1)
  seed = None # Complete repdroducibility for now

  generate_data(seed)