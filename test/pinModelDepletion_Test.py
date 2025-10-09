import openmc
import math
import openmc.deplete
import matplotlib.pyplot as plt
import os
import time

def main():
  name = 'Pin_Model_Depletion_Test/'
  script_dir = os.path.dirname(os.path.abspath(__file__))

  openmc_exec_path = os.path.abspath(os.path.join(script_dir, "../external/openmc/build/bin/openmc"))
  results_dir = os.path.abspath(os.path.join(script_dir, "results/" + name))
  os.makedirs(results_dir, exist_ok=True)

  # Path for the openmc executable and configure it
  os.environ['OPENMC_EXEC'] = openmc_exec_path
  openmc.config['cross_sections'] = os.path.join(script_dir, "../data/cross_sections.xml")
  chain_file = os.path.join(script_dir, "../data/chain_casl_pwr.xml")

  # Create materials
  fuel = openmc.Material(name="uo2")
  fuel.add_element("U", 1, percent_type="ao", enrichment=4.25)
  fuel.add_element("O", 2)
  fuel.set_density("g/cc", 10.4)
  fuel.depletable = True  # Mark as depletable

  clad = openmc.Material(name="clad")
  clad.add_element("Zr", 1)
  clad.set_density("g/cc", 6)

  water = openmc.Material(name="water")
  water.add_element("O", 1)
  water.add_element("H", 2)
  water.set_density("g/cc", 1.0)
  water.add_s_alpha_beta("c_H_in_H2O")

  # Define radii first (needed for volume calculation)
  radii = [0.42, 0.45]

  # SET VOLUMES BEFORE CREATING MATERIALS COLLECTION
  fuel.volume  = math.pi * radii[0]**2                    # fuel pin
  clad.volume  = math.pi * (radii[1]**2 - radii[0]**2)   # cladding
  water.volume = 0.62**2 - math.pi * radii[1]**2

  # Now create materials collection
  materials = openmc.Materials([fuel, clad, water])

  # Create geometry
  pin_surfaces = [openmc.ZCylinder(r=r) for r in radii]
  pin_univ = openmc.model.pin(pin_surfaces, materials)

  half_width = 0.62 / 2
  left   = openmc.XPlane(x0=-half_width, boundary_type='reflective')
  right  = openmc.XPlane(x0=half_width, boundary_type='reflective')
  bottom = openmc.YPlane(y0=-half_width, boundary_type='reflective')
  top    = openmc.YPlane(y0=half_width, boundary_type='reflective')

  bound_box = +left & -right & +bottom & -top
  root_cell = openmc.Cell(fill=pin_univ, region=bound_box)
  root_univ = openmc.Universe(cells=[root_cell])
  geometry = openmc.Geometry(root_univ)

  # Settings - INCREASED particles for better statistics
  settings = openmc.Settings()
  settings.particles = 1_000
  settings.inactive = 10
  settings.batches = 50

  # Add source
  source = openmc.IndependentSource()
  source.space = openmc.stats.Point((0, 0, 0))
  source.angle = openmc.stats.Isotropic()
  source.energy = openmc.stats.Watt()
  settings.source = source

  materials.export_to_xml(path=results_dir)
  geometry.export_to_xml(path=results_dir)
  settings.export_to_xml(path=results_dir)

  # Create model
  model = openmc.model.Model(geometry, materials, settings)

  # Set up depletion
  print("Starting depletion calculation...")
  chain_dict = openmc.deplete.Chain.from_xml(chain_file)
  print(chain_dict)

  # Create operator - FIXED: Changed to "source-rate" normalization
  os.chdir(results_dir)
  operator = openmc.deplete.CoupledOperator(model, chain_file)

  # Adjust power and time steps
  power = 174  # Watts (this is now interpreted as source rate)
  time_steps = [30*24*3600]*36 # 6 months in seconds

  # Use PredictorIntegrator
  integrator = openmc.deplete.PredictorIntegrator(
      operator, 
      time_steps, 
      power,
      timestep_units='s', 
  )

  integrator.integrate()

  print("Depletion calculation complete!")

  # Load and plot results
  results_file = os.path.join(results_dir, "depletion_results.h5")
  results = openmc.deplete.ResultsList.from_hdf5(results_file)
  time, k = results.get_eigenvalue()
  time /= (24 * 60 * 60)  # Convert to days
  print("k-array:", k)

  # Plot k-effective
  plt.figure(figsize=(10, 6))
  plt.errorbar(time, k[:, 0], yerr=k[:, 1], fmt='o-', capsize=5)
  plt.xlabel("Time [d]")
  plt.ylabel(r"$k_{eff} \pm \sigma$")
  plt.title("Reactivity vs Time")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(results_dir, "k_eff_vs_time.png"), dpi=300)
  plt.show()

  # Plot U-235 depletion
  _time, u5 = results.get_atoms("1", "U235")
  plt.figure(figsize=(10, 6))
  plt.plot(time, u5, label="U-235", marker='o', linewidth=2)
  plt.xlabel("Time [d]")
  plt.ylabel("Number of atoms")
  plt.title("U-235 Depletion")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(results_dir, "u235_depletion.png"), dpi=300)
  plt.show()

  # Plot Xe-135 buildup
  _time, xe135 = results.get_atoms("1", "Xe135")
  plt.figure(figsize=(10, 6))
  plt.plot(time, xe135, label="Xe-135", marker='o', color='red', linewidth=2)
  plt.xlabel("Time [d]")
  plt.ylabel("Number of atoms")
  plt.title("Xe-135 Buildup")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(results_dir, "xe135_buildup.png"), dpi=300)
  plt.show()

  # Plot U-235 fission rate
  _time, u5_fission = results.get_reaction_rate("1", "U235", "fission")
  plt.figure(figsize=(10, 6))
  plt.plot(time, u5_fission, marker='o', color='green', linewidth=2)
  plt.xlabel("Time [d]")
  plt.ylabel("Fission rate [reactions/s]")
  plt.title("U-235 Fission Rate")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(results_dir, "fission_rate.png"), dpi=300)
  plt.show()

# This is very important, without this, multiple imports will be run 
# from the python multiprocessing module
if __name__ == "__main__":
  start_time = time.perf_counter()  # Start timing
  main()
  end_time = time.perf_counter()    # End timing
  elapsed = end_time - start_time
  print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")