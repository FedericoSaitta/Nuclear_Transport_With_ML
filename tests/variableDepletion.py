import openmc
import math
import openmc.deplete
import matplotlib.pyplot as plt
import os
import time

def main():
  name = 'Variable_Pin_Depletion/'
  script_dir = os.path.dirname(os.path.abspath(__file__))

  openmc_exec_path = os.path.abspath(os.path.join(script_dir, "../external/openmc/build/bin/openmc"))
  results_dir = os.path.abspath(os.path.join(script_dir, "results/" + name))
  os.makedirs(results_dir, exist_ok=True)

  os.environ['OPENMC_EXEC'] = openmc_exec_path
  openmc.config['cross_sections'] = os.path.join(script_dir, "../data/cross_sections.xml")
  chain_file = os.path.join(script_dir, "../data/simple_chain.xml")

  # Fuel Pin Materials
  fuel = openmc.Material(name="uo2")
  fuel.add_element("U", 1, percent_type="ao", enrichment=4.25)
  fuel.add_element("O", 2)
  fuel.set_density("g/cc", 10.4)
  fuel.depletable = True

  clad = openmc.Material(name="clad")
  clad.add_element("Zr", 1)
  clad.set_density("g/cc", 6)

  water = openmc.Material(name="water")
  water.add_element("O", 1)
  water.add_element("H", 2)
  water.set_density("g/cc", 1.0)
  water.add_s_alpha_beta("c_H_in_H2O")

  # Radii
  radii = [0.42, 0.45]

  # SET VOLUMES
  fuel.volume  = math.pi * radii[0]**2
  clad.volume  = math.pi * (radii[1]**2 - radii[0]**2)
  water.volume = 0.62**2 - math.pi * radii[1]**2

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

  settings = openmc.Settings()
  settings.particles = 1_000
  settings.inactive = 5
  settings.batches = 20

  source = openmc.IndependentSource()
  source.space = openmc.stats.Point((0, 0, 0))
  source.angle = openmc.stats.Isotropic()
  source.energy = openmc.stats.Watt()
  settings.source = source

  settings.temperature = {
    "default": 294.0,
    "method": "interpolation",
  }

  # Export geometry and settings (these don't change)
  geometry.export_to_xml(path=results_dir)
  settings.export_to_xml(path=results_dir)
  os.chdir(results_dir)

  print("Loading depletion chain...")
  chain = openmc.deplete.Chain.from_xml(chain_file)

  power = [10, 50, 100, 200, 400]  # Watts
  time_steps = [30*24*3600]*5  # 5 months each
  fuel_temps = [900, 1000, 1100, 1200, 1300]  # K
  mod_temps  = [580, 590, 600, 610, 620]  # K
  
  for i in range(len(time_steps)):
      print(f"Step {i+1}/{len(time_steps)}: T_fuel={fuel_temps[i]}K, T_mod={mod_temps[i]}K, Power={power[i]}W")
      
      # Update temperatures
      fuel.temperature = fuel_temps[i]
      water.temperature = mod_temps[i]
      materials.export_to_xml(path=results_dir)

      model = openmc.model.Model(geometry, materials, settings)
      
      if i == 0:
          operator = openmc.deplete.CoupledOperator(model, chain)
      else:
          # Subsequent steps: reuse create operator based on prev results
          operator = openmc.deplete.CoupledOperator(
              model, 
              chain,  
              prev_results=openmc.deplete.Results("depletion_results.h5")
          )
      
      # Depletion Integration step
      integrator = openmc.deplete.PredictorIntegrator(
          operator, [time_steps[i]], [power[i]], timestep_units='s'
      )
      integrator.integrate()

  # All results get appended to same depletion file
  results = openmc.deplete.Results("depletion_results.h5")
  time, k = results.get_keff()
  time /= (24 * 60 * 60)

  # Plot k-effective
  plt.figure(figsize=(10, 6))
  plt.errorbar(time, k[:, 0], yerr=k[:, 1], fmt='o-', capsize=5)
  plt.xlabel("Time [d]")
  plt.ylabel(r"$k_{eff} \pm \sigma$")
  plt.title("Reactivity vs Time (Variable Temperature)")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig("k_eff_vs_time.png", dpi=300)
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
  plt.savefig("u235_depletion.png", dpi=300)
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
  plt.savefig("fission_rate.png", dpi=300)
  plt.show()

if __name__ == "__main__":
  start_time = time.perf_counter()
  main()
  end_time = time.perf_counter()
  elapsed = end_time - start_time
  print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")