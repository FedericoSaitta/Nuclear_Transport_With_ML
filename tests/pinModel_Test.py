import openmc
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../util")))

import plot_helper

def plot_neutron_spectrum(sp, results_dir):
  """Plot the neutron energy spectrum"""
  # Get the spectrum tally
  spectrum_tally = sp.get_tally(name='neutron_spectrum')
  
  # Extract flux values and energy bins
  flux = spectrum_tally.mean.flatten()
  
  # Get the energy filter and its bins correctly
  energy_filter = spectrum_tally.filters[1]
  # The bins attribute gives you the bin edges as a 1D array
  energy_bins = energy_filter.values  # Use .values instead of .bins
  
  # Calculate bin centers for plotting
  energy_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])  # Geometric mean
  
  # Calculate lethargy width for per-lethargy flux
  lethargy_width = np.log(energy_bins[1:] / energy_bins[:-1])
  flux_per_lethargy = flux / lethargy_width
  
  # Create figure with two subplots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
  
  # Plot 1: Standard flux vs energy
  ax1.loglog(energy_centers, flux, 'b-', linewidth=2)
  ax1.set_xlabel('Energy (eV)', fontsize=12)
  ax1.set_ylabel('Flux (n/cm²-s)', fontsize=12)
  ax1.set_title('Neutron Energy Spectrum', fontsize=14, fontweight='bold')
  ax1.grid(True, alpha=0.3, which='both')
  
  # Add typical energy region labels
  ax1.axvspan(0, 0.625, alpha=0.2, color='blue', label='Thermal (<0.625 eV)')
  ax1.axvspan(0.625, 1e5, alpha=0.2, color='green', label='Epithermal (0.625 eV - 100 keV)')
  ax1.axvspan(1e5, 1e7, alpha=0.2, color='red', label='Fast (>100 keV)')
  ax1.legend(loc='best', fontsize=10)
    
  ax2.semilogx(energy_centers, flux_per_lethargy, 'r-', linewidth=2)
  ax2.set_xlabel('Energy (eV)', fontsize=15)
  ax2.set_ylabel('Flux per unit lethargy', fontsize=15)
  ax2.set_title('Neutron Spectrum (per unit lethargy)', fontsize=18, fontweight='bold')

  # <<< Increase tick number sizes >>>
  ax2.tick_params(axis='both', which='major', labelsize=14)
  ax2.tick_params(axis='both', which='minor', labelsize=12)

  ax2.grid(True, alpha=0.3)
  from matplotlib.ticker import LogLocator

  ax2.set_xscale('log')
  ax2.xaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
  ax2.axvspan(0, 0.625, alpha=0.2, color='blue', label='Thermal (<0.625 eV)')
  ax2.axvspan(0.625, 1e5, alpha=0.2, color='green', label='Epithermal (0.625 eV - 100 keV)')
  ax2.axvspan(1e5, 1e7, alpha=0.2, color='red', label='Fast (>100 keV)')
  ax2.legend(loc='best', fontsize=14)

  plt.tight_layout()
  plt.savefig(os.path.join(results_dir, 'neutron_spectrum.png'), dpi=300, bbox_inches='tight')
  plt.close()
  
  print(f"Neutron spectrum saved to {os.path.join(results_dir, 'neutron_spectrum.png')}")
  
  # Print some statistics
  thermal_idx = energy_centers < 0.625
  epithermal_idx = (energy_centers >= 0.625) & (energy_centers < 1e5)
  fast_idx = energy_centers >= 1e5
  
  thermal_fraction = flux[thermal_idx].sum() / flux.sum()
  epithermal_fraction = flux[epithermal_idx].sum() / flux.sum()
  fast_fraction = flux[fast_idx].sum() / flux.sum()
  
  print("\n=== Neutron Spectrum Statistics ===")
  print(f"Thermal fraction (<0.625 eV): {thermal_fraction*100:.2f}%")
  print(f"Epithermal fraction (0.625 eV - 100 keV): {epithermal_fraction*100:.2f}%")
  print(f"Fast fraction (>100 keV): {fast_fraction*100:.2f}%")
  print(f"Average energy: {np.average(energy_centers, weights=flux):.2f} eV")


def main():
  name = 'Pin_Model_Test'
  # Path for the openmc executable
  script_dir = os.path.dirname(os.path.abspath(__file__))

  results_dir = os.path.abspath(os.path.join(script_dir, "results/" + name))
  openmc.config['cross_sections'] = os.path.join(script_dir, "../data/cross_sections.xml")

  os.makedirs(results_dir, exist_ok=True)

  # For pure elements use add_element, for composites use add_nuclide

  # Fuel material, 3% Enriched UO2
  uo2 = openmc.Material(name="uo2")
  uo2.add_nuclide('U235', 0.03)
  uo2.add_nuclide('U238', 0.97)
  uo2.add_nuclide('O16', 2.0)
  uo2.set_density('g/cm3', 10.0)

  # Cladding Material
  zirconium = openmc.Material(name="zirconium")
  zirconium.add_element('Zr', 1.0)
  zirconium.set_density('g/cm3', 6.6)

  # Moderator
  water = openmc.Material(name="h2o")
  water.add_nuclide('H1', 2.0)
  water.add_nuclide('O16', 1.0)
  water.set_density('g/cm3', 1.0)
  water.add_s_alpha_beta('c_H_in_H2O')


  print(f"Water thermal scattering: {water.get_nuclides()}") 
  # add_s_alpha_beta: adds scattering data to the moderator, hydrogen in bound molecules behaves very 
  # differently from free hydrogen, WITHOUT THIS NEUTRNOS WILL NOT MODERATE PROPERLY

  # Export materials
  materials = openmc.Materials([uo2, zirconium, water])

  # --- Fuel pin geometry ---
  fuel_outer_radius = openmc.ZCylinder(r=0.39)
  clad_inner_radius = openmc.ZCylinder(r=0.40)
  clad_outer_radius = openmc.ZCylinder(r=0.46)

  fuel_region = -fuel_outer_radius
  gap_region = +fuel_outer_radius & -clad_inner_radius
  clad_region = +clad_inner_radius & -clad_outer_radius

  fuel = openmc.Cell(name='fuel', fill=uo2, region=fuel_region)
  gap = openmc.Cell(name='air gap', region=gap_region)
  clad = openmc.Cell(name='clad', fill=zirconium, region=clad_region)

  # Moderator boundary (rectangular prism with reflective boundaries)
  # Pitch = center-to-center distance between adjacent fuel pins in a lattice (units usually in cm).
  # The pitch controls how much moderator (water) surrounds the fuel

  pitch = 1.26
  left   = openmc.XPlane(x0=-pitch/2, boundary_type='reflective')
  right  = openmc.XPlane(x0= pitch/2, boundary_type='reflective')
  bottom = openmc.YPlane(y0=-pitch/2, boundary_type='reflective')
  top    = openmc.YPlane(y0= pitch/2, boundary_type='reflective')

  prism_region = +left & -right & +bottom & -top
  moderator_region = prism_region & +clad_outer_radius

  moderator = openmc.Cell(name='moderator', fill=water, region=moderator_region)

  # Root universe and geometry
  root_universe = openmc.Universe(cells=[fuel, gap, clad, moderator])
  geometry = openmc.Geometry(root_universe)

  # --- Settings ---
  point = openmc.stats.Point((0, 0, 0))
  source = openmc.Source(space=point)

  settings = openmc.Settings(path=results_dir)
  settings.source = source
  settings.batches = 100
  settings.inactive = 50
  settings.particles = 10_000

  m = openmc.RegularMesh()
  m.dimension = [50, 50, 1]  # finer mesh improves entropy smoothness
  ll, ur = geometry.bounding_box

  print(ll, ur)
  m.lower_left = (float(ll[0])-0.0001, float(ll[1])-0.0001, -100000.0)
  m.upper_right = (float(ur[0])+0.0001, float(ur[1])+0.0001,  100000.0)

  settings.entropy_mesh = m

  # --- Tallies ---
  cell_filter = openmc.CellFilter(fuel)

  # Create energy bins for spectrum (logarithmic spacing is typical)
  # From thermal (0.001 eV) to fast (20 MeV)
  energy_bins = np.logspace(-3, 7, 200)  # 200 bins from 0.001 eV to 10 MeV
  energy_filter = openmc.EnergyFilter(energy_bins)

  # Flux tally with energy filter for spectrum
  spectrum_tally = openmc.Tally(name='neutron_spectrum')
  spectrum_tally.filters = [cell_filter, energy_filter]
  spectrum_tally.scores = ['flux']

  # Your original tally
  tally = openmc.Tally(name='fuel_tally')
  tally.filters = [cell_filter]
  tally.nuclides = ['U235']
  tally.scores = ['total', 'fission', 'absorption', '(n,gamma)']

  tallies = openmc.Tallies([tally, spectrum_tally])

  materials.export_to_xml(path=results_dir)
  geometry.export_to_xml(path=results_dir)
  settings.export_to_xml(path=results_dir)
  tallies.export_to_xml(path=results_dir)


  print(">>> Current working directory:", os.getcwd())
  print(">>> Intended OpenMC working directory:", results_dir)

  # --- Run OpenMC ---
  openmc.run(
    cwd=results_dir,
    output=True  # This will show more detailed output
)

  # With this:
  sp_path = os.path.join(results_dir, 'statepoint.100.h5')
  sp = openmc.StatePoint(sp_path)

  entropy = sp.entropy  # list of entropy values per active batch
  batches = range(1, len(entropy)+1)  # x-axis: batch number

  plot_helper.Shannon_Entropy(batches, entropy, save_path=os.path.join(results_dir, "shannon_entropy.png"))
  
  # k-effective per batch
  kvals = [k for k in sp.k_generation]
  plot_helper.K_Effective(batches, kvals, save_path=os.path.join(results_dir, "keff_vs_batch.png"))

  # Plot neutron spectrum
  plot_neutron_spectrum(sp, results_dir)
  
if __name__ == "__main__":
  main()
