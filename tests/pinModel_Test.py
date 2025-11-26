import openmc
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../util")))

import plot_helper

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
  settings.inactive = 0
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
  tally = openmc.Tally(name='fuel_tally')
  tally.filters = [cell_filter]
  tally.nuclides = ['U235']
  tally.scores = ['total', 'fission', 'absorption', '(n,gamma)']

  tallies = openmc.Tallies([tally])


  materials.export_to_xml(path=results_dir)
  geometry.export_to_xml(path=results_dir)
  settings.export_to_xml(path=results_dir)
  tallies.export_to_xml(path=results_dir)


  print(">>> Current working directory:", os.getcwd())
  print(">>> Intended OpenMC working directory:", results_dir)

  # --- Run OpenMC ---
  openmc.run(
    cwd=results_dir,
    
  )

  sp_path = '/home/t97807fs/Nuclear_Transport_With_ML/test/results/Pin_Model_Test/statepoint.100.h5'

  sp = openmc.StatePoint(sp_path)

  entropy = sp.entropy  # list of entropy values per active batch
  batches = range(1, len(entropy)+1)  # x-axis: batch number

  plot_helper.Shannon_Entropy(batches, entropy, save_path=os.path.join(results_dir, "shannon_entropy.png"))
  
  # k-effective per batch
  kvals = [k for k in sp.k_generation]
  plot_helper.K_Effective(batches, kvals, save_path=os.path.join(results_dir, "keff_vs_batch.png"))

if __name__ == "__main__":
  main()
