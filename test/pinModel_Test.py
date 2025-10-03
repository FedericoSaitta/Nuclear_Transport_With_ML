import openmc
import os

def main():
  # Path for the openmc executable
  openmc_exec_path = "../openmc/build/bin/openmc"
  script_dir = os.path.dirname(os.path.abspath(__file__))
  openmc.config['cross_sections'] = os.path.join(script_dir, "../data/cross_sections.xml")

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
  materials.export_to_xml()

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
  geometry.export_to_xml()

  # --- Settings ---
  point = openmc.stats.Point((0, 0, 0))
  source = openmc.Source(space=point)

  settings = openmc.Settings()
  settings.source = source
  settings.batches = 100
  settings.inactive = 10
  settings.particles = 1000
  settings.export_to_xml()

  # --- Tallies ---
  cell_filter = openmc.CellFilter(fuel)
  tally = openmc.Tally(name='fuel_tally')
  tally.filters = [cell_filter]
  tally.nuclides = ['U235']
  tally.scores = ['total', 'fission', 'absorption', '(n,gamma)']

  tallies = openmc.Tallies([tally])
  tallies.export_to_xml()

  # --- Run OpenMC ---
  openmc.run(
    openmc_exec=openmc_exec_path,
  )

if __name__ == "__main__":
  main()
