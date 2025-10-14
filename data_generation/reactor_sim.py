# This script sets up the openMC model used for data generation
import openmc
import math

def create_materials(config_dict):

  enrichment = config_dict['enrichment']
  fuel_density = config_dict['fuel_density'] # in g/cm^3

  fuel = openmc.Material(name="uo2")
  fuel.add_element("U", 1, percent_type="ao", enrichment=enrichment)
  fuel.add_element("O", 2)
  fuel.set_density("g/cc", fuel_density)
  fuel.depletable = True

  clad = openmc.Material(name="clad")
  clad.add_element("Zr", 1)
  clad.set_density("g/cc", 6)

  water = openmc.Material(name="water")
  water.add_element("O", 1)
  water.add_element("H", 2)
  water.set_density("g/cc", 1.0)
  water.add_s_alpha_beta("c_H_in_H2O")
  
  return fuel, clad, water

# Note these are called volumes but as this is a 2D problem they are effectively areas
def set_material_volumes(fuel, clad, water, radii=[0.42, 0.45], pitch=0.62):
  fuel.volume  = math.pi * radii[0]**2
  clad.volume  = math.pi * (radii[1]**2 - radii[0]**2)
  water.volume = pitch**2 - math.pi * radii[1]**2


def create_geometry(materials, radii=[0.42, 0.45], pitch=0.62):
  pin_surfaces = [openmc.ZCylinder(r=r) for r in radii]
  pin_univ = openmc.model.pin(pin_surfaces, materials)

  half_width = pitch / 2
  left   = openmc.XPlane(x0=-half_width, boundary_type='reflective')
  right  = openmc.XPlane(x0=half_width, boundary_type='reflective')
  bottom = openmc.YPlane(y0=-half_width, boundary_type='reflective')
  top    = openmc.YPlane(y0=half_width, boundary_type='reflective')

  bound_box = +left & -right & +bottom & -top
  root_cell = openmc.Cell(fill=pin_univ, region=bound_box)
  root_univ = openmc.Universe(cells=[root_cell])
  geometry = openmc.Geometry(root_univ)
  
  return geometry


def create_settings(config_dict): 
  seed = config_dict['seed']
  particles = config_dict['particles']
  
  inactive = config_dict['inactive']
  batches = config_dict['batches']
  temp_method = config_dict['temp_method']

  settings = openmc.Settings()
  settings.particles = particles
  settings.inactive = inactive
  settings.batches = batches

  source = openmc.IndependentSource()
  source.space = openmc.stats.Point((0, 0, 0))
  source.angle = openmc.stats.Isotropic()
  source.energy = openmc.stats.Watt()
  settings.source = source

  settings.temperature = {"method": temp_method}

  if seed is not None:
    settings.seed = seed
  
  return settings