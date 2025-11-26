import openmc
import openmc.deplete
import os
import tempfile
import pytest

def test_openmc_import():
  """Test that OpenMC can be imported"""
  assert openmc.__version__ is not None
  print("OpenMC version:", openmc.__version__)

def test_material_creation():
  """Test that we can create an OpenMC material"""
  material = openmc.Material(name="Empty")
  assert material is not None
  assert material.name == "Empty"

def test_geometry_creation():
  """Test that we can create an OpenMC geometry"""
  geometry = openmc.Geometry()
  cell = openmc.Cell(name="Empty cell")
  geometry.root = cell
  
  assert geometry is not None
  assert cell.name == "Empty cell"
  assert geometry.root == cell

def test_deplete_module():
  """Test that OpenMC depletion module is available"""
  assert hasattr(openmc, 'deplete')
  assert openmc.deplete is not None

def test_pin_model_with_simulation():
  """Test complete pin model with minimal cross sections"""
  
  with tempfile.TemporaryDirectory() as results_dir:
      
      # Use OpenMC's test data (comes with conda install)
      # This only has a few isotopes but enough for basic testing
      try:
          openmc.config['cross_sections'] = os.path.join(
              os.environ.get('CONDA_PREFIX', ''),
              'share/openmc/data/endfb-viii.0-hdf5/cross_sections.xml'
          )
      except:
          pytest.skip("Test cross sections not available")
      
      # --- Materials ---
      uo2 = openmc.Material(name="uo2")
      uo2.add_nuclide('U235', 0.03)
      uo2.add_nuclide('U238', 0.97)
      uo2.add_nuclide('O16', 2.0)
      uo2.set_density('g/cm3', 10.0)
      
      zirconium = openmc.Material(name="zirconium")
      zirconium.add_element('Zr', 1.0)
      zirconium.set_density('g/cm3', 6.6)
      
      water = openmc.Material(name="h2o")
      water.add_nuclide('H1', 2.0)
      water.add_nuclide('O16', 1.0)
      water.set_density('g/cm3', 1.0)
      water.add_s_alpha_beta('c_H_in_H2O')
      
      materials = openmc.Materials([uo2, zirconium, water])
      
      # --- Geometry ---
      fuel_outer_radius = openmc.ZCylinder(r=0.39)
      clad_inner_radius = openmc.ZCylinder(r=0.40)
      clad_outer_radius = openmc.ZCylinder(r=0.46)
      
      fuel_region = -fuel_outer_radius
      gap_region = +fuel_outer_radius & -clad_inner_radius
      clad_region = +clad_inner_radius & -clad_outer_radius
      
      fuel = openmc.Cell(name='fuel', fill=uo2, region=fuel_region)
      gap = openmc.Cell(name='air gap', region=gap_region)
      clad = openmc.Cell(name='clad', fill=zirconium, region=clad_region)
      
      pitch = 1.26
      left = openmc.XPlane(x0=-pitch/2, boundary_type='reflective')
      right = openmc.XPlane(x0=pitch/2, boundary_type='reflective')
      bottom = openmc.YPlane(y0=-pitch/2, boundary_type='reflective')
      top = openmc.YPlane(y0=pitch/2, boundary_type='reflective')
      
      prism_region = +left & -right & +bottom & -top
      moderator_region = prism_region & +clad_outer_radius
      
      moderator = openmc.Cell(name='moderator', fill=water, region=moderator_region)
      
      root_universe = openmc.Universe(cells=[fuel, gap, clad, moderator])
      geometry = openmc.Geometry(root_universe)
      
      # --- Settings (minimal for fast CI) ---
      point = openmc.stats.Point((0, 0, 0))
      source = openmc.Source(space=point)
      
      settings = openmc.Settings()
      settings.source = source
      settings.batches = 10       # Minimal batches for speed
      settings.inactive = 2       # Quick warmup
      settings.particles = 100    # Very few particles
      
      # --- Tallies ---
      cell_filter = openmc.CellFilter(fuel)
      tally = openmc.Tally(name='fuel_tally')
      tally.filters = [cell_filter]
      tally.nuclides = ['U235']
      tally.scores = ['fission']
      
      tallies = openmc.Tallies([tally])
      
      # --- Export XML files ---
      materials.export_to_xml(path=results_dir)
      geometry.export_to_xml(path=results_dir)
      settings.export_to_xml(path=results_dir)
      tallies.export_to_xml(path=results_dir)
      
      # --- RUN IT ---
      print("Running OpenMC simulation...")
      openmc.run(cwd=results_dir, output=False)  # output=False for cleaner logs
      
      # --- Verify it worked ---
      statepoint = os.path.join(results_dir, 'statepoint.10.h5')
      assert os.path.exists(statepoint), "Statepoint file not created"
      
      # Load results
      sp = openmc.StatePoint(statepoint)
      k_eff = sp.keff
      
      # Basic sanity checks
      assert k_eff.n > 0.5, f"k_eff too low: {k_eff.n}"
      assert k_eff.n < 2.0, f"k_eff too high: {k_eff.n}"
      
      # Check tally results
      tally_result = sp.get_tally(name='fuel_tally')
      fission_rate = tally_result.mean.flatten()[0]
      assert fission_rate > 0, "No fissions detected"
      
      print(f"✓ Simulation completed successfully!")
      print(f"  k_eff = {k_eff.n:.4f} ± {k_eff.s:.4f}")
      print(f"  Fission rate = {fission_rate:.3e}")