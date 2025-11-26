# tests/test.py
import openmc
import openmc.deplete

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