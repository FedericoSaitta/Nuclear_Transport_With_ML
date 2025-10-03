import openmc

print("OpenMC version:", openmc.__version__)

material = openmc.Material(name="Empty")
print("Material created:", material)

geometry = openmc.Geometry()
cell = openmc.Cell(name="Empty cell")
geometry.root = cell
print("Geometry object created:", geometry)

print("OpenMC Python API works")