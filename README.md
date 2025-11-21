# Nuclear_Transport_With_ML

## Installation Guide

To clone this repository and its submodules:
``` bash
git clone --recurse-submodules https://github.com/FedericoSaitta/Nuclear_Transport_With_ML.git
```

To build the openMC library and use its Python API
``` bash
cd Nuclear_Transport_With_ML/external/openmc
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make -j$(nproc)  # Build using multiple cores
make install
cd ..
```
To install the Python bindings
``` bash
python -m pip install .
``` 

To install the ML package and its dependencies run in the Root folder:
``` bash
pip install -e .
``` 


To test Python library installation:
``` bash
python test/test.py
```

## Downloading Data
To download the cross section data (7 Gb) and Depletion chains (30 Mb) can be done here: https://openmc.org/official-data-libraries/.
The cross section data contains an .xml file along with three folders: Neutron, Photon and wmp. The depletion data is a single .xlm file.
