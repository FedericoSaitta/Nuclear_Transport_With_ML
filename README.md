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
python3 -m pip install openmc
``` 

To test Python library installation:
``` bash
python3 test.py
``` 
