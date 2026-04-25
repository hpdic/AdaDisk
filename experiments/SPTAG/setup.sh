#!/bin/bash

# Install Docker
sudo apt-get install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker

# Install Python dependencies
source ../AdaDisk/venv/bin/activate
python --version

# Install SPTAG
cd ~/hpdic
git clone git@github.com:hpdic/SPTAG.git
cd SPTAG
git submodule update --init --recursive
docker build -t sptag_env .
docker run -it -v $PWD:/workspace sptag_env /bin/bash
# In docker container:
cd /app/build
rm -f CMakeCache.txt
cmake .. -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so
make -j
cd /app/Release
./SPTAGTest
