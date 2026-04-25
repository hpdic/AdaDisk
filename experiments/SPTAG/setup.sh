#!/bin/bash

# Install Docker
sudo apt-get install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker

# Install Python dependencies
source ~/hpdic/AdaDisk/venv/bin/activate
python --version

# Install SPTAG
cd ~/hpdic
git clone git@github.com:hpdic/SPTAG.git
cd SPTAG
git submodule update --init --recursive
docker build -t sptag_env .
docker run -it --name my_spann -v $PWD:/workspace sptag_env /bin/bash
# In docker container, e.g., docker exec -it my_spann /bin/bash
cd /app/build
rm -f CMakeCache.txt
cmake .. -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so
make -j
cd /app/Release
./SPTAGTest
# ctrl + c to exit the container after testing successfully starts
cp -r /app/Release /workspace/
exit
ls Release/
sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
docker run -it --name spann_lab -v ~/hpdic/SPTAG:/app -v ~/hpdic/sift1b_data:/data sptag_env /bin/bash
ls ../data
cd Release
mkdir -p /data/indices/spann_sift_1b
cat << EOF > paper_config.ini
[Base]
ValueType=UInt8
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=128
VectorPath=/data/data_sift1b.bin
VectorType=DEFAULT
IndexDirectory=/data/indices/spann_sift_1b

[SelectHead]
isExecute=true
SamplesNumber=5000000
NumberOfThreads=96

[BuildHead]
isExecute=true
TPTNumber=32
NumberOfThreads=96

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum=64
ReplicaCount=8
NumberOfThreads=96
EOF
apt install vim -y
./ssdserving paper_config.ini
