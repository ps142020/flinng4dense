#!/bin/bash
cd src
make clean;make 
wait
sudo make uninstall; sudo make install
wait
cd ../test
make clean;make
