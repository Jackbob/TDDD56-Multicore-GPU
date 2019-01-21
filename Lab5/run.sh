#!/bin/bash
nvcc filter.cu -c -arch=sm_30 -o filter.o
g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
./filter