#!/bin/bash

cd dataset/images2
python generate.py
cd ..
cd ..
octave DOGS_CATS.m