### RNNSysID

Authors: Riley Ballachay

Language: 100% Python

Date: May-September 2020

Supervision: Dr. Bhushan Gopaluni, Department of Chemical and Biological Engineering, UBC

## Into
This repository contains the code for the project Connecting Deep and Bayesian Learning: Estimating Parameters with Uncertainty for Linear Systems.
There are two main scripts: Model.py and Signal.py, off of which the rest of the scripts depend. These can be found in the src folder.

## Files: 
Signal.py - Contains a function to create PRBS signal with varying frequency, function to produce system responses and stack into multidimensional array, function to shuffle and transform signals prior to training

Model.py - Contains the deep learning architecture, function to train the deep learning model using an instance of the Signal class and function to test a saved deep learning model with instance of Signal class.

TO DO:
- Convert into importable package for use by others
