#!/bin/bash

# Ensure the required dependencies are installed
pip install --upgrade pip
pip install setuptools wheel

# Install the rest of the dependencies
pip install -r requirements.txt
