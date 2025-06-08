# Computer Animation and Special Effects Final Project

This repository contains the final project for the course **Computer Animation and Special Effects**.

## Project Overview

We implemented hair and fur simulation using different methods and compared their performance and visual results.

### Structure

- **2d/**:  
  Contains 2D simulation scripts for testing individual hair strands.
  - `compare.py`: Compare different simulation methods
  - `interactive.py`: Interactive testing interface
  - `metric.py`: Evaluation metrics (e.g., constraint error, stability)

- **3d/**:  
  Contains the 3D simulation based on Blender and Python.
  - `hair.py`: 3D hair simulation script
  - `hair_code.blend`: Blender file with hair model and setup

## Simulation Methods

Implemented and compared the following methods:
- Dynamic Follow-The-Leader (DFTL)
- Position-Based Dynamics (PBD)
- Symplectic Euler