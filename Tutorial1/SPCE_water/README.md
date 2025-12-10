# SPCE Water Molecular Dynamics Simulation

This folder contains scripts for running SPCE water molecular dynamics simulations.

## Files

- `Tutorial_01.py`: Python script to generate initial SPCE water configuration
- `in.lmp`: LAMMPS input script for SPCE water simulation

## Usage

1. Generate initial configuration:
   ```bash
   python Tutorial_01.py
   ```
   This creates `data.lmp` with 903 SPCE water molecules in a 30×30×30 Å box.

2. Run LAMMPS simulation:
   ```bash
   lmp -in in.lmp
   ```

## Force Field

The simulation uses the SPCE (Simple Point Charge Extended) water model:
- Bond length: 1.0 Å
- Bond angle: 109.47°
- Charges: O = -0.8476, H = +0.4238
- LJ parameters: O-O (ε=0.15535 kcal/mol, σ=3.166 Å)

## Output

- `dump_EQ.xyz`: Equilibration trajectory
- `dump_Prod.xyz`: Production trajectory with velocities and forces







