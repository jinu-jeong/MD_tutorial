#%%
"""
This code is to generate a LAMMPS data file for SPCE water model.
I recommend in your actual implementation, you might want to use either use VMD, Packmol, etc., because they generate well-equilibrated system, minimizing additional equilibration and stabilization.

Another advice is to make sure about the units you are using.
If you are a beginner, I recommend you to follow LAMMPS unit systems: LJ unit, real unit, and metal unit are popularly used
https://lammps.sandia.gov/doc/units.html
"""
#%%
import numpy as np

# Physics parameters for SPCE water model. Unit: LAMMPS real unit.
mass_O = 15.9994  # Oxygen mass in atomic mass units
mass_H = 1.008  # Hydrogen mass in atomic mass units
charge_O = -0.8476  # SPCE charge on Oxygen
charge_H = 0.4238  # SPCE charge on Hydrogen
bond_length = 1.0  # Bond length in SPCE model in Angstrom
angle = 109.47  # H-O-H bond angle in SPCE model

# System parameters. Unit: LAMMPS real unit.
num_molecules = 903  # Number of water molecules (approximation)
box_size = 30  # Box size in Angstrom
temperature = 300  # K

# Function to generate positions in a simple cubic lattice
def generate_positions(num_molecules, box_size, bond_length, angle):
    num_atoms = num_molecules * 3
    positions = np.zeros((num_atoms, 3))
    spacing = (box_size / num_molecules**(1/3))
    count = 0
    
    # Convert angle to radians for trigonometric functions
    angle_rad = np.radians(angle)
    H_O_H_half_angle = angle_rad / 2
    
    # Calculate positions of hydrogen atoms relative to the oxygen atom
    h1_x = bond_length * np.sin(H_O_H_half_angle)
    h1_y = bond_length * np.cos(H_O_H_half_angle)
    h2_x = -h1_x
    h2_y = h1_y

    for i in range(num_molecules):
        x = (i % num_molecules**(1/3)) * spacing
        y = ((i // num_molecules**(1/3)) % num_molecules**(1/3)) * spacing
        z = (i // (num_molecules**(1/3) * num_molecules**(1/3))) * spacing
        positions[count] = [x, y, z]
        positions[count + 1] = [x + h1_x, y + h1_y, z]
        positions[count + 2] = [x + h2_x, y + h2_y, z]
        count += 3
    return positions

# Function to generate velocities based on temperature
def generate_velocities(num_atoms, temperature):
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    T = temperature
    mass_O_kg = mass_O * 1.66053906660e-27  # Convert to kg
    mass_H_kg = mass_H * 1.66053906660e-27  # Convert to kg

    std_dev_O = np.sqrt(k_B * T / mass_O_kg)
    std_dev_H = np.sqrt(k_B * T / mass_H_kg)

    velocities = np.zeros((num_atoms, 3))
    for i in range(0, num_atoms, 3):
        velocities[i] = np.random.normal(0, std_dev_O, 3)
        velocities[i+1] = np.random.normal(0, std_dev_H, 3)
        velocities[i+2] = np.random.normal(0, std_dev_H, 3)
    return velocities

# Generate positions and velocities
positions = generate_positions(num_molecules, box_size, bond_length, angle)
velocities = generate_velocities(len(positions), temperature)

# Create the LAMMPS data file
with open('data.lmp', 'w') as f:
    f.write("LAMMPS data file for SPCE water\n\n")
    f.write(f"{len(positions)} atoms\n")
    f.write(f"{2 * num_molecules} bonds\n")
    f.write(f"{num_molecules} angles\n\n")
    f.write("2 atom types\n")
    f.write("1 bond types\n")
    f.write("1 angle types\n\n")
    f.write(f"0.0 {box_size} xlo xhi\n")
    f.write(f"0.0 {box_size} ylo yhi\n")
    f.write(f"0.0 {box_size} zlo zhi\n\n")
    
    # Masses
    f.write("Masses\n\n")
    f.write(f"1 {mass_O}\n")
    f.write(f"2 {mass_H}\n")
    f.write("\n")
    
    # Atoms
    f.write("Atoms\n\n")
    atom_id = 1
    for mol_id in range(num_molecules):
        f.write(f"{atom_id} {mol_id + 1} 1 {charge_O} {positions[atom_id - 1][0]} {positions[atom_id - 1][1]} {positions[atom_id - 1][2]}\n")
        f.write(f"{atom_id + 1} {mol_id + 1} 2 {charge_H} {positions[atom_id][0]} {positions[atom_id][1]} {positions[atom_id][2]}\n")
        f.write(f"{atom_id + 2} {mol_id + 1} 2 {charge_H} {positions[atom_id + 1][0]} {positions[atom_id + 1][1]} {positions[atom_id + 1][2]}\n")
        atom_id += 3
    f.write("\n")
    
    # Velocities
    f.write("Velocities\n\n")
    for i in range(len(velocities)):
        f.write(f"{i + 1} {velocities[i][0]} {velocities[i][1]} {velocities[i][2]}\n")
    f.write("\n")
    
    # Bonds
    f.write("Bonds\n\n")
    bond_id = 1
    for i in range(0, len(positions), 3):
        f.write(f"{bond_id} 1 {i + 1} {i + 2}\n")
        bond_id += 1
        f.write(f"{bond_id} 1 {i + 1} {i + 3}\n")
        bond_id += 1
    f.write("\n")
    
    # Angles
    f.write("Angles\n\n")
    for i in range(num_molecules):
        f.write(f"{i + 1} 1 {3*i + 2} {3*i + 1} {3*i + 3}\n")
    f.write("\n")

# %%
