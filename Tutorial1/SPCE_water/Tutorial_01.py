import numpy as np

# SPCE parameters
mass_O = 15.9994
mass_H = 1.008
charge_O = -0.8476
charge_H = 0.4238
bond_length = 1.0     # Å
angle_deg = 109.47    # degrees

num_molecules = 903
box_size = 30.0       # Å

# -------------------------------------------------------
# Only generate positions and topology (NO velocities!)
# -------------------------------------------------------
def generate_positions(num_molecules, box_size, bond_length, angle_deg):
    positions = np.zeros((num_molecules * 3, 3))

    n_per_dim = int(np.ceil(num_molecules ** (1/3)))
    spacing = box_size / n_per_dim

    angle = np.radians(angle_deg / 2)
    hx = bond_length * np.sin(angle)
    hy = bond_length * np.cos(angle)

    m = 0
    for ix in range(n_per_dim):
        for iy in range(n_per_dim):
            for iz in range(n_per_dim):
                if m >= num_molecules:
                    return positions

                ox = (ix + 0.5) * spacing
                oy = (iy + 0.5) * spacing
                oz = (iz + 0.5) * spacing

                O = 3*m
                H1 = O + 1
                H2 = O + 2

                positions[O]  = [ox, oy, oz]
                positions[H1] = [ox + hx, oy + hy, oz]
                positions[H2] = [ox - hx, oy + hy, oz]

                m += 1

    return positions

positions = generate_positions(num_molecules, box_size, bond_length, angle_deg)

# -------------------------------------------------------
# Write data.lmp (NO velocities!)
# -------------------------------------------------------
with open("data.lmp", "w") as f:
    f.write("LAMMPS data file for SPCE water (minimal)\n\n")
    f.write(f"{len(positions)} atoms\n")
    f.write(f"{num_molecules*2} bonds\n")
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
    f.write(f"2 {mass_H}\n\n")

    # Atoms
    f.write("Atoms\n\n")
    atom_id = 1
    for mol_id in range(num_molecules):
        O = 3*mol_id
        H1 = O + 1
        H2 = O + 2
        f.write(f"{atom_id}     {mol_id+1}  1 {charge_O}  "
                f"{positions[O][0]} {positions[O][1]} {positions[O][2]}\n")
        f.write(f"{atom_id+1}   {mol_id+1}  2 {charge_H}  "
                f"{positions[H1][0]} {positions[H1][1]} {positions[H1][2]}\n")
        f.write(f"{atom_id+2}   {mol_id+1}  2 {charge_H}  "
                f"{positions[H2][0]} {positions[H2][1]} {positions[H2][2]}\n")
        atom_id += 3
    f.write("\n")

    # Bonds
    f.write("Bonds\n\n")
    bond_id = 1
    for mol_id in range(num_molecules):
        Oid = 3*mol_id + 1
        H1id = Oid + 1
        H2id = Oid + 2
        f.write(f"{bond_id} 1 {Oid} {H1id}\n")
        bond_id += 1
        f.write(f"{bond_id} 1 {Oid} {H2id}\n")
        bond_id += 1
    f.write("\n")

    # Angles
    f.write("Angles\n\n")
    for mol_id in range(num_molecules):
        Oid = 3*mol_id + 1
        H1id = Oid + 1
        H2id = Oid + 2
        f.write(f"{mol_id+1} 1 {H1id} {Oid} {H2id}\n")
