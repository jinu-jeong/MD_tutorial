#%%
from utils import *
import matplotlib.pyplot as plt
import os

# Define systems to process
systems = [
    {
        'name': 'SPCE_water',
        'path': '../Tutorial1/SPCE_water/dump_Prod.xyz',
        'com_func': compute_com_spce_water,
        'output_dir': 'SPCE_water'
    }
]

# Process each system
for system in systems:
    print(f"\n{'='*60}")
    print(f"Processing {system['name']}")
    print(f"{'='*60}")
    
    path = system['path']
    com_func = system['com_func']
    output_dir = system['output_dir']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping...")
        continue
    
    # Read xyz file
    print(f"Reading {path}...")
    xyz, atom_type, box = read_xyz_file(path, same_atom_type=True, same_box_size=True)
    
    if len(xyz) == 0:
        print(f"Warning: No data found in {path}. Skipping...")
        continue
    
    # Compute center of mass
    print("Computing center of mass...")
    com = com_func(xyz, atom_type)
    
    # Extract COM positions and velocities
    com_pos = com[:, :, :3]  # (n_frames, n_molecules, 3)
    com_vel = com[:, :, 3:6]  # (n_frames, n_molecules, 3)
    
    # Box size
    box = np.diag(box)
    print(f"Box size: {box}")
    print(f"Number of molecules: {com_pos.shape[1]}")
    print(f"Number of frames: {com_pos.shape[0]}")
    
    # RDF computation
    print("Computing RDF...")
    compute_RDF = RDF_computer(torch.tensor(box), torch.device('cpu'), 10.0)
    # Sample every 10th frame for RDF
    r_list, RDF = compute_RDF(torch.tensor(com_pos[::10], dtype=torch.float32))
    
    # MSD computation
    print("Computing MSD...")
    # maximum time delay: 1ps (1000 fs), number of ensembles: 1
    compute_MSD = MSD_computer(1000, 1)
    MSD = compute_MSD(torch.tensor(com_pos, dtype=torch.float32))
    
    # VACF computation
    print("Computing VACF...")
    compute_VACF = VACF_computer(1000)
    VACF = compute_VACF(torch.tensor(com_vel, dtype=torch.float32))
    
    # Save results
    print(f"Saving results to {output_dir}/...")
    torch.save(r_list, f'{output_dir}/r_list.pt')
    torch.save(RDF, f'{output_dir}/RDF.pt')
    torch.save(MSD, f'{output_dir}/MSD.pt')
    torch.save(VACF, f'{output_dir}/VACF.pt')
    
    # Plot RDF
    plt.figure(figsize=(8, 6))
    plt.plot(r_list.numpy(), RDF.numpy(), label=f'{system["name"]} RDF')
    plt.xticks(fontsize=18)
    plt.xlabel('r [$\AA$]', fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('RDF', fontsize=18)
    plt.legend(fontsize=18)
    plt.title(f'{system["name"]} - Radial Distribution Function')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/RDF.png', dpi=150)
    plt.close()
    
    # Plot MSD
    plt.figure(figsize=(8, 12))
    plt.subplot(2, 1, 1)
    plt.plot(MSD.numpy(), label=f'{system["name"]} MSD')
    plt.xticks(fontsize=18)
    plt.xlabel('t [fs]', fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('MSD [$\AA^2$]', fontsize=18)
    plt.legend(fontsize=18)
    plt.title(f'{system["name"]} - Mean Squared Displacement (linear scale)')
    
    plt.subplot(2, 1, 2)
    plt.plot(MSD.numpy(), label=f'{system["name"]} MSD')
    plt.xticks(fontsize=18)
    plt.xlabel('t [fs]', fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('MSD [$\AA^2$]', fontsize=18)
    plt.legend(fontsize=18)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'{system["name"]} - Mean Squared Displacement (log-log scale)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/MSD.png', dpi=150)
    plt.close()
    
    # Plot VACF
    plt.figure(figsize=(8, 6))
    plt.plot(VACF.numpy(), label=f'{system["name"]} VACF')
    plt.xticks(fontsize=18)
    plt.xlabel('t [fs]', fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('VACF [$\AA^2/fs^2$]', fontsize=18)
    plt.legend(fontsize=18)
    plt.title(f'{system["name"]} - Velocity Autocorrelation Function')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/VACF.png', dpi=150)
    plt.close()
    
    # Calculate diffusion coefficients
    print("\nDiffusion coefficient calculation:")
    print(f"MSD method [m^2/s]: {MSD[-1].item() / len(MSD) / 2 * 1e-20 / 1e-15:.4e}")
    print(f"VACF method [m^2/s]: {torch.sum(VACF).item() * 1e-20 / 1e-15:.4e}")
    
    print(f"\n{system['name']} processing complete!")
    print(f"Results saved in {output_dir}/")

print("\n" + "="*60)
print("All systems processed!")
print("="*60)

# %%
