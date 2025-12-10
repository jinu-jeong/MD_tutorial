# SPCE Water Post-Processing

This folder contains post-processing scripts for SPCE water simulations.

## Files

- `Tutorial_02.py`: Main post-processing script for computing RDF, MSD, and VACF
- `utils.py`: Utility functions for reading XYZ files and computing correlation functions

## Usage

1. Make sure you have run the MD simulation in `Tutorial1` and generated `dump_Prod.xyz`
2. Install PyTorch (CPU-only example) and matplotlib:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install matplotlib
   ```
3. Run the post-processing script:
   ```bash
   python Tutorial_02.py
   ```

## Output

The script generates:
- `r_list.pt`: Radial distance list for RDF
- `RDF_OO.pt`: Oxygen-oxygen radial distribution function
- `MSD.pt`: Mean squared displacement
- `VACF.pt`: Velocity autocorrelation function







