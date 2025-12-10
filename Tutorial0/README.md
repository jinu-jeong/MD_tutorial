# Tutorial 0: LAMMPS Installation and Hello, World!

This tutorial guides you through installing LAMMPS and running a simple "Hello, world!" script.

## Table of Contents
1. [Creating a Conda Environment](#creating-a-conda-environment)
2. [Installing LAMMPS](#installing-lammps)
3. [Running the Hello, World! Script](#running-the-hello-world-script)
4. [Troubleshooting](#troubleshooting)

---

## Creating a Conda Environment

### 1. Check Conda Installation

First, verify that conda is installed on your system:

```bash
conda --version
```

If conda is not installed, please install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### 2. Create a New Conda Environment

It's a good practice to create a dedicated conda environment for LAMMPS. This keeps your installation isolated and makes it easier to manage dependencies.

Create a new environment named `lammps_env` (you can choose any name you prefer):

```bash
conda create -n lammps_env python=3.9
```

You can also specify a different Python version if needed:

```bash
conda create -n lammps_env python=3.10
```

### 3. Activate the Environment

After creating the environment, activate it:

```bash
conda activate lammps_env
```

You should see `(lammps_env)` at the beginning of your command prompt, indicating that the environment is active.

### 4. Add Conda-forge Channel

Add the conda-forge channel to ensure you can install LAMMPS:

```bash
conda config --add channels conda-forge
```

Note: This is a global setting, so you only need to do it once per conda installation.

---

## Installing LAMMPS

### 1. Install LAMMPS

With your conda environment activated, install LAMMPS:

```bash
conda install -c conda-forge lammps
```

Or to install a specific version:

```bash
conda install -c conda-forge lammps=2023.06.23
```

### 2. Verify Installation

Check if LAMMPS is properly installed:

```bash
lmp -version
```

Or:

```bash
lammps -version
```

If the installation was successful, you should see LAMMPS version information.

---

## Running the Hello, World! Script

### 1. Check Script File

Verify that the `hello_world.lmp` file exists in the current directory:

```bash
ls -la hello_world.lmp
```

### 2. Activate the Environment

Make sure your conda environment is activated:

```bash
conda activate lammps_env
```

### 3. Run LAMMPS

Execute the script using the following command:

```bash
lmp -in hello_world.lmp
```

Or:

```bash
lammps -in hello_world.lmp
```

### 4. Expected Output

If executed successfully, you should see the following output:

```
LAMMPS (date and version information)
Hello, world!
Welcome to LAMMPS!
```

### 5. Deactivate the Environment (Optional)

When you're done working, you can deactivate the conda environment:

```bash
conda deactivate
```

**Note**: Remember to activate the environment (`conda activate lammps_env`) each time you start a new terminal session to use LAMMPS.

---

## Troubleshooting

### Cannot Find LAMMPS Command

If you cannot find the `lmp` or `lammps` command:

1. **Activate the Environment**: Make sure your conda environment is activated.
   ```bash
   conda activate lammps_env
   ```

2. **Check Path**: Verify the location of the LAMMPS executable.
   ```bash
   which lmp
   which lammps
   ```

3. **Reinstall**: If the problem persists, reinstall LAMMPS.
   ```bash
   conda activate lammps_env
   conda remove lammps
   conda install -c conda-forge lammps
   ```

### Managing the Conda Environment

**List all environments:**
```bash
conda env list
```

**Remove the environment (if needed):**
```bash
conda env remove -n lammps_env
```

**Export environment (for sharing):**
```bash
conda activate lammps_env
conda env export > environment.yml
```

**Recreate environment from file:**
```bash
conda env create -f environment.yml
```

### Alternative Installation Methods

There are other ways to install LAMMPS besides conda:

- **Build from Source**: Download and build from source code available on the [LAMMPS official website](https://www.lammps.org/)
- **Package Manager**: Use system package managers (apt, yum, brew, etc.)

---

## Next Steps

Once you have completed the LAMMPS installation and basic execution, proceed to the next tutorials:

- **Tutorial 1**: System Equilibration and Production
- **Tutorial 2**: Basic Post-Processing

---

## References

- [LAMMPS Official Documentation](https://docs.lammps.org/)
- [LAMMPS GitHub](https://github.com/lammps/lammps)
- [Conda-forge LAMMPS Package](https://anaconda.org/conda-forge/lammps)
