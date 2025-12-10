cd CO2/
mpirun -np 8 lmp_mpi -in in.lmp
cd ..
cd Graphene_confined_SPCE
mpirun -np 8 lmp_mpi -in in.lmp
cd ..
cd SPCE_water
mpirun -np 8 lmp_mpi -in in.lmp
cd ..
