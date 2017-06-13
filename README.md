# Introduction

Constant potential method is an approach to describe charges on electrode atoms in Molecular Dynamics(MD) simulations of Electric Double-Layer Capacitors(EDLCs). The advantage is to take into account the charge fluctuations on the electrode induced by local density fluctuations in the electrolyte solution. This method was developed by Reed et al.<sup>[1]</sup> and some derivation was corrected by Gingrich and Wilson<sup>[2]</sup> later.

The aim of this project is to implement this method into LAMMPS.

Please cite the following article if using this code.

Z. Wang, Y. Yang, D. L. Olmsted, M. Asta and B. B. Laird, J. Chem. Phys. 141, 184102 (2014). http://dx.doi.org/10.1063/1.4899176

# Installation

1. Download **fix_conp.cpp** and **fix_conp.h** to [LAMMPS home directory]/src/

2. If using gnu compiler, download and compile LAPACK. Note BLAS within LAPACK package needs to be compiled first.
http://www.netlib.org/lapack/

3. Add library files of LAPACK and BLAS (file name is refblas if using default setting when compiling BLAS within LAPACK package) to link flag in LAMMPS Makefile. For Intel compiler with MKL, the corresponding library files are mkl_blacs and mkl_lapack95.

4. Compile LAMMPS as usual.

# Syntax
This method is turned on through a FIX command

```
fix [ID] all conp [Nevery] [η] [Molecule-ID 1] [Molecule-ID 2] [Potential 1] [Potential 2] [Method] [Log] [Matrix]
```

**ID** = ID of FIX command

**Nevery** = Compute charge every this many steps (set to 1 for current version)

**η** = Parameter for Gaussian charge. The unit is is angstrom<sup>-1</sup> (see note below)

**Molecule-ID 1** = Molecule ID of first electrode (the second column in data file)

**Molecule-ID 2** = Molecule ID of second electrode

**Potential 1** = Potential on first electrode (unit: V)

**Potential 2** = Potential on second electrode

**Method** = Method for solving linear equations. "inv" for inverse matrix and "cg" for conjugate gradient

**Log** = Name of log file recording time usage of different parts

**Matrix** = Optional argument. File name of A matrix to read in. If it is assigned, A matrix is read in instead of calculation

# Note

Current version is compatible with 11Apr14 or later version of LAMMPS. Also some limitations exist and certain settings are required.

* Only simulation with two electrodes is supported
* Only pair style **lj_cut_coul_long** is supported
* **RESPA** is not supported
* **Fix npt** is not supported
* **Newton** must be **off**
* **Unit** must be **real**. As so the unit of **η** is angstrom<sup>-1</sup>. For example, as in our work it is 19.79 nm<sup>-1</sup>, the actual value of **η** in command is 1.979
* Electrodes need to be frozen (set the force on electrode atoms to 0 and exclude electrode atoms from integration)
* The simulation cell must be symmetric with respect to z=0 plane
* Two electrodes must be assigned equal but opposite potentials. For example, for a 5V potential difference, the potential on lower electrode should be -2.5V and on upper electrode should be 2.5V

# Example input file

The example files are for a system of acetonitrile between two graphite electrodes with potential difference as 1V. Acetonitrile is described by a united atom model (in example data file, CH<sub>3</sub>, C and N are named as CAC, CAB and NAA respectively). Inital charge on carbon in electrodes is zero (carbon is named as CG in example data file). The references of parameters can be found in our paper.

# Reference
[1] S. K. Reed, O. J. Lanning, and P. A. Madden, J. Chem. Phys. 126, 084704 (2007).

[2] T. R. Gingrich and M. Wilson, Chem. Phys. Lett. 500, 178 (2010).
