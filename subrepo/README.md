## `Defective hBN/SiC vdW Heterostructure using VASP-NEP-GPUMD`

This project provides a guide for developing **machine learning interatomic potentials** and conducting molecular dynamics (MD) simulations with [GPUMD software](https://gpumd.org/), for a prototypical bilayer heterostructure composed of hexagonal boron nitride (hBN) and silicon carbide (SiC).

The developed potential can be used to (i) assess the stability of the two-dimensional hBN/SiC heterostructure, (ii) elucidate **Si–N interlayer bond formation** triggered by a boron vacancy (*V*<sub>B</sub>), and (iii) investigate the behaviour of **Cu adatoms** on the defective hBN/SiC surface.

---

## Database Compilation

<p>
  <img src="figures/model.png"
    alt="The hBN/SiC heterostructure. Unique $\rm{V}_{\rm{B}}$ sites are determined by red circles."
    width="410"
    align="right"
  >
</p>

Detailed instructions are provided for building a robust database (DB) for neural evolution potential (NEP) developments.
[Density-functional theory (DFT)](https://www.synopsys.com/glossary/what-is-density-functional-theory.html) calculations were performed with the [Vienna Ab-initio Simulation Package (VASP)](https://www.vasp.at/) to obtain the total energy, atomic forces, and virial stress for each geometry.
Defects were introduced into a supercell containing 100 boron (B), 100 nitrogen (N), 64 silicon (Si), and 64 carbon (C) atoms.
More specifically, *V*<sub>B</sub> and copper (Cu) adatoms serving as defects were incorporated into the structure.
Detailed information on the geometries and structures is provided below.

#### 1. Boron Vacancy (V<sub>B</sub>) Defects
**Lattice mismatch** lead to 15 symmetry-distinct V<sub>B</sub> defects in the heterostrcuture supercell, marked by the red circles in the figure.
All unique defects were created, and their geometries were optimized.
Depending on the *V*<sub>B</sub> site, various numbers of **local** interlayer chemical bonds (ranging from 0 to 4) are formed.
Trajectories produced during geometry-optimization (using ISIF=4) are stored in the VASP `XDATCAR` files.
The workflow then proceeds in two steps:
- Use the script `src/vasp_structure_rattler_deformer.py` with `--max_strain=0.05`, `--max_amplitude=0.1`, and `--step_size=2` to generate `POSCAR` files from XDATCARs (forming a dataset of about **1113 structures**).
- Perform single-shot DFT calculations, **with higher precision**, for these structures.

#### 2. Ab-Initio Molecular Dynamics for Stable Defects
- For the **most stable defect structure** (with 4 chemical bonds), ab-initio molecular dynamics (AIMD) simulations were conducted:
  - **Temperatures:** 300 K, 500 K, 700 K, and 900 K
  - **Ensemble:** NVT
  - **Numer of steps:** 6000 steps
  - **VASP parameters:** ENCUT = 400 eV, EDIFF = 1E-4 eV, timestep = 0.5 fs 
- Using `--step_size = 40`, `--max_strain=0.0`, and `--max_amplitude=0.0`, we added 600 additional structures to the DB and performed DFT calculations for each of them, but this time with higher accuracy. Below is the INCAR file for VASP calculations:
  ```
  PREC=Accurate
  ENCUT=500
  IBRION=-1
  IVDW=12
  EDIFF=1E-7
  ISMEAR=0
  SIGMA=0.01
  LMAXMIX=4
  NWRITE=1
  GGA=PE
  NCORE=16
  ALGO = Normal
  ISPIN=2
  LASPH = .TRUE.
  LWAVE=.FALSE.
  LCHARG=.FALSE.
  NELM=77
  ```
**`In some cases, SCF convergence can be challenging. For those systems, we recommend restarting the calculation from the WAVECAR files.`** This INCAR file was used for all data in the data sets.

#### 3. Pristine System
- For the pristine system, **120 additional structures** were generated through atomic rattling, in-plane layer shifting, and variations in interlayer distances, and subsequently added to the database. To mimic interlayer sliding, the hBN sheet was incrementally translated from (0, 0) to (ax/2, ay/2), where the rectangular SiC lattice parameters are ax = 3.09 Å and ay = 5.35 Å.

#### 4. Data Enhancement via Iterative Model Refinement
An initial NEP model was trained on the 1833 configurations described above.
Using this model, we performed GPUMD molecular dynamics simulations to identify configurations exhibiting non-physical behavior or other notable characteristics.
These newly identified configurations were iteratively added to the database and used to retrain the potential, progressively enhancing its accuracy and stability.
The MD simulations were conducted in the NPT ensemble (isothermal–isobaric) using the Berendsen barostat (P = 1 kbar).
A range of temperatures from 200 K to 1000 K was explored during these calculations. In total, 901 new configurations were added to the DB, including pristine structures as well as mono-vacancies.
Note, the supercells used in this case are rectangular and contain the same number of atoms as the original triangular cell.

`The NEP was developed using the **nep** executable from the GPUMD package with the following input data:`
```
version      4
model_type   0

type         4 B N Si C

cutoff       4.2 3.5
n_max        4 4
l_max        4 0 0

neuron       30
batch        100000
generation   200000
```
The root-mean-square-error (RMSE) values for energy and force reach below 0.003 eV/atom and 0.06 eV/Å, respectively.
The NEP potential was employed for machine learning MD (MLMD) simulations using the following `run.in' input:

```
potential  ./nep.txt
velocity    300 

time_step   0.5

ensemble    nvt_ber 300 300 100
dump_thermo 100
dump_exyz   100 0 0
run         1000000

```
To select specific snapshots, we visualized the GPUMD‐generated `dump.xyz` trajectories and used `src/dump2poscar.py` to convert the chosen frames into POSCAR files.
Clearly, these POSCARs are used in DFT calculations to enhance DB quality.

#### 5. Di- and Tri-Boron Vacancy defects
- To expand our DB with higher concentrations and random defect geometries, we generated unique defect configurations by randomly removing two and three boron vacancies. This dataset contributed 400 new entries to the DB.
All structures were relaxed with the `Step 4` NEP potential and subsequently subjected to single‑point calculations using the same high‑precision settings described above.

<table border="1"><tr><td>
<strong>NOTE&nbsp;</strong>─ Next, we broaden the database to capture the nuanced anchoring of Cu adatoms to *V*<sub>B</sub> on the hBN/SiC surface.
</td></tr></table>

#### 6. AIMD Simulations of Distinct Monovacancy-Cu Coupling Configurations
- For four energetically distinct defective hBN/SiC structures, resulted by various number of **N-Si bonds**, we performed AIMD simulations similar to `Step 2`. One structure was extracted every 300 steps, resulting in a total of 80 new configurations added to the database.

<p>
  <img src="figures/confcoords.png"
    alt="Energy profile of the structural transition from a bonded hBN-SiC configuration to a bonded structure in the presence of Cu metal."
    width="225"
    align="right"
  >
</p>

#### 7. Configurational Coordinates Transitioning from Bonded to Non-Bonded Defective Structures
- Copper atoms can be introduced into both interlayer bonded and non-bonded structures. To investigate the transition barrier, we explored the energy profile along a one-dimensional coordinate. A total of 21 structures, along with their rattled and deformed counterparts (42 structures in total), were added to the database.

<table border="1"><tr><td>
<strong>NOTE&nbsp;</strong>─ Using the updated database, a new NEP model was trained to enhance accuracy, and the process of `Data Enhancement via Iterative Model Refinement` will continue in a sequential manner by adding more Cu atoms. The hBN/SiC surface migth host several Cu and V<sub>B</sub> defect.
</td></tr></table>

`The NEP was developed using the following input data:`
```
version      4
model_type   0

type         5 B N Si C Cu

cutoff       4.2 3.5
n_max        4 4
l_max        4 0 0

neuron       30
batch        100000
generation   200000
```
Similarly, the RMSE values for energy and force reach below 0.003 eV/atom and 0.06 eV/Å, respectively.

#### 8. Data Enhancement via Iterative Model Refinement
The NEP was refined iteratively in separate stages, with additional data incorporated at each step.
Representative defects and the resulting structures are listed below. In each example, the initial geometry is a pristine rectangular cell with a target boron vacancy and a Cu adatom. Running MLMD in the NVT ensemble then yields a variety of configurations. For instance, some Cu atoms may anchor at the vacancy and interlayer bonds may form.
- 1*V*<sub>B</sub> & 1Cu (108 strcutures)
- 1*V*<sub>B</sub> & 2Cu (128 structures)
- 1*V*<sub>B</sub> & 3Cu (106 structures)
- 2*V*<sub>B</sub> & 1Cu (106 structures)
- 2*V*<sub>B</sub> & 2Cu (75 structures)
- 1*V*<sub>B</sub> & 4Cu (109 structures)
- 1*V*<sub>B</sub> & 5Cu (92 structures)
- 1*V*<sub>B</sub> & 6Cu (66 structures)
- 1*V*<sub>B</sub> & 7Cu (17 structures)
- 4*V*<sub>B</sub> & 9Cu (9 structures)

<table border="1"><tr><td>
<strong>NOTE&nbsp;</strong>─ The finalized dataset consists of 3,953 entries, of which 3,361 (85%) were used for training and 592 (15%) reserved for testing.
</td></tr></table>

---

## Final NEP Quality
The NEP model (`nep.txt`) was trained using radial and angular cutoffs of 5 Å and 3.5 Å to represent the potential‐energy surface.
A single hidden layer of 50 neurons yielded the following RMSEs:

* **Test set:**
  – Energy: 2.2 meV/atom
  – Force: 123 meV/Å
  – Virial: 9.4 meV/atom

* **Training set:**
  – Energy: 2.3 meV/atom
  – Force: 115 meV/Å
  – Virial: 9.4 meV/atom

Dynamic simulations demonstrate the NEP model's stability under diverse conditions (see embedded videos for real-time trajectories).
Detailed results, including trajectory snapshots and stability metrics, are presented in the figure below.

<p align="center">
  <img src="figures/nep.png"
    alt="Evolution of the NEP training process and model accuracy. (a–c) Log‑loss curves for energy, force, and virial, respectively, plotted against generation for both training (solid lines) and testing (dashed lines) data sets. (d–f) Correlation between NEP‑predicted and DFT reference values for energy, force, and virial on the training (circles) and testing (squares) sets; the dashed diagonal indicates perfect agreement, and RMSE values are reported in each legend."
    width="900"
    align="center"
  >
</p>


---

### How to Use `vasp_structure_rattler_deformer.py`

The `vasp_structure_rattler_deformer.py` script generates strained and rattled `POSCAR` files from a VASP `XDATCAR`.

#### Example Command
Run the script with the following command:
```bash
python vasp_structure_rattler_deformer.py \
    --max_strain 0.05 \
    --max_amplitude 0.01 \
    --start_structure_id 1 \
    --vasp_file ~/XDATCAR \
    --step_size 1 \
    --number_of_rattling 1
```
### Command-Line Parameters

| Parameter | Description | Example|
| ------ | ------ | ------ |
| `--max_strain` | Maximum random strain to apply. | `0.05`
| `--max_amplitude` | Maximum random displacement amplitude. | `0.1`
| `--start_structure_id` | Starting ID for generated structures, greater than 0. | `1`
| `--step_size` | Interval to extract structures from file. | `1`
| `--number_of_rattling` | Number of rattle operations for each configuration. | `1`
| `--vasp_file`  | Path to the VASP structure file. This argument is required. | `./XDATCAR`
| `--output_dir` | Directory to store the generated POSCAR files. | `./poscars_db`

### Citation
If you use this workflow or data in your research, please cite the following:
  - Our paper ...

### License
This project is licensed under the MIT License. See the LICENSE file for details.
