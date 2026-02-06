import argparse
import os
from ase.io import read, write
from typing import List
import numpy as np
import random
from pathlib import Path
from ase import Atoms
import copy

class CrystalConfigurationGenerator:
    """
    A class to process crystal structures from XDATCAR, POSCAR, or CONTCAR files,
    applying random crystal deformation and geometry rattling.

    Parameters:
        vasp_file (Path): Path to the input XDATCAR, POSCAR, or CONTCAR file.
        output_dir (Path, optional): Directory to store the generated POSCAR files. Default is './poscars_db'.
        max_strain (float, optional): Maximum random strain to apply. Must be between 0 and 1. Default is 0.05.
        max_amplitude (float, optional): Maximum random displacement amplitude. Must be between 0 and 1. Default is 0.1.
        start_structure_id (int, optional): Starting ID for generated structures. Default is 1.
        number_of_rattling (int, optional): Number of rattling for each structure. Defaul is 1.
        step_size (int, optional): Interval for selecting a configuration from list of structures.

    Raises:
        FileNotFoundError: If the file_path does not point to an existing file.
        ValueError: If max_strain or max_amplitude is not between 0 and 1.
    """
    def __init__(
        self,
        vasp_file: Path,
        output_dir: Path = Path("./poscars_db"),
        max_strain: float = 0.05,
        max_amplitude: float = 0.1,
        start_structure_id: int = 1,
        number_of_rattling: int = 1,
        step_size: int = 10
        
    ) -> None:

        if not (0 <= max_strain <= 1):
            raise ValueError(f"max_strain must be between 0 and 1. Got {max_strain}.")
        self.max_strain = max_strain
        
        if not (0 <= max_amplitude <= 1):
            raise ValueError(f"max_amplitude must be between 0 and 1. Got {max_amplitude}.")
        self.max_amplitude = max_amplitude
        
        if not vasp_file.is_file():
            raise FileNotFoundError(f"The file at path {self.file_path} does not exist.")
        self.file_path = Path(vasp_file)
        
        if not (start_structure_id > 0):
            raise ValueError(f"Minimum 'start_structure_id' must be 0. Got {start_structure_id}.")
        self.start_structure_id = start_structure_id

        if not isinstance(step_size, int):
            raise ValueError(f"step_size must be an integer. Got {type(step_size).__name__}")
        self.step_size = step_size

        if not isinstance(number_of_rattling, int):
            raise ValueError(f"number_of_rattling must be an integer. Got {type(number_of_rattling).__name__}")
        self.number_of_rattling = number_of_rattling

        self.output_dir = output_dir
        
    @staticmethod
    def get_seed(limit=999) -> int:
        """
        Generate a random integer seed up to a given limit.

        Parameters:
            limit (int): Maximum number for the seed.

        Returns:
            int: Random seed.
        """
        return random.randint(0, limit)

    def generate_strain_matrix(self) -> np.ndarray:
        """
        Generate a random 3x3 strain matrix.
        
        Returns:
            np.ndarray: A 3x3 array of random numbers between `-self.max_strain` and `self.max_strain`.
        """
        rng = np.random.default_rng()
        return rng.uniform(-self.max_strain, self.max_strain, size=(3, 3))

    def get_rattled_displacement_amplitude(self) -> float:
        """
        Generate a random displacement amplitude.

        Returns:
            float: A random displacement amplitude between 0 and 'self.max_amplitude'.
        """
        rng = np.random.default_rng()
        return rng.uniform(0.0, self.max_amplitude)

    def read_vasp_file(self) -> List:
        """
        Read and return the content of a VASP file.
        
        Returns:
            List: A list of structures (atoms objects) read from the VASP file.
        
        Raises:
            IOError: If the file cannot be read for other reasons.
        """
        try:
            return read(self.file_path, index=":")
        except IOError as e:
            raise IOError(f"An unexpected error occurred while reading the file at {self.file_path}: {e}")

    def create_output_directory(self) -> None:
        """
        Create the output directory if it does not exist.
        
        Ensures that `self.output_dir` exists by creating it, including any necessary
        parent directories. If the directory already exists, no action is taken.
        
        Raises:
            PermissionError: If the program lacks permissions to create the directory.
            OSError: If there is an issue creating the directory (e.g., invalid path).
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Permission denied: Unable to create the directory at {self.output_dir}.")
        except OSError as e:
            raise OSError(f"Failed to create the directory at {self.output_dir}. Error: {e}")

    def write_poscar(self, atoms_obj: Atoms, structure_id: int) -> None:
        """
        Write a POSCAR file for the given atomic structure.
        
        Parameters:
            atoms_obj (Atoms): The ASE Atoms object representing the atomic structure.
            structure_id (int): A unique identifier for the structure. This will be appended to the file name.
            
        Raises:
            ValueError: `atoms_obj` is not an instance of `Atoms`.
            IOError: If the file cannot be written to the specified directory.
        """
        if not isinstance(atoms_obj, Atoms):
            raise ValueError("The provided `atoms_obj` must be an instance of `ase.Atoms`.")

        file_path = os.path.join(self.output_dir, f'POSCAR-{structure_id}')
        
        try:
            write(file_path, atoms_obj, format='vasp', direct=True)
        except IOError as e:
            raise IOError(f"Failed to write POSCAR file at {file_path}. Error: {e}")

    def apply_cell_deformation(self, atoms_obj: Atoms) -> Atoms:
        """
        Apply a deformative strain to the cell vectors of the given atomic structure.
        
        Parameters:
            atoms_obj (Atoms): The ASE Atoms object representing the atomic structure.

        Returns:
            Atoms: The modified Atoms object with deformed cell vectors.

        Raises:
            ValueError: If `atoms_obj` is not an instance of `ase.Atoms`.
        """
        if not isinstance(atoms_obj, Atoms):
            raise ValueError("The provided `atoms_obj` must be an instance of `ase.Atoms`.")

        current_cell = atoms_obj.get_cell()
        strain_matrix = self.generate_strain_matrix()
        deformed_cell = current_cell * (1 + strain_matrix)
        
        atoms_obj.set_cell(deformed_cell, scale_atoms=True)
        return atoms_obj
    
    def rattle_structure(self, atoms_obj: Atoms) -> Atoms:
        """
        Apply random displacements (rattling) to the atomic positions.
        
        Parameters:
            atoms_obj (Atoms): The ASE Atoms object representing the atomic structure.
            
        Returns:
            Atoms: The modified Atoms object with rattled geometry.
            
        Raises:
            ValueError: If `atoms_obj` is not an instance of `ase.Atoms`.
            RuntimeError: If an error occurs during the rattling process.
        """
        if not isinstance(atoms_obj, Atoms):
            raise ValueError("The provided `atoms_obj` must be an instance of `ase.Atoms`.")
        
        try:
            displacement_amplitude = self.get_rattled_displacement_amplitude()
            random_seed = self.get_seed()
            atoms_obj.rattle(stdev=displacement_amplitude, seed=random_seed)
            return atoms_obj
        except Exception as e:
            raise RuntimeError(f"Failed to rattle the atomic structure: {e}")
        
    def process(self):
        """
        Process atomic structures from XDATCAR, POSCAR, or CONTCAR files.
        
        For each configuration in the input file, the method:
            1.1 Saves the original structure.
            2.1. Applies random strains to deform the structure.
            2.2. Rattles the atomic geometry to introduce random displacements.
            2.3. Saves the deformed and rattled structure to the output directory.

        Raises:
            RuntimeError: If any critical step in the process fails.
        """
        try:
            vaspcar = self.read_vasp_file()
            self.create_output_directory()
            structure_id = self.start_structure_id
        except Exception as e:
            raise RuntimeError(f"Error during initialization: {e}")
        
        interval = self.step_size
        for i, atoms in enumerate(vaspcar[::interval]):
            self.write_poscar(atoms, structure_id)
            structure_id += 1
            
            if self.max_strain != 0.0 or self.max_amplitude != 0:
                atoms = self.apply_cell_deformation(atoms)
                for j in range(self.number_of_rattling):
                    deformed_atoms = copy.deepcopy(atoms)
                    rattled_atoms = self.rattle_structure(deformed_atoms)
                    self.write_poscar(rattled_atoms, structure_id)
                    structure_id += 1
        print("Processing completed successfully.")

def main() -> None:
    """
    Main function to parse arguments and process VASP structure-related files.
    """
    parser = argparse.ArgumentParser(
        description="Process XDATCAR and generate strained and rattled structures."
    )
    parser.add_argument(
        "--vasp_file", type=Path, required=True,
        help="Path to the XDATCAR file. This argument is required."
    )
    parser.add_argument(
        "--max_strain", type=float, default=0.05,
        help="Maximum random strain to apply. Default is 0.05."
    )
    parser.add_argument(
        "--max_amplitude", type=float, default=0.1,
        help="Maximum random displacement amplitude. Default is 0.1."
    )
    parser.add_argument(
        "--start_structure_id", type=int, default=1,
        help="Starting ID for generated structures. Default is 1."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./poscars_db"),
        help="Directory to store the generated POSCAR files. Default is './poscars_db'."
    )
    parser.add_argument(
        "--number_of_rattling", type=int, default=1,
        help="Number of rattle operations for each configuration. Default is 1."
    )
    parser.add_argument(
        "--step_size", type=int, default=1,
        help="Interval to extract structures from XDATCAR. Default is 1."
    )

    args = parser.parse_args()

    try:
        processor = CrystalConfigurationGenerator(
            vasp_file=args.vasp_file,
            max_strain=args.max_strain,
            max_amplitude=args.max_amplitude,
            start_structure_id=args.start_structure_id,
            output_dir=args.output_dir,
            number_of_rattling=args.number_of_rattling,
            step_size=args.step_size
        )
        processor.process()
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
