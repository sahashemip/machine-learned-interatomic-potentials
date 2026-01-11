#!/usr/bin/env python3
import argparse
from ase.io import read, write

def main():
    parser = argparse.ArgumentParser(description="Convert frame from dump.xyz to VASP POSCAR")
    parser.add_argument('indx', type=int, help='Frame index to read from dump.xyz')

    args = parser.parse_args()
    indx = args.indx

    dump = read('dump.xyz', format='extxyz', index=f'{indx}')
    write(f'POSCAR-{indx}', dump, format='vasp', direct=True, sort=True)

if __name__ == "__main__":
    main()
