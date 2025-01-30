import sys
from vina import Vina

# Read ligand PDBQT string from stdin
ligand_pdbqt_content = sys.stdin.read()

# Initialize Vina
v = Vina(sf_name='vina')

# Load receptor
v.set_receptor('Data/9F6A_prepared.pdbqt')

# Set search space
v.compute_vina_maps(center=[136, 172, 99], box_size=[12, 7, 8])

# Load ligand from the provided string
v.set_ligand_from_string(ligand_pdbqt_content)

# Dock
v.dock(exhaustiveness=8)

# Output results
print(v.poses())
