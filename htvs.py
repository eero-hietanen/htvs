import ringtail as rtc
from joblib import Parallel, delayed
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

# Function to convert a ligand to .pdbqt format using Meeko
def convert_to_pdbqt_string(mol_idx, mol):
    try:
        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        Chem.rdDistGeom.EmbedMolecule(mol)

        # Prepare ligand with Meeko
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        for setup in mol_setups:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                return mol_idx, pdbqt_string
        return mol_idx, pdbqt_string, None  # No error
    except Exception as e:
        return mol_idx, None, str(e)  # Return error

# Function to dock a batch of ligands with Vina and capture results as strings
def dock_ligands_with_vina(vina_instance, receptor_file, docking_box, ligands):
    vina_instance.set_receptor(receptor_file)
    vina_instance.compute_vina_maps(**docking_box)

    # Initialize an empty dictionary to store the results
    vina_output = {}

    # Store results as strings
    vina_results = []
    for mol_idx, pdbqt_string in ligands:
        try:
            vina_instance.set_ligand_from_string(pdbqt_string)
            vina_instance.dock(exhaustiveness=8, n_poses=5)
            vina_poses = vina_instance.poses()

            # Build a results dictionary from the Vina docking results that can be used as an input for Ringtail database.
            # The result dictionary key should be the pdbqt_string of the liganda, and the value should be the vina_poses.
            vina_output[pdbqt_string] = vina_poses
  
        except Exception as e:
            print(f"Error docking ligand {mol_idx}: {e}")

    return vina_output

# Batch processing parameters
batch_size = 1000  # Number of ligands per batch
n_jobs_meeko = -1  # Number of CPU cores for Meeko conversion (use all available cores)

# Load ligands from .sdf file
sdf_file = "Data/ligands_10.sdf"
suppl = Chem.SDMolSupplier(sdf_file)

# Define receptor and docking box
receptor_file = "Data/9f6a.pdbqt"
docking_box = {"center": [136.733, 172.819, 99.189], "box_size": [11.69, 7.09, 7.60]}

# Initialize Vina instance (using 8 threads for docking)
vina_instance = Vina(sf_name='vina', cpu=8)

# Initialize Ringtail database
db = rtc.RingtailCore(db_file = "Output/output.db", docking_mode = "vina")

# Function to handle batching of ligands
def process_batches():
    # Process ligands in batches
    batch_start_idx = 0
    total_ligands = len(suppl)

    while batch_start_idx < total_ligands:
        # Get the next batch of ligands
        batch_end_idx = min(batch_start_idx + batch_size, total_ligands)
        #batch_ligands = suppl[batch_start_idx:batch_end_idx]
        batch_ligands = suppl

        # Convert ligands in the batch using Meeko (parallelized)
        converted_batch = Parallel(n_jobs=n_jobs_meeko)(  # Use all available CPU cores
            delayed(convert_to_pdbqt_string)(idx, mol)
            for idx, mol in enumerate(batch_ligands)
            if mol is not None
        )

        # Filter successful conversions
        #valid_ligands = [(mol_idx, pdbqt_string) for mol_idx, pdbqt_string in converted_batch]
        valid_ligands = converted_batch

        # Perform docking for the valid ligands in the batch
        vina_results = dock_ligands_with_vina(vina_instance, receptor_file, docking_box, valid_ligands)

        # Add the results to Ringtail
        #vina_output = "\n".join(vina_results)
        
        # Add results to the Ringtail database using add_results_from_vina_string which takes a dictionary of docking results from Vina as an input.
        db.add_results_from_vina_string(vina_results)

        db.filter(eworst = 1.5, bookmark_name = "bookmark1")
        db.write_molecule_sdfs(sdf_path="Output/sdf_files", write_nonpassing = True, bookmark_name = "bookmark1") 

        # Move to the next batch
        batch_start_idx = batch_end_idx
    
    db.finalize_write()

# Process all batches and add to Ringtail database
process_batches()

print("Docking results successfully added to the Ringtail database.")

#TODO: Fix batching.
#NOTE: The Ringtail command line option to write sdf files is rt_process_vs read --input_db output.db --bookmark_name bookmark1 --export_sdf_path sdf_files/
#NOTE: Maybe it's better to process the output back to .pdb (or .pdbqt) and write the files as that format? Or rather, fetch those entries from the DB and then write them as .pdb files.
