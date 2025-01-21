import ringtail as rtc
from joblib import Parallel, delayed
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina
import os

# START: Script params.

batch_size = 5  # Number of ligands per batch. Processing is done in batches as the ligand strings are held in memory during processing.
n_cores_meeko = -1  # Core count for Meeko multiprocessing using joblib.
n_cores_vina = 18 # Core count for Vina multiprocessing.
vina_scoring_function = "vina" # Scoring function to use with Vina.

# Data input directory holding ligand library and receptor.
input_directory = "Data/"
if not os.path.exists(input_directory):
    os.makedirs(input_directory)

# Output directory for the Ringtail database.
output_directory = "Output/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Ligand library file in .sdf format.
sdf_file = input_directory + "ligands_10.sdf"

# Receptor file in .pdbqt format.
# The receptor should be previously prepared with Meeko 'mk_prepare_receptor.py' (doesn't seem to be a Python API available, but could be done by calling it as a subprocess).
receptor_file = input_directory + "9f6a.pdbqt"
save_receptor_to_db = True

# Docking box params for Vina.
docking_box = {"center": [136.733, 172.819, 99.189], "box_size": [11.69, 7.09, 7.60]}

# Initialize a Vina instance.
vina_instance = Vina(sf_name = vina_scoring_function, cpu = n_cores_vina)

# Initialize the Ringtail database.
db = rtc.RingtailCore(db_file = output_directory + "output.db", docking_mode = "vina")

# END: Script params.

##############################################

# Function to convert a ligand to .pdbqt format using Meeko on the fly.
# Uses rdkit to add explicit hydrogens and 3D coordinates for the ligand.
def molecule_prep(idx, mol):
    try:
        # Add explicit hydrogens and embed 3D coordinates.
        # NOTE: Maybe this preparation step needs an energy minimization step?
        mol = Chem.AddHs(mol)
        Chem.rdDistGeom.EmbedMolecule(mol)

        # Prepare the ligand with Meeko.
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        for setup in mol_setups:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                return idx, pdbqt_string
        return idx, pdbqt_string, None  # No error.
    except Exception as e:
        return idx, None, str(e)  # Return error.

# Docking function using Vina.
# Uses Vina's own multiprocessing.
def dock_ligands_with_vina(vina_instance, receptor_file, docking_box, ligands):
    vina_instance.set_receptor(receptor_file)
    vina_instance.compute_vina_maps(**docking_box)

    # Initialize an empty dictionary to store the results. This will be the input for Ringtail.
    vina_output = {}

    # Store results as strings.
    for idx, pdbqt_string in ligands:
        try:
            vina_instance.set_ligand_from_string(pdbqt_string)
            vina_instance.dock(exhaustiveness=32, n_poses=5) # NOTE: Ringtail saves the top 3 poses per ligand by default.
            vina_poses = vina_instance.poses()

            # Build a results dictionary from the Vina docking results that can be used as an input for Ringtail database.
            # The result dictionary key should be the pdbqt_string of the liganda, and the value should be the vina_poses.
            vina_output[pdbqt_string] = vina_poses
  
        except Exception as e:
            print(f"Error docking ligand {idx}: {e}")

    return vina_output

# Function to handle batching of ligands.
def process_batches(sdf_file, batch_size):

    # Initialize the moleculer supplier.
    suppl = Chem.SDMolSupplier(sdf_file) # SDMolSupplier is an iterator. There's also an experimental 'MultithreadedSDMolSupplier' that may be faster.

    batch = []

    for idx, mol in enumerate(suppl):
        if mol is not None:
            batch.append(mol)

        if len(batch) == batch_size:

            # Convert the ligands from the .sdf file to .pdbqt strings using Meeko.
            converted_batch = Parallel(n_jobs=n_cores_meeko)(
                delayed(molecule_prep)(idx, mol) # Delayed call for molecule_prep() once below evaluation to done.
                for idx, mol in enumerate(batch)
                if mol is not None
            )

            # Dock the converted batch of ligands using Vina.
            vina_results = dock_ligands_with_vina(vina_instance, receptor_file, docking_box, converted_batch)
            db.add_results_from_vina_string(vina_results, finalize = False) # Add docking results to the Ringtail database.
            batch.clear() # Clear the batch in preparation for the next batch.
            converted_batch.clear() # Clear the converted batch in preparation for the next batch.

    # Process the leftover ligands in the last batch.
    if batch:
        converted_batch = Parallel(n_jobs=n_cores_meeko)(
                delayed(molecule_prep)(idx, mol) # Delayed call for molecule_prep() once below evaluation to done.
                for idx, mol in enumerate(batch)
                if mol is not None
        )
        vina_results = dock_ligands_with_vina(vina_instance, receptor_file, docking_box, converted_batch)
        db.add_results_from_vina_string(vina_results, finalize = False)
        batch.clear()
    
    # Save the receptor to the Ringtail database.
    if save_receptor_to_db:
        db.save_receptor(receptor_file = receptor_file)
    print("\nReceptor successfully saved to the database.")
    
    db.finalize_write()

# Process all batches and add docking results to Ringtail database.
process_batches(sdf_file, batch_size)
print("\nDocking results successfully added to the Ringtail database.")

#NOTE: The Ringtail command line option to write sdf files is rt_process_vs read --input_db output.db --bookmark_name bookmark1 --export_sdf_path sdf_files/
#NOTE: The output structures had some formatting issues when tested the last time, so check the file format and maybe try Open Babel conversion.
