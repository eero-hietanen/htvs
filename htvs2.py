import ringtail as rtc
from joblib import Parallel, delayed
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina
import os
from scrubber import Scrub
from rdkit.Chem.MolStandardize import rdMolStandardize # Consider using rdMolStandardize to standardize the SMILES strings as it could affect the Scrubber step.
import time

# START: Script params.

batch_size = 5  # Number of ligands per batch. Processing is done in batches as the ligand strings are held in memory during processing.
n_cores_meeko = 12  # Core count for Meeko multiprocessing using joblib.
n_cores_vina = 2 # Core (process) count for Vina multiprocessing.
n_vina_threads = 8 # Number of threads to use per Vina instance (Vina's 'cpu' setting").
vina_scoring_function = "vina" # Scoring function to use with Vina.

# Data input directory holding ligand library and receptor.
input_directory = "Data/"

# Output directory for the Ringtail database.
output_directory = "Output/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Ligand library file in .smi format.
smi_file = input_directory + "ligands_10_cxtest.cxsmiles"

# Receptor file in .pdbqt format.
# The receptor should be previously prepared with Meeko 'mk_prepare_receptor.py' (doesn't seem to be a Python API available, but could be done by calling it as a subprocess).
receptor_file = input_directory + "9F6A_prepared.pdbqt"
save_receptor_to_db = False

# Docking box params for Vina.
docking_box = {"center": [136.733, 172.819, 99.189], "box_size": [11.69, 7.09, 7.60]}

# Initialize the Ringtail database.
db = rtc.RingtailCore(db_file = output_directory + "output.db", docking_mode = "vina")

# END: Script params.

##############################################

# Function to convert a ligand to .pdbqt format using Meeko on the fly.
# Uses rdkit to add explicit hydrogens and 3D coordinates for the ligand.
def molecule_prep(smiles, mol_name):

    # Set up Scrubber parameters. Variants are genereated for each ligand. The variant names are appended with an index number and all of them are directed to Vina for docking.
    scrub = Scrub(
        ph_low = 6.9,
        ph_high = 7.9,
    )

    try:
        # Reconstruct the RDKit molecule from the SMILES string.
        mol = Chem.MolFromSmiles(rdMolStandardize.StandardizeSmiles(smiles)) # rdMolStandardize.StandardizeSmiles() used to catch non-standard SMILES strings.
        if mol is None:
            print(f"Warning: Invalid SMILES string for {mol_name}: {smiles}")
            return []

        # Assign the molecule name to the RDKit object.
        mol.SetProp('_Name', mol_name)

        # Prepare the ligand with Meeko.
        preparator = MoleculePreparation()

        variants = []

        # Wrap the scrub(mol) in a try-except block
        try:
            # Scrubber handles protonation states, 3D coordinates, and tautomers. Will raise valance and kekulization exceptions quite often.
            for mol_index, mol_state in enumerate(scrub(mol)):
                variant_mol_name = f"{mol_name}-{mol_index}" # TODO: See how to get the ligand variant docking to work without changing the name for each variant. Then see how Ringtail handles the duplicates.
                # variant_mol_name = f"{mol_name}" # If this is used then the docking result dictionary building breaks due to dictionary key collision.

                # fragments = Chem.GetMolFrags(mol_state, asMols=True)
                # if len(fragments) > 1:
                #     mol_state = max(fragments, key=lambda m: m.GetNumAtoms())

                chooser = rdMolStandardize.LargestFragmentChooser() # In case of fragmented ligands, choose the largest fragment.
                mol_state = chooser.choose(mol_state)

                mol_setups = preparator.prepare(mol_state)

                for setup in mol_setups:
                    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
                    if is_ok:
                        modified_pdbqt = f"REMARK Name = {variant_mol_name.strip()}\n{pdbqt_string}"
                        variants.append((variant_mol_name, modified_pdbqt))
        except RuntimeError as e:
            print(f"Warning: Failed to process {mol_name} with Scrubber: {str(e)}")
            # Continue to the next molecule by returning an empty list
            return []

        return variants

    except Exception as e:
        print(f"Error processing molecule '{mol_name}': {str(e)}")
        return []

# Docking function for a single ligand using Vina
def dock_single_ligand(receptor_file, docking_box, ligand):
    idx, pdbqt_string = ligand
    try:
        v = Vina(sf_name = vina_scoring_function, cpu = n_vina_threads, verbosity = 0)
        v.set_receptor(receptor_file)
        v.set_ligand_from_string(pdbqt_string)
        v.compute_vina_maps(**docking_box)
        v.dock(exhaustiveness=8, n_poses=5)
        vina_poses = v.poses()
        return idx, vina_poses
    except Exception as e:
        print(f"Error docking ligand {idx}: {e}")
        return idx, None

# Docking function using Vina with joblib parallelization
def dock_ligands_with_vina(receptor_file, docking_box, ligands):
    # Use joblib to parallelize the docking
    results = Parallel(n_jobs=n_cores_vina)(
        delayed(dock_single_ligand)(receptor_file, docking_box, ligand)
        for ligand in ligands
    )

    # Convert results to dictionary. Ligand variant names are appended by a number for now.
    """TODO: Building of this results dict is broken due to a key collision issue with same ligand names that are used as the dictionary key.
    At the same time, Ringtail needs the dictionary to be in this format for result parsing.
    """
    vina_output = {}
    for idx, poses in results:
        if poses is not None:
            vina_output[idx] = poses
    return vina_output

# Function to handle batching of ligands.
def process_batches(ligand_input_file, batch_size):
    start_time = time.time()
    batch = []
    file_extension = os.path.splitext(ligand_input_file)[1].lower()

    # Count total lines to calculate total batches
    total_lines = sum(1 for _ in open(ligand_input_file))
    if file_extension == '.cxsmiles':
        total_lines -= 1  # Adjust for header
    total_batches = (total_lines + batch_size - 1) // batch_size
    current_batch = 0
    total_processed = 0

    print(f"\nProcessing {total_lines} ligands in {total_batches} batches...")

    with open(ligand_input_file, "r") as f:
        if file_extension == '.cxsmiles':
            next(f)

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:  # Ensure the line has both SMILES and name.
                smiles, mol_name = line, parts[1]
                batch.append((smiles, mol_name))  # Append a tuple of SMILES and name.

            # When the batch reaches the specified size, process it.
            if len(batch) == batch_size:
                batch_start_time = time.time()
                current_batch += 1
                total_processed += len(batch)
                print(f"\nProcessing batch {current_batch}/{total_batches} ({total_processed}/{total_lines} ligands processed)")

                # Time the preparation step
                prep_start = time.time()
                converted_batch = Parallel(n_jobs=n_cores_meeko)(delayed(molecule_prep)(smiles, mol_name) for smiles, mol_name in batch)
                flattened_batch = [variant for molecule_variants in converted_batch for variant in molecule_variants]
                prep_time = time.time() - prep_start
                print(f"Generated {len(flattened_batch)} variants for {len(batch)} ligands in batch {current_batch} ({prep_time:.2f}s)")

                # Time the docking step
                dock_start = time.time()
                vina_results = dock_ligands_with_vina(receptor_file, docking_box, flattened_batch)
                dock_time = time.time() - dock_start
                print(f"Successfully docked {len(vina_results)} variants in batch {current_batch} ({dock_time:.2f}s)")

                batch_time = time.time() - batch_start_time
                avg_time_per_ligand = batch_time / len(batch)
                print(f"Batch {current_batch} completed in {batch_time:.2f}s (avg {avg_time_per_ligand:.2f}s per ligand)")

                db.add_results_from_vina_string(vina_results, finalize = False)
                batch.clear()
                converted_batch.clear()

    # Process the leftover ligands in the last batch.
    if batch:
        batch_start_time = time.time()
        current_batch += 1
        total_processed += len(batch)
        print(f"\nProcessing final batch {current_batch}/{total_batches} ({total_processed}/{total_lines} ligands processed)")

        prep_start = time.time()
        converted_batch = Parallel(n_jobs=n_cores_meeko)(
            delayed(molecule_prep)(smiles, mol_name) for smiles, mol_name in batch
        )
        flattened_batch = [variant for molecule_variants in converted_batch for variant in molecule_variants]
        prep_time = time.time() - prep_start
        print(f"Generated {len(flattened_batch)} variants for {len(batch)} ligands in final batch ({prep_time:.2f}s)")

        dock_start = time.time()
        vina_results = dock_ligands_with_vina(receptor_file, docking_box, flattened_batch)
        dock_time = time.time() - dock_start
        print(f"Successfully docked {len(vina_results)} variants in final batch ({dock_time:.2f}s)")

        batch_time = time.time() - batch_start_time
        avg_time_per_ligand = batch_time / len(batch)
        print(f"Final batch completed in {batch_time:.2f}s (avg {avg_time_per_ligand:.2f}s per ligand)")
        db.add_results_from_vina_string(vina_results, finalize = False) # Add the "add_interactions" param. here?
        batch.clear()

    # Save the receptor to the Ringtail database.
    if save_receptor_to_db:
        db.save_receptor(receptor_file = receptor_file) # This will cause an issue if the receptor is already saved to the db from a previous script execution.
    print("\nReceptor successfully saved to the database.")

    db.finalize_write()

    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    print(f"Average time per ligand: {total_time/total_lines:.2f}s")

# Process all batches and add docking results to Ringtail database.
process_batches(smi_file, batch_size)

# Make a filter for the top 5% of docking results.
db.filter(score_percentile = 5, bookmark_name = "top_5p_results", order_results = "e")
print("\nDocking results successfully added to the Ringtail database.")

#NOTE: The Ringtail command line option to write sdf files is rt_process_vs read --input_db output.db --bookmark_name bookmark1 --export_sdf_path sdf_files/
#TODO: Implement a receptor preparation method. See about also exporting the specified box .pdb file.
