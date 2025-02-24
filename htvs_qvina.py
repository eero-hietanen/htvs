# Standard library imports
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import List, Tuple, Dict, Any
import multiprocessing

# Third party imports
from joblib import Parallel, delayed
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from meeko import MoleculePreparation, PDBQTWriterLegacy

# Local/library specific imports
from scrubber import Scrub
import ringtail as rtc

# START: Script params.

BATCH_SIZE = 5000  # Number of ligands per batch. The batch is split between GPUs.
START_BATCH = 0 # 0 indexed
RESUME_FROM_LINE = START_BATCH * BATCH_SIZE
N_CORES_MEEKO = 20  # Core count for Meeko multiprocessing using joblib.
N_PROCESSES_QVINA = 3 # Number of VinaGPU processes to run in parallel per GPU, with each one handling a batch. NOTE: Single NVIDIA A40 GPU seems to have enough memory to handle 3 batches at once (Enamine 9.6M library, 5000 batch size).
QVINA_GPU_THREADS = 5000 # Ideally less than 10000 as per the documentation. Suggested for QuickVina2 is 5000.

# Fetch the number of NVIDIA GPUs on the system.
cmd = f"nvidia-smi -L | wc -l"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = proc.communicate(timeout = 10)
assert proc.returncode == 0, f"Error: {stderr}"
NUM_GPUS = int(stdout.decode("utf-8").strip())
NUM_GPUS_USED = NUM_GPUS # By default uses all GPUs on the system by setting NUM_GPUS_USED = NUM_GPUS.

# Path to QuickVina2-GPU binary and the OpenCL binaries.
QVINA_EXECUTABLE = "/path/to/Vina-GPU-2.1/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1"
OPENCL_BINARY_PATH = "/path/to/Vina-GPU-2.1/QuickVina2-GPU-2.1"

# Data input directory holding ligand library and receptor.
INPUT_DIR = "./"

# Output directory for the Ringtail database.
OUTPUT_DIR = "htvs_qvina_output/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Ligand library file in .smi format.
SMI_FILE = os.path.join(INPUT_DIR, "2024.07_Enamine_REAL_DB_9.6M.cxsmiles")

# Receptor file in .pdbqt format.
# The receptor should be previously prepared with Meeko 'mk_prepare_receptor.py' or otherwise.
RECEPTOR_FILE = os.path.join(INPUT_DIR, "9F6A_prepared.pdbqt")
#RECEPTOR_PDB = "9F6A" # Fetch receptor from RCSB PDB with curl and prepare with Meeko using receptor_prep().
SAVE_RECEPTOR = True

# Docking box parameters for Vina.
DOCKING_BOX = {"center": [136.733, 172.819, 99.189], "box_size": [11.69, 7.09, 7.60]}

# Initialize the Ringtail database.
DB = rtc.RingtailCore(db_file = OUTPUT_DIR + "output.db", docking_mode = "vina")

# END: Script params.

##############################################

# Split the batch into GPU batches based on the number of GPUs used. Two batches are run per GPU.
def generate_gpu_args(batch_input, RECEPTOR_FILE, DOCKING_BOX, current_batch, NUM_GPUS_USED):
    # Split the batch into sub-batches for each GPU.
    batches = split_list_into_batches(batch_input, NUM_GPUS_USED*2)

    # Generate the arguments for each GPU.
    gpu_args = []
    for i, batch in enumerate(batches):
        # GPU ID is 0 indexed and it increases every 2 sub-batches.
        gpu_id = i // 2
        gpu_args.append((RECEPTOR_FILE, DOCKING_BOX, batch, f"{current_batch}_{i+1}", gpu_id))

    return gpu_args

# Splits a list into sub-batches that will be distributed to different Vina instances.
def split_list_into_batches(input_list, num_batches):
    list_length = len(input_list)
    
    # Ensure num_batches doesn't exceed list length.
    num_batches = min(num_batches, list_length)
    
    # Calculate base size and remainder.
    base_size = list_length // num_batches
    remainder = list_length % num_batches
    
    batches = []
    start = 0
    
    for i in range(num_batches):
        # Add one extra element to early batches if there's remainder.
        batch_size = base_size + (1 if i < remainder else 0)
        end = start + batch_size
        batches.append(input_list[start:end])
        start = end
    
    return batches

# Cleanup function called with KeyboardInterrupt exceptions and at the end of processing.
def cleanup() -> None:
    print("\nPerforming cleanup...")
    try:
        DB.finalize_write()
        print("Database writing finalized.")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

# Fetch receptor structure from PDB with curl. The base URL for the download is https://files.rcsb.org/download/
# Call Meeko to prepare the receptor and output a .pdbqt file.
def receptor_prep(RECEPTOR_PDB: str) -> str:
    receptor_url = f"https://files.rcsb.org/download/{RECEPTOR_PDB}.pdb"
    receptor_file = os.path.join(INPUT_DIR, f"{RECEPTOR_PDB}.pdb")

    subprocess.run(["curl", "-o", receptor_file, receptor_url])
    subprocess.run(["mk_prepare_receptor.py", "-i", receptor_file, "-p", "-o", f"{RECEPTOR_PDB}_prepared"])
    RECEPTOR_FILE = os.path.join(INPUT_DIR, f"{RECEPTOR_PDB}_prepared.pdbqt")
    return RECEPTOR_FILE

# Function to convert a ligand to .pdbqt format using Meeko on the fly.
# Uses rdkit to add explicit hydrogens and 3D coordinates for the ligand.
def molecule_prep(smiles: str, mol_name: str) -> List[Tuple[str, str]]:

    # Disabling rdkit's error output as the sanitization process often throws errors that can be ignored.
    # The error causing molecules should not be considered.
    rdBase.DisableLog('rdApp.error')
    rdBase.DisableLog('rdApp.warning')

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

        try:
            # Scrubber handles protonation states, 3D coordinates, and tautomers. Will raise valance and kekulization exceptions quite often.
            for mol_index, mol_state in enumerate(scrub(mol)):
                variant_mol_name = f"{mol_name}-{mol_index}"

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
            # Continue to the next molecule by returning an empty list.
            return []

        rdBase.EnableLog('rdApp.error') # Re-enabling error logging.
        rdBase.EnableLog('rdApp.warning')
        return variants

    except Exception as e:
        print(f"Error processing molecule '{mol_name}': {str(e)}")
        rdBase.EnableLog('rdApp.error') # Re-enabling error logging.
        rdBase.EnableLog('rdApp.warnings')
        return []

# Batch directory creation and cleanup.
def create_batch_directory(batch_id: str) -> str:
    batch_dir = tempfile.mkdtemp(prefix=f'batch_{batch_id}_')
    return batch_dir

# Initialize worker with signal handler.
# This is here in hopes of avoiding zombie processes during execution, but it might be useless.
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# Write ligands to batch directory.
def write_ligands_to_directory(ligands: List[Tuple[str, str]], batch_dir: str) -> Dict[str, str]:
    ligand_files = {}
    for idx, pdbqt_string in ligands:
        # Create a filename safe version of the ligand ID.
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', idx)
        file_path = os.path.join(batch_dir, f"{safe_name}.pdbqt")
        with open(file_path, 'w') as f:
            f.write(pdbqt_string)
        ligand_files[file_path] = idx
    return ligand_files

# Modified docking function to handle GPU selection.
def dock_batch_with_qvina(RECEPTOR_FILE: str, DOCKING_BOX: Dict[str, Any], ligands: List[Tuple[str, str]], batch_id: str, gpu_id: int) -> Dict[str, str]:
    batch_dir = None
    try:
        batch_dir = create_batch_directory(f"{batch_id}_gpu{gpu_id}")
        output_dir = os.path.join(batch_dir, 'output')
        os.makedirs(output_dir)

        #print(f"Debug: Created directories - batch: {batch_dir}, output: {output_dir}")
        ligand_files = write_ligands_to_directory(ligands, batch_dir)

        # Add CUDA_VISIBLE_DEVICES prefix to the command to direct batches to different GPUs.
        vina_cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} {QVINA_EXECUTABLE} --receptor {RECEPTOR_FILE} '\
                   f'--ligand_directory {batch_dir} '\
                   f'--thread {QVINA_GPU_THREADS} '\
                   f'--center_x {DOCKING_BOX["center"][0]} '\
                   f'--center_y {DOCKING_BOX["center"][1]} '\
                   f'--center_z {DOCKING_BOX["center"][2]} '\
                   f'--size_x {DOCKING_BOX["box_size"][0]} '\
                   f'--size_y {DOCKING_BOX["box_size"][1]} '\
                   f'--size_z {DOCKING_BOX["box_size"][2]} '\
                   f'--opencl_binary_path {OPENCL_BINARY_PATH} '\
                   f'--output_directory {output_dir}'

        print(f"Debug: Executing command:\n{vina_cmd}")

        process = subprocess.Popen(
            vina_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )

        # Monitor process execution and print QuickVina errors if any.
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Debug: QuickVina stderr:\n{stderr}")
            raise subprocess.CalledProcessError(process.returncode, vina_cmd)

        # Read output files and collect results.
        vina_output = {}
        for output_file in os.listdir(output_dir):
            if output_file.endswith('_out.pdbqt'):
                ligand_name = output_file.replace('_out.pdbqt', '')
                with open(os.path.join(output_dir, output_file)) as f:
                    vina_output[ligand_name] = f.read()
                #print(f"Debug: Read output for {ligand_name}")

        return vina_output

    except Exception as e:
        print(f"Error in docking batch {batch_id}: {str(e)}")
        return {}
    finally:
        if batch_dir and os.path.exists(batch_dir):
            shutil.rmtree(batch_dir)

# Function to handle parallel GPU docking.
def run_parallel_gpu_docking(args: Tuple[str, Dict[str, Any], List[Tuple[str, str]], str, int]) -> Dict[str, str]:
    RECEPTOR_FILE, DOCKING_BOX, batch, batch_id, gpu_id = args
    return dock_batch_with_qvina(RECEPTOR_FILE, DOCKING_BOX, batch, batch_id, gpu_id)

# Function to handle reading the input SMILES file and splitting it into batches.
def process_batches(ligand_input_file: str, BATCH_SIZE: int, start_from_line: int = 0) -> None:
    start_time = time.time()
    lines_skipped = 0
    try:
        batch = []
        file_extension = os.path.splitext(ligand_input_file)[1].lower()

        # Count total lines to calculate total batches.
        total_lines = sum(1 for _ in open(ligand_input_file))
        if file_extension == '.cxsmiles':
            total_lines -= 1  # Adjust for header in .cxsmiles files.

        remaining_lines = total_lines - start_from_line
        total_batches = (total_lines + BATCH_SIZE - 1) // BATCH_SIZE
        remaining_batches = total_batches - (start_from_line // BATCH_SIZE)
        current_batch = start_from_line // BATCH_SIZE
        total_processed = 0

        print(f"\nStarting from batch {current_batch + 1} (line {start_from_line})")
        print(f"Processing {remaining_lines} remaining ligands in {remaining_batches} batches (out of {total_batches} total batches)...")

        with open(ligand_input_file, "r") as f:
            if file_extension == '.cxsmiles':
                next(f)

            # Skip to starting line
            for _ in range(start_from_line):
                next(f)
                lines_skipped += 1

            print(f"Skipped {lines_skipped} lines")

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:  # Ensure the line has both SMILES and ligand name.
                    smiles, mol_name = line, parts[1]
                    batch.append((smiles, mol_name))  # Append a tuple of SMILES and ligand name.

                # When the batch reaches the specified size, process it.
                if len(batch) == BATCH_SIZE:
                    batch_start_time = time.time()
                    current_batch += 1
                    total_processed += len(batch)
                    print(f"\nProcessing batch {current_batch}/{total_batches} ({total_processed}/{total_lines} ligands processed)")

                    # Molecule preparation step using Meeko and joblib parallelization.
                    prep_start = time.time()
                    try:
                        with Parallel(n_jobs=N_CORES_MEEKO) as parallel:
                            converted_batch = parallel(
                                delayed(molecule_prep)(smiles, mol_name)
                                for smiles, mol_name in batch
                            )
                    finally:
                        # Force cleanup of any remaining processes
                        Parallel(n_jobs=1)
                    flattened_batch = [variant for molecule_variants in converted_batch for variant in molecule_variants]
                    prep_time = time.time() - prep_start
                    print(f"Generated {len(flattened_batch)} variants for {len(batch)} ligands in batch {current_batch} ({prep_time:.2f}s)")
                    dock_start = time.time()

                    # Create arguments for parallel processing.
                    gpu_args = generate_gpu_args(flattened_batch, RECEPTOR_FILE, DOCKING_BOX, current_batch, NUM_GPUS_USED)

                    # Run docking in parallel on both GPUs.
                    pool = multiprocessing.Pool(NUM_GPUS_USED*N_PROCESSES_QVINA, init_worker)
                    try:
                        results = pool.map(run_parallel_gpu_docking, gpu_args)
                    finally:
                        pool.close()
                        pool.join()

                    # Merge results from GPUs.
                    vina_results = {}
                    for result in results:
                        vina_results.update(result)

                    if not vina_results:
                        print(f"Warning: No successful docking results in batch {current_batch}, skipping...")
                        batch.clear()
                        converted_batch.clear()
                        continue

                    dock_time = time.time() - dock_start
                    print(f"Successfully docked {len(vina_results)} variants in batch {current_batch} ({dock_time:.2f}s)")

                    batch_time = time.time() - batch_start_time
                    avg_time_per_ligand = batch_time / len(batch)
                    print(f"Batch {current_batch} completed in {batch_time:.2f}s (avg {avg_time_per_ligand:.2f}s per ligand)")

                    DB.add_results_from_vina_string(vina_results, finalize = False)
                    batch.clear()
                    converted_batch.clear()

        # Process the leftover ligands in the last batch.
        # This could be changed so that instead of doing this, check if the remaining ligands are able to
        # fill a batch and if they are not, then add them to the previous batch and process all at once.
        if batch:
            batch_start_time = time.time()
            current_batch += 1
            total_processed += len(batch)
            print(f"\nProcessing final batch {current_batch}/{total_batches} ({total_processed}/{total_lines} ligands processed)")

            prep_start = time.time()
            converted_batch = Parallel(n_jobs=N_CORES_MEEKO)(
                delayed(molecule_prep)(smiles, mol_name) for smiles, mol_name in batch
            )
            flattened_batch = [variant for molecule_variants in converted_batch for variant in molecule_variants]
            prep_time = time.time() - prep_start
            print(f"Generated {len(flattened_batch)} variants for {len(batch)} ligands in final batch ({prep_time:.2f}s)")
            dock_start = time.time()

            # Create arguments for parallel processing.
            gpu_args = generate_gpu_args(flattened_batch, RECEPTOR_FILE, DOCKING_BOX, current_batch, NUM_GPUS_USED)

            with multiprocessing.Pool(NUM_GPUS_USED*N_PROCESSES_QVINA) as pool:
                results = pool.map(run_parallel_gpu_docking, gpu_args)

            # Merge results from GPUs.
            vina_results = {}
            for result in results:
                vina_results.update(result)

            if not vina_results:
                print(f"Warning: No successful docking results in final batch, skipping...")
                batch.clear()
            else:
                dock_time = time.time() - dock_start
                print(f"Successfully docked {len(vina_results)} variants in final batch ({dock_time:.2f}s)")

                batch_time = time.time() - batch_start_time
                avg_time_per_ligand = batch_time / len(batch)
                print(f"Final batch completed in {batch_time:.2f}s (avg {avg_time_per_ligand:.2f}s per ligand)")
                DB.add_results_from_vina_string(vina_results, finalize = False)
                batch.clear()

        # Save the receptor to the Ringtail database.
        if SAVE_RECEPTOR:
            DB.save_receptor(receptor_file = RECEPTOR_FILE)
        print("\nReceptor successfully saved to the database.")

        DB.finalize_write()

        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.2f}s")
        print(f"Average time per ligand: {total_time/total_lines:.2f}s")
    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user.")
        cleanup()
        raise
    except Exception as e:
        print(f"An error occurred during batch processing: {str(e)}")
        cleanup()
        raise
    finally:
        cleanup()

def main() -> None:
    try:
        # Declare the receptor file as a global variable so that it's accessible.
        global RECEPTOR_FILE
        # Fetch and prepare the receptor structure based on the PDB ID.
        # RECEPTOR_FILE = receptor_prep(RECEPTOR_PDB)
        RECEPTOR_FILE = "9F6A_prepared.pdbqt"

        # Process all batches and add docking results to Ringtail database.
        process_batches(SMI_FILE, BATCH_SIZE, start_from_line=RESUME_FROM_LINE)

        # Make a filter for the top 1% of docking results.
        DB.filter(score_percentile=1, bookmark_name="top_1p_results", order_results="e")
        print("\nDocking results successfully added to the Ringtail database.")
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()

#NOTE: The Ringtail command line option to write sdf files is rt_process_vs read --input_db output.db --bookmark_name my_bookmark --export_sdf_path sdf_files/
#TODO: All subprocess use cases should probably be done with shell=False and split commands. Instances that require piping need directing of p1 output to the stdin of p2 to mimic shell piping.