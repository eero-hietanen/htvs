import ringtail as rtc
from joblib import Parallel, delayed
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina
from openbabel import openbabel, pybel
import os
from re import split, sub, MULTILINE

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

# Ligand library file in .smi format.
smi_file = input_directory + "ligands_10.smi"
smi_file_no_header = input_directory + "ligands_10_2.smi"

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

# Helper function for format conversion.
def smiles_to_obmol(smiles_string):
    # Create an OpenBabel conversion object
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "mol")
    
    # Create an OBMol object
    obmol = openbabel.OBMol()
    
    # Convert SMILES to OBMol
    obConversion.ReadString(obmol, smiles_string)
    
    return obmol

# Function to convert a ligand to .pdbqt format using Meeko on the fly.
# Uses rdkit to add explicit hydrogens and 3D coordinates for the ligand.
def molecule_prep(smiles, mol_name):

    # Extract the SMILES string for the molecule.
    # TODO: Extract the molecule name / SMILES here and inject it into the PDBQT string.
    # Input mol here is rdkit mol object from either .smi or .sdf.

    try:
        # Reconstruct the RDKit molecule from the SMILES string.
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Assign the molecule name to the RDKit object.
        mol.SetProp('_Name', mol_name)

        # Add explicit hydrogens and embed 3D coordinates.
        mol = Chem.AddHs(mol)
        Chem.rdDistGeom.EmbedMolecule(mol)

        # Prepare the ligand with Meeko.
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)

        for setup in mol_setups:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                # Include the molecule name in the PDBQT string.
                modified_pdbqt = f"REMARK Name = {mol_name.strip()}\n{pdbqt_string}"
                return mol_name, modified_pdbqt

        # If no PDBQT string could be generated, raise an error.
        raise ValueError("Failed to generate PDBQT string.")
    except Exception as e:
        raise RuntimeError(f"Error processing molecule '{mol_name}': {str(e)}")
    
# Alternate molecule preparation function using Open Babel to account for protonation states.
# Input is fixed to be the ligand library in .smi format.
# Input molecules are in pybel molecule format.
def molecule_prep2(idx, mol):
    try:

        # Fetch the SMILES string and molecule name.
        # smiles_string = split("\t", mol.write(format = "smi"))[0]
        # mol_name = split("\t", mol.write(format = "smi"))[1]
        smiles_string = split("\t", mol)[0] # The mol object here is just a string as we do conversion during the batching.
        mol_name = split("\t", mol)[1]

        # Convert the pybel molecule to Open Babel molecule.
        # obmol = mol.OBMol
        obmol = smiles_to_obmol(mol)

        # Add hydrogens and consider protonation states.
        obmol.AddHydrogens(False, True, 7.4)

        # Convert the Open Babel molecule back to pybel molecule.
        mol = pybel.Molecule(obmol)

        mol.make3D()

        output_string = mol.write(format = "pdbqt")
        # modified_pdbqt = f"REMARK  Name = {mol_name.strip()}\nREMARK  SMILES = {smiles_string.strip()}\n{pybel_string}"
        # modified_pdbqt = f"REMARK SMILES {smiles_string.strip()}\n{output_string}"
        # modified_pdbqt = sub(r'^REMARK\s+Name\s*=.*$', f'REMARK Name = {mol_name.strip()}', modified_pdbqt, flags=MULTILINE) # Replaces the Name remark with the molecule name.
        
        modified_pdbqt = sub(r'^REMARK\s+Name\s*=.*\n', '', output_string, flags=MULTILINE)
        modified_pdbqt = f"REMARK Name = {mol_name.strip()}\nREMARK SMILES {smiles_string.strip()}\n{modified_pdbqt}"
        


        print(modified_pdbqt)

        return idx, modified_pdbqt
    except Exception as e:
        return idx, None, str(e)
    
# Testing of modified molecule_prep2 function to include protonation states.
def molecule_prep3(smiles, mol_name):
    try:      
        
        smiles_string = smiles.strip().split("\t")[0]

        # Convert the pybel molecule to Open Babel molecule.
        # obmol = mol.OBMol
        obmol = smiles_to_obmol(smiles)

        # Add hydrogens and consider protonation states.
        obmol.AddHydrogens(False, True, 7.4)

        # Convert the Open Babel molecule back to pybel molecule.
        mol = pybel.Molecule(obmol)

        mol.make3D()

        output_string = mol.write(format = "pdbqt")
        
        modified_pdbqt = sub(r'^REMARK\s+Name\s*=.*\n', '', output_string, flags=MULTILINE)
        modified_pdbqt = f"REMARK Name = {mol_name.strip()}\nREMARK SMILES {smiles_string.strip()}\n{modified_pdbqt}"
        
        return mol_name, modified_pdbqt
    except Exception as e:
        return mol_name, None, str(e)

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
            vina_output[idx] = vina_poses # The dictionary key should be the molecule name instead of the whole string.

            # vina_instance.write_poses('docked.pdbqt', n_poses=5, overwrite=True) # Testing of docking output.
            # print(vina_poses) # Testing docking output for correct file format. Checked with SMILEs input: OK.
         
        except Exception as e:
            print(f"Error docking ligand {idx}: {e}")

    return vina_output

# Function to handle batching of ligands.
def process_batches(ligand_input_file, batch_size):

    # Initialize the moleculer supplier. NOTE: None of these suppliers are needed if we batch using the input .smi file and direct the batches to joblib.
    suppl = Chem.SDMolSupplier(ligand_input_file) # SDMolSupplier is an iterator. There's also an experimental 'MultithreadedSDMolSupplier' that may be faster.
    suppl2 = Chem.SmilesMolSupplier(smi_file, delimiter="\t") # Iterator for the .smi ligand file.
    # suppl2 = Chem.SmilesMolSupplierFromText(smi_file, delimiter="\t") # This iterator might work better than the above one.
    # NOTE: Suspicion is that the pybel.readfile() iterator causes the pickling issue with joblib.
    suppl4 = pybel.readfile("smi", smi_file_no_header) # Test pybel iterator for the ligand batching.

    batch = []

    # Read the SMILES file and extract SMILES strings and names.
    # TODO: Check if adding the remaining cxsmiles fields should be done here.
    with open(smi_file_no_header, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:  # Ensure the line has both SMILES and name.
                smiles, mol_name = line, parts[1]
                batch.append((smiles, mol_name))  # Append a tuple of SMILES and name.

            # When the batch reaches the specified size, process it.
            if len(batch) == batch_size:
                converted_batch = Parallel(n_jobs=n_cores_meeko)(
                    delayed(molecule_prep3)(smiles, mol_name) for smiles, mol_name in batch
                )

                # Dock the converted batch of ligands using Vina.
                vina_results = dock_ligands_with_vina(vina_instance, receptor_file, docking_box, converted_batch)
                db.add_results_from_vina_string(vina_results, finalize = False) # Add docking results to the Ringtail database.
                batch.clear() # Clear the batch in preparation for the next batch.
                converted_batch.clear() # Clear the converted batch in preparation for the next batch.

    # Process the leftover ligands in the last batch.
    if batch:
        converted_batch = Parallel(n_jobs=n_cores_meeko)(
            delayed(molecule_prep3)(smiles, mol_name) for smiles, mol_name in batch
        )
        vina_results = dock_ligands_with_vina(vina_instance, receptor_file, docking_box, converted_batch)
        db.add_results_from_vina_string(vina_results, finalize = False) # Add the "add_interactions" param. here.
        batch.clear()
    
    # Save the receptor to the Ringtail database.
    if save_receptor_to_db:
        db.save_receptor(receptor_file = receptor_file) # This will cause an issue if the receptor is already saved to the db from a previous script execution.
    print("\nReceptor successfully saved to the database.")
    
    db.finalize_write()

# Process all batches and add docking results to Ringtail database.
process_batches(smi_file, batch_size)
db.filter(score_percentile = 50, bookmark_name = "bm1")
db.write_molecule_sdfs(bookmark_name = "bm1")
print("\nDocking results successfully added to the Ringtail database.")

#NOTE: The Ringtail command line option to write sdf files is rt_process_vs read --input_db output.db --bookmark_name bookmark1 --export_sdf_path sdf_files/
#NOTE: The output structures had some formatting issues when tested the last time, so check the file format and maybe try Open Babel conversion.
