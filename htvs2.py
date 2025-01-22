import ringtail as rtc
from joblib import Parallel, delayed
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina
from openbabel import openbabel, pybel
import os

# NOTE: Mostly works but the database input is missing SMILES strings, so when you try to export from the db Ringtail throws an error.
# TODO: Take the mol prep wrapper from this and implemenet it to the original htvs.py. 

# Script params
batch_size = 5
n_cores_meeko = -1
n_cores_vina = 18
vina_scoring_function = "vina"

input_directory = "Data/"
if not os.path.exists(input_directory):
    os.makedirs(input_directory)

output_directory = "Output/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

sdf_file = input_directory + "ligands_10.sdf"
smi_file = input_directory + "ligands_10_2.smi"
receptor_file = input_directory + "9f6a.pdbqt"
save_receptor_to_db = True

docking_box = {"center": [136.733, 172.819, 99.189], "box_size": [11.69, 7.09, 7.60]}

vina_instance = Vina(sf_name=vina_scoring_function, cpu=n_cores_vina)
db = rtc.RingtailCore(db_file=output_directory + "output.db", docking_mode="vina")

def molecule_prep(idx, mol):
    try:
        mol = Chem.AddHs(mol)
        Chem.rdDistGeom.EmbedMolecule(mol)
        
        # Get SMILES from RDKit molecule
        smiles = Chem.MolToSmiles(mol)
        
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        for setup in mol_setups:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                # Add SMILES as a remark
                pdbqt_with_smiles = f"REMARK     SMILES: {smiles}\n" + pdbqt_string
                return idx, pdbqt_with_smiles
        return idx, pdbqt_with_smiles, None
    except Exception as e:
        return idx, None, str(e)

def molecule_prep2_wrapper(idx, smiles, title=None):
    try:
        mol = pybel.readstring("smi", smiles)
        if title:
            mol.title = title

        obmol = mol.OBMol
        obmol.AddHydrogens(False, True, 7.4)
        mol = pybel.Molecule(obmol)
        mol.make3D()
        
        # Add SMILES as a remark in the PDBQT output
        pdbqt_string = mol.write(format="pdbqt")
        pdbqt_with_smiles = f"REMARK     SMILES: {smiles}\n" + pdbqt_string
        
        return idx, pdbqt_with_smiles
    except Exception as e:
        return idx, None, str(e)

def dock_ligands_with_vina(vina_instance, receptor_file, docking_box, ligands):
    vina_instance.set_receptor(receptor_file)
    vina_instance.compute_vina_maps(**docking_box)

    vina_output = {}
    for idx, pdbqt_string in ligands:
        try:
            vina_instance.set_ligand_from_string(pdbqt_string)
            vina_instance.dock(exhaustiveness=32, n_poses=5)
            vina_poses = vina_instance.poses()
            vina_output[pdbqt_string] = vina_poses
        except Exception as e:
            print(f"Error docking ligand {idx}: {e}")

    return vina_output

def process_batches(input_file, batch_size, input_format="sdf"):
    batch = []
    batch_idx = []

    if input_format == "sdf":
        suppl = Chem.SDMolSupplier(input_file)
        for idx, mol in enumerate(suppl):
            if mol is not None:
                batch.append(mol)
                batch_idx.append(idx)

            if len(batch) == batch_size:
                converted_batch = Parallel(n_jobs=n_cores_meeko)(
                    delayed(molecule_prep)(idx, mol)
                    for idx, mol in zip(batch_idx, batch)
                    if mol is not None
                )
                process_batch(converted_batch)
                batch.clear()
                batch_idx.clear()

    else:
        for idx, mol in enumerate(pybel.readfile("smi", input_file)):
            if mol is not None:
                batch.append((mol.write("smi").strip(), mol.title))
                batch_idx.append(idx)

            if len(batch) == batch_size:
                converted_batch = Parallel(n_jobs=n_cores_meeko)(
                    delayed(molecule_prep2_wrapper)(idx, smiles, title)
                    for idx, (smiles, title) in zip(batch_idx, batch)
                )
                process_batch(converted_batch)
                batch.clear()
                batch_idx.clear()

    if batch:
        if input_format == "sdf":
            converted_batch = Parallel(n_jobs=n_cores_meeko)(
                delayed(molecule_prep)(idx, mol)
                for idx, mol in zip(batch_idx, batch)
                if mol is not None
            )
        else:
            converted_batch = Parallel(n_jobs=n_cores_meeko)(
                delayed(molecule_prep2_wrapper)(idx, smiles, title)
                for idx, (smiles, title) in zip(batch_idx, batch)
            )
        process_batch(converted_batch)

def process_batch(converted_batch):
    vina_results = dock_ligands_with_vina(vina_instance, receptor_file, docking_box, converted_batch)
    db.add_results_from_vina_string(vina_results, finalize=False)

if __name__ == "__main__":
    input_format = "smi" if smi_file.endswith(".smi") else "sdf"
    input_file = smi_file if input_format == "smi" else sdf_file
    
    process_batches(input_file, batch_size, input_format)
    
    if save_receptor_to_db:
        db.save_receptor(receptor_file=receptor_file) # This will cause an issue if the receptor is already saved to the db from a previous script execution.
    print("\nReceptor successfully saved to the database.")
    
    db.finalize_write()
    print("\nDocking results successfully added to the Ringtail database.")