#!/bin/bash

"""Example of the batch file output format:
ad4_receptor_input/9F6A_prepared_AD4.maps.fld
ligands/000A-0084.pdbqt
000A-0084
ligands/000A-0138.pdbqt
000A-0138
ligands/000A-0221.pdbqt
000A-0221
ligands/000A-0480.pdbqt
000A-0480
ligands/000A-0676.pdbqt
000A-0676
ligands/000A-1031.pdbqt
000A-1031
ligands/0085-0007.pdbqt
0085-0007
ligands/0162-0041.pdbqt
0162-0041
ligands/0180-0433.pdbqt
0180-0433
"""

# Define the receptor file
RECEPTOR="9f5s_prep.maps.fld"

# Define the ligands folder
LIGAND_FOLDER="ligands2/"

# Define the output batch file
OUTPUT_BATCH_FILE="ligand_batch2.txt"

# Create or empty the output batch file
> $OUTPUT_BATCH_FILE

# Write the receptor entry once at the top of the batch file
echo "./$RECEPTOR" >> $OUTPUT_BATCH_FILE

# Loop through each ligand in the folder and append the formatted lines to the batch file
for LIGAND in $LIGAND_FOLDER*.pdbqt; do
    # Get the ligand name (without the path and extension)
    LIGAND_NAME=$(basename "$LIGAND" .pdbqt)

    # Append the ligand and its name to the batch file
    echo "./$LIGAND" >> $OUTPUT_BATCH_FILE
    echo "$LIGAND_NAME" >> $OUTPUT_BATCH_FILE
done

echo "Batch file created: $OUTPUT_BATCH_FILE"
