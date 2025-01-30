import subprocess

# Example ligand PDBQT content
ligand_pdbqt_content = """
REMARK Name = s_527____153905____156108-0
REMARK SMILES CC(C)NC(=O)c1ccco1
REMARK SMILES IDX 4 1 5 2 6 3 2 5 1 6 3 7 7 8 8 9 11 10 9 11 10 12
REMARK H PARENT 4 4
ROOT
ATOM      1  N   UNL     1      -0.509   0.334   0.026  1.00  0.00    -0.347 N 
ATOM      2  C   UNL     1       0.212  -0.614  -0.673  1.00  0.00     0.287 C 
ATOM      3  O   UNL     1      -0.321  -1.454  -1.393  1.00  0.00    -0.266 OA
ATOM      4  H   UNL     1       0.019   0.896   0.685  1.00  0.00     0.164 HD
ENDROOT
BRANCH   1   5
ATOM      5  C   UNL     1      -1.959   0.327   0.089  1.00  0.00     0.076 C 
ATOM      6  C   UNL     1      -2.477   1.730   0.385  1.00  0.00     0.030 C 
ATOM      7  C   UNL     1      -2.447  -0.664   1.143  1.00  0.00     0.030 C 
ENDBRANCH   1   5
BRANCH   2   8
ATOM      8  C   UNL     1       1.643  -0.529  -0.468  1.00  0.00     0.191 A 
ATOM      9  C   UNL     1       2.682  -1.274  -0.993  1.00  0.00     0.056 A 
ATOM     10  O   UNL     1       2.150   0.422   0.382  1.00  0.00    -0.459 OA
ATOM     11  C   UNL     1       3.878  -0.755  -0.442  1.00  0.00     0.043 A 
ATOM     12  C   UNL     1       3.500   0.275   0.387  1.00  0.00     0.196 A 
ENDBRANCH   2   8
TORSDOF 2
"""

# Run the separate Python script and pass ligand data through stdin
proc = subprocess.Popen(
    ["python3", "vina_dock.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Send the ligand content and retrieve the output
stdout, stderr = proc.communicate(input=ligand_pdbqt_content)

# Print docking results
print(stdout)

# Print errors if any
if stderr:
    print("Error:", stderr)
