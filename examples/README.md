# Examples

Quick things to try once fpocket is installed (`conda env create -f
../environment.yml`).

A small enzyme by PDB ID, treated as a generic small-molecule inhibitor:

    python ../cli.py --pdb-id 2V5Z --ligand-class small_molecule_inhibitor --top 5

A local file with a ligand given as SMILES (descriptors via RDKit):

    python ../cli.py --pdb-file my_receptor.pdb --smiles "CC(=O)Oc1ccccc1C(=O)O"

Pick PDB IDs relevant to YOUR targets. For the Parkinson's panel you are
working on, swap in your own structures and try the matching `--ligand-class`
(e.g. `maob_inhibitor`, `kinase_inhibitor`, `a2a_antagonist`) or extend the
taxonomy in `pocketbox/ligand.py`.

Always re-dock a known co-crystallised ligand first to validate the box before
trusting scores on a novel compound.
