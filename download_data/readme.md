# Distant data collection steps

1) Download the CAEVO system from https://www.usna.edu/Users/cs/nchamber/caevo/

2) Download Gigaword.

3) Run process.py for all folders within gigaword.

4) Modify the default.sieves file in caevo-master. Comment out all sieves except the AdjacentVerbTimex sieve and TimeTimeSieve.
   Run runcaevo.sh on all files generated from step 3.

5) Grep through the output of runcaevo.sh to identify files with the following text: tlink="ee"
   These files contain event pairs that were ordered according to the above two sieves.

6) Run create_file.py on the files obtained after Step 5.
