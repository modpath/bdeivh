# BDEIVH - training, test and CI calibration dataset simulation

This folder contains the [Snakemake](https://snakemake.readthedocs.io/en/stable/) pipelines needed for training, test and calibration dataset simulation and analysis.

## 1. Test set simulation
To simulate a test set of 1,000 transmission trees for BDEIVH model, using [treesimulator](https://github.com/evolbioinfo/treesimulator),
for the tree size of 30-50 tips on a local computer 
and put them into a folder named `TT_test`:
```bash
snakemake --snakefile Snakefile_simulate_test \
--rerun-triggers mtime --keep-going --cores 12 \
--use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~" \
--config mint=30 maxt=50 sf='TT_test' num=1000
```
This will create a folder `TT_test/30_50`, 
each containing 1,000 trees `tree.0.nwk`, ..., `tree.999.nwk`, 
and the log files containing the parameters used for the simulation `tree.0.log`, ..., `tree.999.log` 
for each tree.

Once completed, this script will also automatically call the `Snakemake_merge_test` pipeline to merge the tree files into one file, 
and encode them in [PhyloCNN](https://github.com/manolofperez/phyloCNN) format,
creating the file `TT_test/30_50/trees.nwk` containing 1,000 trees 
the file `TT_test/30_50/trees.tsv.xz` containing the encoded features for each tree 
and the file `TT_test/30_50/trees.log` containing the parameters used for the simulation for each tree.

To run the same pipeline on a slurm cluster, the command should be somewhat similar to:
```bash
 snakemake --snakefile Snakefile_simulate_test --rerun-triggers mtime --keep-going --cores 1 \
 --use-singularity --singularity-prefix $HOME/.singularity --singularity-args "-B /pasteur" \
 --cluster "sbatch -c {threads} -o {folder}/logs/{params.name}.log -e {folder}/logs/{params.name}.log --mem={params.mem} \
 -p common --qos=normal  -J {params.name}" --jobs 500 \
 --config mint=30 maxt=50 sf='TT_test' num=1000 
```
### 1.1 Transformation to OROV-like trees

The following pipeline transforms the 1,000 aforementioned perfect 30-50-tip test transmission trees 
into Oropouche virus (OROV)-like test trees 
by (1) simulating the number of substitutions along each transmission-tree branch and tip sampling dates 
according to the OROV substitution rate,
(2) collapsing the branches with zero substitutions, 
and finally (3) time-scaling the resulting trees according to the OROV substitution rate 
and the simulated tip sampling times with [LSD2](https://github.com/tothuhien/lsd2).
```bash
snakemake --snakefile Snakefile_transform_OROV_test \
--rerun-triggers mtime --keep-going --cores 12 \
--use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~" \
--config mint=30 maxt=50 sf='TT_test' cf='OROV_test' num=1000
```
This will create a folder `OROV_test/30_50`, 
each containing 1,000 trees `tree.0.nwk`, ..., `tree.999.nwk`, 
and the log files containing the parameters used for the simulation `tree.0.log`, ..., `tree.999.log` 
for each tree.

Once completed, this script will also automatically call the `Snakemake_merge_test` pipeline to merge the tree files into one file,, 
and encode them in [PhyloCNN](https://github.com/manolofperez/phyloCNN) format,
creating the file `OROV_test/30_50/trees.nwk` containing 1,000 trees 
the file `OROV_test/30_50/trees.tsv.xz` containing the encoded features for each tree 
and the file `OROV_test/30_50/trees.log` containing the parameters used for the simulation for each tree.

To run the same pipeline on a slurm cluster, the command should be somewhat similar to:
```bash
 snakemake --snakefile Snakefile_transform_OROV_test --rerun-triggers mtime --keep-going --cores 1 \
 --use-singularity --singularity-prefix $HOME/.singularity --singularity-args "-B /pasteur" \
 --cluster "sbatch -c {threads} -o {folder}/logs/{params.name}.log -e {folder}/logs/{params.name}.log --mem={params.mem} \
 -p common --qos=normal  -J {params.name}" --jobs 500 \
 --config mint=30 maxt=50 sf='TT_test' cf='OROV_test' num=1000 
```

## 2. CI calibration set simulation
To simulate a CI calibration set of 2,000 trees for BDEIVH model
set the parameter `num=2000` (instead of 1000), the parameter `sf='TT_calibrate'` (instead of 'TT_test') 
and the parameter `cf='OROV_calibrate'` (instead of 'OROV_test') in the test set simulations described above.

## 3. Training set simulation
To simulate a training set of 100 * 100 * 10 = 100,000 30-50-tip-training trees for BDEIVH model
 on a local computer and put them into a folder named `TT_train`:
```bash
snakemake --snakefile Snakefile_simulate_train \
--rerun-triggers mtime --keep-going --cores 12 \
--use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~" \
--config mint=30 maxt=50 sf='TT_train' trees=100 folders=100 treesperfile=10
```
This will create a folder `TT_train/30_50` 
containing 100 subfolders `0`, ..., `100`, 
each containing 100 tree files `trees.0.nwk`, ..., `trees.127.nwk` with 10 trees each, 
and the log files containing the parameters used for the simulation `trees.0.log`, ..., `trees.128.log` for each tree file (with 10 parameter set each).

Once completed, this script will also automatically call the `Snakemake_merge_train` pipeline to merge the tree files into one file,
and encode them in [PhyloCNN](https://github.com/manolofperez/phyloCNN) format,
creating the file `TT_train/30_50/trees.nwk` containing 100,000 trees 
and the file `TT_train/30_50/trees.tsv.xz` containing the encoded features for each tree
and the file `TT_train/30_50/trees.log` containing the parameters used for the simulation for each tree.

To run the same pipeline on a slurm cluster, the command should be somewhat similar to:
```bash
 snakemake --snakefile Snakefile_simulate_train --rerun-triggers mtime --keep-going --cores 1 \
 --use-singularity --singularity-prefix $HOME/.singularity --singularity-args "-B /pasteur" \
 --cluster "sbatch -c {threads} -o {folder}/logs/{params.name}.log -e {folder}/logs/{params.name}.log --mem={params.mem} \
 -p common --qos=normal  -J {params.name}" --jobs 1000 \
 --config mint=30 maxt=50 sf='TT_train' trees=100 folders=100 treesperfile=10
```

### 3.1 Transformation to OROV-like trees

The following pipeline transforms the 100,000 aforementioned perfect 30-50-tip training transmission trees 
into OROV-like training trees:
```bash
snakemake --snakefile Snakefile_transform_OROV_train \
--rerun-triggers mtime --keep-going --cores 12 \
--use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~" \
--config mint=30 maxt=50 sf='TT_train' cf='OROV_train' trees=100 folders=100 treesperfile=10
```
This will create a folder `OROV_train/30_50`, 
containing 100 subfolders `0`, ..., `100`, 
each containing 100 tree files `trees.0.nwk`, ..., `trees.127.nwk` with 10 trees each, 
and the log files containing the parameters used for the simulation `trees.0.log`, ..., `trees.128.log` for each tree file (with 10 parameter set each).

Once completed, this script will also automatically call the `Snakemake_merge_train` pipeline to merge the tree files into one file,
and encode them in [PhyloCNN](https://github.com/manolofperez/phyloCNN) format,
creating the file `OROV_train/30_50/trees.nwk` containing 100,000 trees 
the file `OROV_train/30_50/trees.tsv.xz` containing the encoded features for each tree 
and the file `OROV_train/30_50/trees.log` containing the parameters used for the simulation for each tree.

To run the same pipeline on a slurm cluster, the command should be somewhat similar to:
```bash
 snakemake --snakefile Snakefile_transform_OROV_train --rerun-triggers mtime --keep-going --cores 1 \
 --use-singularity --singularity-prefix $HOME/.singularity --singularity-args "-B /pasteur" \
 --cluster "sbatch -c {threads} -o {folder}/logs/{params.name}.log -e {folder}/logs/{params.name}.log --mem={params.mem} \
 -p common --qos=normal  -J {params.name}" --jobs 500 \
--config mint=30 maxt=50 sf='TT_train' cf='OROV_train' trees=100 folders=100 treesperfile=10
```

## 4. ML analysis of the test set

The following script runs BDEI and BD model parameter estimations on each of the 1,000 30-50-tip test set transmission trees 
(using maximum-likelihood estimators [pyBDEI](https://github.com/evolbioinfo/bdei) and [bd](https://github.com/evolbioinfo/bdct))
and puts the results into files `tree.0.est_bdei`, ..., `tree.999.est_bdei` and `tree.0.est_bd`, ..., `tree.999.est_bd` 
inside the test folder `TT_test/30_50`:

```bash
snakemake --snakefile Snakefile_estimate_test \
--rerun-triggers mtime --keep-going --cores 12 \
--use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~" \
--config mint=30 maxt=50 sf='TT_test' num=1000
```

To run the same pipeline on a slurm cluster, the command should be somewhat similar to:
```bash
 snakemake --snakefile Snakefile_estimate_test --rerun-triggers mtime --keep-going --cores 1 \
 --use-singularity --singularity-prefix $HOME/.singularity --singularity-args "-B /pasteur" \
 --cluster "sbatch -c {threads} -o {folder}/logs/{params.name}.log -e {folder}/logs/{params.name}.log --mem={params.mem} \
 -p common --qos=normal  -J {params.name}" --jobs 500 \
 --config mint=30 maxt=50 sf='TT_test' num=1000 
```