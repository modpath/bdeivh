# BDEIVH · Phylodynamic inference for Oropouche virus

Estimate vector-mediated transmission from a dated Oropouche virus (OROV) phylogeny and a human genomic sampling probability. The repository pairs the Birth–Death Exposed–Infectious Vector–Human (BDEIVH) model with a convolutional neural network adapted from PhyloCNN. The pretrained estimator accepts a single cluster with 30–350 sampled tips and returns six epidemiological estimates. It can also repeat inference as samples accumulate over time.

## Predict from a dated tree

Run the commands from the repository root. You need Python with TensorFlow, Keras, NumPy, pandas and ETE3, the prediction script and encoder, and a matching model architecture (`.json`) and weights (`.weights.h5`) in `trained_models/`. The inspected 50–150-tip OROV model JSON records Keras 2.13.1 and contains a serialized Lambda layer. Use trusted repository models in a compatible TensorFlow/Keras environment. The script uses a private Keras serialization API, so compatibility with other versions may vary.

**Check the encoder filename first.** The prediction script imports `PhyloCNN_Encoding`. On a case-sensitive filesystem, a file named `Phylocnn_Encoding.py` requires changing that import to `import Phylocnn_Encoding as PhyloCNN_Encoding`, or renaming the module to match the existing import.

### Choose the model and sampling probability

Use the `OROV_OROV` model family for empirical OROV time trees. These models were trained on simulations with reduced phylogenetic resolution. The `SIM_SIM` family uses simulated transmission trees (TTs). For whole-tree inference, select the architecture and weights explicitly. The command defaults to the 30–50-tip model regardless of the number of tips in the input.

| Example tree | Cluster | Tips in supplied file | Genomes in manuscript Table 2 | Study sampling probability | Whole-tree model prefix |
|---|---|---:|---:|---:|---|
| `cea.nwk` | CE-2024, Ceará | 86 | 90 | 0.396 | `50_150_OROV_OROV` |
| `es1.nwk` | ES-I, Espírito Santo | 123 | 124 | 0.016 | `50_150_OROV_OROV` |
| `am1.nwk` | AM-I, Amazonas | 258 | 260 | 0.109 | `250_350_OROV_OROV` |

The probabilities above describe the manuscript clusters. The examples apply those values to the corresponding supplied files, whose tip counts differ. For a new cluster, supply its own human genomic sampling probability as a fraction. The study estimated this as the number of cluster genomes divided by the estimated number of associated cases in the same location and time interval. The training simulations used probabilities from 0.01 to 0.50.

### Whole-tree estimate

This example assumes the Ceará tree is in `datasets/orov_clusters/` and its branch lengths are in years. Set `--tree_scale 1` for years or `--tree_scale 0` for days, according to how the tree was dated. A Newick file with branch lengths in substitutions per site requires time dating before inference.

```bash
TREE=datasets/orov_clusters/cea.nwk
MODEL=trained_models/50_150_OROV_OROV_trained_BDEIVH

python Genomic_tree_predictions_temporal.py \
  --tree "$TREE" --tree_scale 1 --sampling_probability 0.396 \
  --model_json "${MODEL}.json" --model_weights "${MODEL}.weights.h5"
```

The script encodes the tree and writes `cea_predicted_parameters.csv` beside the input. Add `--keep_intermediate` to retain the tree in days and its encoded TSV.

### Cumulative estimates through time

```bash
python Genomic_tree_predictions_temporal.py \
  --tree datasets/orov_clusters/cea.nwk --tree_scale 1 \
  --sampling_probability 0.396 --temporal --step_days 7 \
  --temporal_mode observed --model_dir trained_models --model_prefix OROV_OROV
```

In `observed` mode, each estimate uses all tips sampled up to the cutoff, measured in days from the root. Each row therefore describes a cumulative tree. The final cutoff includes the full tree. The same sampling probability applies throughout. The optional `geometric` mode cuts branches crossing the cutoff and creates artificial tips there.

The script writes `cea_temporal_predictions.csv` with the cutoff, tip count, model class, status and predictions. It selects models for **30–49**, **50–149**, **150–249** and **250–350** tips using filename prefixes `30_50`, `50_150`, `150_250` and `250_350`, respectively. Cutoffs outside this range receive a skip status. Place the model pairs for all sizes the analysis can reach in the model directory.

## Interpret the predictions

BDEIVH represents exposed and infectious states in humans and vectors. An infectious human can infect vectors, and an infectious vector can infect humans. The simulations sample infectious humans. Each target has three CSV columns: the target name contains the median, and the suffixes `_lower` and `_upper` identify the 2.5th and 97.5th percentile predictions.

| Target | Interpretation |
|---|---|
| `Rvh` | Expected number of humans infected by one infectious vector |
| `Rhv` | Expected number of vectors infected by one infectious human |
| `Rhh` | Expected secondary human infections from one infectious human over a complete human–vector–human cycle |
| `d_v` | Total duration of infection in vectors, including incubation and infectious periods, in days |
| `d_h` | Total duration of infection in humans, including incubation and infectious periods, in days |
| `v2h_ratio` | Ratio of infected vectors to infected humans in the model |

In the transmission model, $R_{HH}=R_{HV}R_{VH}$. The CNN estimates each target's quantiles separately, so use the `Rhh` column for the median cycle-level estimate. The genomic prediction script exports raw model quantiles. The testing notebook applies a separate conformal adjustment when evaluating prediction intervals on simulated test trees.

## Reproduce training and evaluation

Pretrained prediction starts directly from a dated tree. To regenerate an estimator, follow the steps below for a single tree-size class and representation at a time. The manuscript covers four size classes. Individual Snakefiles target one class per run.

| Stage | Entry points | Output |
|---|---|---|
| Simulate TTs | `Snakefile_simulate_train`, `Snakefile_simulate_calibrate`, `Snakefile_simulate_test` call `Generate_tree_sets.py` | Newick trees and matching comma-separated parameter logs |
| Generate OROV-like trees | `Snakefile_transform_OROV_{train,calibrate,test}`, with `Snakefile_transform_OROV` as the inner training workflow | Reduced-resolution, time-dated trees with the original parameter rows |
| Merge each split | `Snakefile_merge_TT_{train,calibrate,test}` or `Snakefile_merge_OROV_{train,calibrate,test}` | One Newick forest and one parameter log per split, representation and size class |
| Encode | `Phylocnn_Encoding.py` | A tab-separated matrix with 800 rows per tree |
| Train | `Phylocnn_training.ipynb` | Model architecture, weights and loss-curve PDF in `trained_models/` |
| Calibrate and test | `Phylocnn_testing.ipynb` | Per-parameter test coverage and CSV files of true and predicted values in `predictions/` |

### 1. Generate and transform trees

The workflows use Snakemake with Singularity/Apptainer. The supplied cluster commands target Snakemake 7.32.x and Slurm at Institut Pasteur. For local use, check filesystem bindings first: the inner training transformation passes `-B /pasteur` explicitly. The transformation uses `transmission2phylogeny.py`, LSD2 and `nex2nwk.py`.

For example, the default training simulation generates 100 sets × 100 files × 10 trees in the 150–250-tip class. Select the same size explicitly when transforming these trees:

```bash
snakemake -s Snakefile_simulate_train --cores 10 --use-singularity
snakemake -s Snakefile_transform_OROV_train --cores 1 \
  --config mint=150 maxt=250
```

The transformation converts elapsed time to substitution counts under a variable molecular clock and an alignment length of 11,873 sites. It collapses internal branches with zero substitutions, retains sampled tips, and uses LSD2 to infer dated trees. The training workflow preserves tree order within each file.

Calibration and test simulations default to 2,000 and 1,000 single-tree files, respectively, both with 250–350 tips. Match the simulation, transformation, merge and notebook inputs to the intended size class. The simulation Snakefiles specify the target directory in `rule all`. Their default targets also request BD/BDEI benchmark fits for calibration and testing. For tree generation alone, remove the optional `.est_bdei` target from those two rules. To run the benchmarks, correct the sampling-probability column lookup in both fitting rules from `i <= $NF` to `i <= NF`.

### 2. Merge and encode

The merge workflows concatenate trees and parameter rows in matching numeric order, retaining one log header. Training uses set number followed by file number. Calibration and testing use file number.

Before running the OROV training merge, remove the stray `250 to 350` text after the `train_folder` assignment and the standalone `to` from `Snakefile_merge_OROV_train`. Use `sf=OROV_train` to match the transformation's output directory. The merge output filenames hard-code size ranges, so update their prefixes in both `rule all` and `rule merge_all` when changing classes. The TT test merge, for example, defaults to 30–50 tips while the test simulation defaults to 250–350.

For the default 250–350-tip OROV test data:

```bash
snakemake -s Snakefile_merge_OROV_test --cores 1
FOREST=OROV_test/250_350/250_350_test_merged_all_OROV.nwk
python Phylocnn_Encoding.py --tree "$FOREST" --output "${FOREST}_encoded"
```

Use the actual encoder filename if you renamed it during setup. Point the notebooks to the resulting encoded TSV and matching merged log, or place them under the filenames expected in `datasets/`. Each tree contributes 799 node-feature rows and one branch-length rescaling row. Verify that the number of TSV data rows equals **800 × the number of log data rows**, and check the encoder output for skipped trees. The notebooks align inputs by row order.

### 3. Train, calibrate and test

Open the notebooks in Jupyter with TensorFlow/Keras, NumPy, pandas and Matplotlib installed. Set the input and output paths in the notebook cells before running them in order.

1. **Train.** In `Phylocnn_training.ipynb`, point both input paths to the same training split and representation. The supplied encoding path is extensionless. The notebook scales both duration targets, appends the sampling probability to the node features, and forms a `400 × 12 × 2` tensor with separate tip and internal-node channels. It trains three ordered quantiles per target using pinball loss, a 30% validation split and early stopping.
2. **Select the model to evaluate.** The training notebook saves a `250_350_OROV_OROV` model. The testing notebook loads a `250_350_SIM_SIM` model and reads OROV-like calibration and test trees. Keep these paths for that cross-representation comparison, or select the JSON and weights written by training to evaluate the newly fitted model. Update the prediction filenames to describe the comparison you ran.
3. **Evaluate.** The testing notebook uses the separate calibration split to adjust the interval endpoints for nominal 95% coverage per target. It reports empirical coverage on the test split, converts the duration estimates back to the original time units, and exports the true values, predicted medians and calibrated bounds.

### Compare an empirical tree with the training distribution

`sumstat_calculator.py` computes tree summary statistics, fits a scaler on the selected training trees, and writes a `.sumstats_log` for the empirical tree. Set the training directory, file pattern and empirical tree path in the script to match the size class and representation under study. The script example uses a five-standard-deviation threshold. To reproduce the manuscript's cluster screen, use three standard deviations and exclude a cluster if more than one statistic exceeds that threshold.

## Associated manuscript

Ighor Arantes, Gonzalo Bello, Felipe Gomes Naveca, Edson Delatorre, Antonio Silva Lima Neto, Sebastian Duchene and Anna Zhukova. *Deep Learning-based Phylodynamic Inference of Oropouche Virus Transmission Dynamics in Brazil Between 2023 and 2025*. Table 2 gives the empirical sampling probabilities and Section 4.11 describes their estimation.

The neural tree encoding and convolutional architecture build on [PhyloCNN](https://github.com/manolofperez/phyloCNN).
