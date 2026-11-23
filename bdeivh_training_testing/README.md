# Training BDEIVH model with PhyloCNN

This folder contains two Jupyter notebooks:
* [Phylocnn_training.ipynb](Phylocnn_training.ipynb), demonstrating how to train the BDEIVH model 
with PhyloCNN using the training trees simulated in the [tree_simulation](../tree_simulation) folder,
and 
* [Phylocnn_testing.ipynb](Phylocnn_testing.ipynb) demonstrating how to test the trained model with the testing trees simulated in the [tree_simulation](../tree_simulation) folder.

These notebooks were adapted to BDEIVH from [PhyloCNN](https://github.com/manolofperez/phyloCNN).

You might need to adapt the following constants in the notebooks to your case:
- `TREE_SIZE` -- the size of the training trees (e.g., '30_50' for 30-50-tip trees)
- `TREE_TYPE` -- the type of the training trees (e.g., 'OROV' for OROV-like trees or 'TT' for transmission trees)
- `TRAIN_PARAMS` -- the path to the training tree parameter csv file (e.g., '../tree_simulation/OROV_train/30_50/trees.log')
- `TRAIN_ENC_TREES` -- the path to the tsv-encoded training tree file (e.g., '../tree_simulation/OROV_train/30_50/trees.tsv.xz')
- `MODEL_PATH` -- the path to the folder where the trained model will be/is stored (e.g., '../bdeivh/models')
- `MODEL_NAME` -- the name of the trained model (e.g., '30_50_OROV_BDEIVH')
- `TEST_PARAMS` -- the path to the training tree parameter csv file (e.g., '../tree_simulation/OROV_test/30_50/trees.log')
- `TEST_ENC_TREES` -- the path to the tsv-encoded test tree file (e.g., '../tree_simulation/OROV_test/30_50/trees.tsv.xz')
- `TEST_EST` -- the path where the estimated parameters will be saved (e.g., '../tree_simulation/OROV_test/30_50/estimates_bdeivh.csv')
- `TEST_TRUE` -- the path where the true parameters will be saved (e.g., '../tree_simulation/OROV_test/30_50/target_bdeivh.csv')
- `CALIBRATION_PARAMS` -- the path to the calibration tree parameter csv file (e.g., '../tree_simulation/OROV_calibrate/30_50/trees.log')
- `CALIBRATION_ENC_TREES` -- the path to the tsv-encoded calibration tree file (e.g., '../tree_simulation/OROV_calibrate/30_50/trees.tsv.xz')
