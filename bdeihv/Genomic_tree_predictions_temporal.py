#!/usr/bin/env python3
"""
Predict epidemiological parameters from a genomic tree using a trained PhyloCNN model.

This version supports models trained with pinball loss, producing prediction intervals:
    6 target parameters x 3 quantiles = 18 output values

Output columns:
    Rvh_lower, Rvh, Rvh_upper
    Rhv_lower, Rhv, Rhv_upper
    Rhh_lower, Rhh, Rhh_upper
    d_v_lower, d_v, d_v_upper
    d_h_lower, d_h, d_h_upper
    v2h_ratio_lower, v2h_ratio, v2h_ratio_upper
"""

import os
import sys
import argparse
import warnings
import tempfile
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from keras.src.saving import serialization_lib

from bdeihv.Phylocnn_Encoding import process_tree

serialization_lib.enable_unsafe_deserialization()

from ete3 import Tree

warnings.filterwarnings("ignore")
sys.setrecursionlimit(100000)

TARGET_COLS = ["Rvh", "Rhv", "Rhh", "d_v", "d_h", "v2h_ratio"]
QUANTILES = ("lower", "median", "upper")
COLS_TO_RESCALE = ["d_v", "d_h"]

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

OROV = 'OROV'
TRANSMISSION = 'TRANSMISSION'


def genomic_tree_rescaling(original_tree, tree_scale):
    """Rescale branch lengths to days when the supplied tree is in years.

    tree_scale:
        0 = tree already in days
        1 = tree in years, converted to days
    """
    if tree_scale not in (0, 1):
        raise ValueError("tree_scale must be 0 for days or 1 for years.")

    scaled_tree = original_tree.copy(method="deepcopy")
    if tree_scale == 1:
        for node in scaled_tree.traverse():
            node.dist *= 365.0
    return scaled_tree


def encode_pad_0s_rootage_single(enc, max_len=400):
    """Pad leaves and internal nodes to match the model input shape."""
    enc_pad = []

    for i in range(enc.shape[0]):
        leaves = enc[i][enc[i, :, 3] == 1]
        leaves = leaves[np.argsort(leaves[:, 1])]

        if leaves.shape[0] > max_len:
            raise ValueError(f"Too many leaves: {leaves.shape[0]} > {max_len}.")

        leaves = np.pad(
            leaves,
            [(0, max_len - leaves.shape[0]), (0, 0)],
            mode="constant",
        )

        nodes = enc[i][enc[i, :, 3] > 1]
        nodes = nodes[np.argsort(nodes[:, 1])]

        if nodes.shape[0] == 0:
            raise ValueError("No internal nodes were found in the encoded tree.")

        # Same behavior used in the training notebook: duplicate the last internal node.
        nodes = np.append(nodes, nodes[-1].reshape(1, -1), axis=0)

        if nodes.shape[0] > max_len:
            raise ValueError(f"Too many internal nodes: {nodes.shape[0]} > {max_len}.")

        nodes = np.pad(
            nodes,
            [(0, max_len - nodes.shape[0]), (0, 0)],
            mode="constant",
        )

        enc_pad.append(np.stack((leaves, nodes), axis=2))

    return np.array(enc_pad, dtype="float32")


def load_encoded_tree(encoded_file):
    """Load a PhyloCNN encoded tree file and extract its rescaling factor."""
    if not os.path.exists(encoded_file):
        raise FileNotFoundError(f"Encoded file was not created: {encoded_file}")

    enc_real_df = pd.read_csv(encoded_file, sep="\t", header=0)

    drop_cols = [c for c in ("Index", "Unnamed: 0") if c in enc_real_df.columns]
    if drop_cols:
        enc_real_df = enc_real_df.drop(columns=drop_cols)

    enc_real = enc_real_df.values.astype("float32")

    rows_per_tree = 800
    n_rows = enc_real.shape[0]
    if n_rows % rows_per_tree != 0:
        raise ValueError(
            f"Encoded file has {n_rows} rows, which is not a multiple of {rows_per_tree}."
        )

    n_trees = n_rows // rows_per_tree
    enc_real = enc_real.reshape(n_trees, rows_per_tree, enc_real.shape[1])

    resc_factor_real = enc_real[:, -1, -1]
    enc_real = np.delete(enc_real, -1, axis=1)

    return enc_real, resc_factor_real


def add_sampling_probability(enc_real, sampling_probability):
    """Add the sampling probability p as one extra feature column for every row."""
    if not 0 <= sampling_probability <= 1:
        raise ValueError("sampling_probability must be between 0 and 1.")

    p_col = np.full(
        (enc_real.shape[0], enc_real.shape[1], 1),
        sampling_probability,
        dtype="float32",
    )
    return np.concatenate((enc_real, p_col), axis=2)


def load_trained_model(model_json_file, model_weights_file):
    """Load a trusted Keras model containing Lambda layers."""

    if not os.path.exists(model_json_file):
        raise FileNotFoundError(f"Model JSON not found: {model_json_file}")

    if not os.path.exists(model_weights_file):
        raise FileNotFoundError(f"Model weights not found: {model_weights_file}")

    with open(model_json_file, "r") as json_file:
        model_json = json_file.read()

    # Required by this Keras version to deserialize trusted Lambda layers.
    from keras.src.saving import serialization_lib
    serialization_lib.enable_unsafe_deserialization()

    model = model_from_json(model_json)
    model.load_weights(model_weights_file)

    return model


def predictions_to_dataframe(pred, resc_factor_real):
    """Convert model output to a dataframe with lower/median/upper columns."""
    expected_outputs = len(TARGET_COLS) * len(QUANTILES)

    if len(pred.shape) != 2:
        pred = np.reshape(pred, (pred.shape[0], -1))

    if pred.shape[1] != expected_outputs:
        raise ValueError(
            f"Unexpected model output shape {pred.shape}. Expected {expected_outputs} columns "
            f"({len(TARGET_COLS)} targets x {len(QUANTILES)} quantiles)."
        )

    pred = pred.reshape(-1, len(TARGET_COLS), len(QUANTILES))

    pred_cols = []
    for target in TARGET_COLS:
        pred_cols.extend([f"{target}_lower", target, f"{target}_upper"])

    predicted = pd.DataFrame(
        pred.reshape(pred.shape[0], -1),
        columns=pred_cols,
    )

    # d_v and d_h were trained in rescaled units, so all quantiles must be rescaled back.
    for col in COLS_TO_RESCALE:
        for q_col in (f"{col}_lower", col, f"{col}_upper"):
            predicted[q_col] = predicted[q_col].values * resc_factor_real

    return predicted


def predict_parameters(encoded_file, sampling_probability, model_json_file, model_weights_file):
    enc_real, resc_factor_real = load_encoded_tree(encoded_file)
    enc_real = add_sampling_probability(enc_real, sampling_probability)
    enc_real = encode_pad_0s_rootage_single(enc_real)

    model = load_trained_model(model_json_file, model_weights_file)
    pred = model.predict(enc_real)

    return predictions_to_dataframe(pred, resc_factor_real)



# Tree-size ranges used by the size-specific models.
# Boundaries are treated as [30, 50), [50, 150), [150, 250), [250, 350].
# If your simulation pipeline used a different convention for exact boundary
# values (50, 150, 250), adjust this table accordingly.
TEMPORAL_MODEL_BINS = (
    (30, 50, "30_50"),
    (50, 150, "50_150"),
    (150, 250, "150_250"),
    (250, 350, "250_350"),
)


def get_tree_height(tree):
    """Return the maximum root-to-tip distance of a tree."""
    leaves = tree.get_leaves()
    if not leaves:
        raise ValueError("The tree has no leaves.")
    return max(tree.get_distance(leaf) for leaf in leaves)


def prune_tree_to_observed_tips(original_tree, cutoff_days, min_tips=2):
    """Return the tree that would be observable up to ``cutoff_days``.

    Only original tips whose root-to-tip distance is <= cutoff_days are kept.
    Future tips are removed and no artificial tips are created at the cutoff.
    This is the recommended mode for retrospective real-time analyses because
    it avoids using lineages that are only known from future samples.
    """
    if cutoff_days < 0:
        raise ValueError("cutoff_days must be >= 0.")

    tree = original_tree.copy(method="deepcopy")
    keep = [
        leaf for leaf in tree.get_leaves()
        if tree.get_distance(leaf) <= cutoff_days + 1e-9
    ]

    if len(keep) < min_tips:
        return None

    tree.prune(keep, preserve_branch_length=True)
    return tree


def truncate_tree_geometrically(original_tree, cutoff_days):
    """Geometrically truncate every lineage crossing ``cutoff_days``.

    Branches crossing the time slice are shortened to the cutoff and their
    descendants are removed, creating artificial tips at the slice. This is
    useful for a literal time-slice analysis, but it is not equivalent to the
    phylogeny that would have been observable in real time.
    """
    if cutoff_days < 0:
        raise ValueError("cutoff_days must be >= 0.")

    tree = original_tree.copy(method="deepcopy")
    synthetic_tip_counter = [0]

    def _walk(node, parent_age):
        for child in list(node.children):
            child_age = parent_age + child.dist

            if child_age > cutoff_days + 1e-9:
                # This branch crosses the requested time slice.
                child.dist = max(cutoff_days - parent_age, 0.0)

                for descendant in list(child.children):
                    descendant.detach()

                synthetic_tip_counter[0] += 1
                child.name = (
                    f"slice_{cutoff_days:g}d_"
                    f"{synthetic_tip_counter[0]:04d}"
                )
            else:
                _walk(child, child_age)

    _walk(tree, 0.0)
    return tree


def select_temporal_model(n_tips, model_dir=MODEL_DIR, tree_type=OROV):
    """Return (label, json_path, weights_path) for a tree-size-specific model."""
    selected = None

    for i, (lower, upper, label) in enumerate(TEMPORAL_MODEL_BINS):
        is_last = i == len(TEMPORAL_MODEL_BINS) - 1
        inside = (lower <= n_tips <= upper) if is_last else (lower <= n_tips < upper)
        if inside:
            selected = label
            break

    if selected is None:
        return None

    model_json = os.path.join(
        model_dir,
        f"{selected}_{tree_type}_BDEIVH.json",
    )
    model_weights = os.path.join(
        model_dir,
        f"{selected}_{tree_type}_BDEIVH.weights.h5",
    )

    return selected, model_json, model_weights


def predict_tree_object(
    tree,
    sampling_probability,
    model,
    workdir,
    prefix,
):
    """Encode one ETE tree and predict its epidemiological parameters."""
    tree_file = os.path.join(workdir, f"{prefix}.nwk")
    encoded_file = os.path.join(workdir, f"{prefix}.tsv")

    tree.write(outfile=tree_file, format=1)
    encode_tree(tree_file, encoded_file)

    enc_real, resc_factor_real = load_encoded_tree(encoded_file)
    enc_real = add_sampling_probability(enc_real, sampling_probability)
    enc_real = encode_pad_0s_rootage_single(enc_real)

    pred = model.predict(enc_real, verbose=0)
    return predictions_to_dataframe(pred, resc_factor_real)


def encode_tree(tree_file: str, encoded_file: str):
    result = process_tree(tree_file)
    if isinstance(result, pd.DataFrame):
        result.to_csv(encoded_file, sep="\t", index=True)
    if isinstance(result, np.ndarray):
        pd.DataFrame(result).to_csv(encoded_file, sep="\t", index=True)


def predict_every_n_days(
    original_tree,
    sampling_probability,
    step_days=7.0,
    model_dir=MODEL_DIR,
    tree_type=OROV,
    include_final=True,
):
    """Predict BDEIVH parameters from sequential time slices of one tree.

    Parameters
    ----------
    original_tree : ete3.Tree
        Tree already expressed in days. Apply ``genomic_tree_rescaling`` first
        when the input tree is in years.
    sampling_probability : float
        Sampling probability p supplied to the PhyloCNN, as in the original
        prediction script. This function assumes p is constant through time.
    step_days : float, default=7
        Interval between successive cutoffs measured from the root.
    model_dir : str
        Directory containing the four tree-size-specific trained models.
    tree_type : str
        Type of tree being analyzed (e.g., 'OROV' or 'TRANSMISSION').
    include_final : bool
        Also analyse the full-tree height when it is not an exact multiple of
        ``step_days``.

    Returns
    -------
    pandas.DataFrame
        One row per cutoff, including cutoff time, tip count, model used,
        status, and the 18 lower/median/upper predictions when inference was
        possible.
    """
    if step_days <= 0:
        raise ValueError("step_days must be > 0.")
    if not 0 <= sampling_probability <= 1:
        raise ValueError("sampling_probability must be between 0 and 1.")

    max_age = get_tree_height(original_tree)
    cutoffs = list(np.arange(step_days, max_age + 1e-9, step_days))

    if include_final and (not cutoffs or not np.isclose(cutoffs[-1], max_age)):
        cutoffs.append(max_age)

    rows = []
    model_cache = {}

    with tempfile.TemporaryDirectory(prefix="phylocnn_temporal_") as workdir:
        for idx, cutoff in enumerate(cutoffs, start=1):
            cut_tree = prune_tree_to_observed_tips(original_tree, cutoff)

            if cut_tree is None:
                rows.append({
                    "cutoff_days": float(cutoff),
                    "n_tips": 0,
                    "model_size": None,
                    "status": "skipped: fewer than 2 observed tips",
                })
                continue

            n_tips = len(cut_tree.get_leaves())
            model_info = select_temporal_model(
                n_tips=n_tips,
                model_dir=model_dir,
                tree_type=tree_type,
            )

            if model_info is None:
                rows.append({
                    "cutoff_days": float(cutoff),
                    "n_tips": int(n_tips),
                    "model_size": None,
                    "status": "skipped: tip count outside trained ranges (30-350)",
                })
                continue

            model_label, model_json, model_weights = model_info

            if model_label not in model_cache:
                model_cache[model_label] = load_trained_model(
                    model_json_file=model_json,
                    model_weights_file=model_weights,
                )

            predicted = predict_tree_object(
                tree=cut_tree,
                sampling_probability=sampling_probability,
                model=model_cache[model_label],
                workdir=workdir,
                prefix=f"slice_{idx:04d}",
            )

            if predicted.shape[0] != 1:
                raise ValueError(
                    f"Expected one prediction for cutoff {cutoff}, "
                    f"but received {predicted.shape[0]}."
                )

            row = {
                "cutoff_days": float(cutoff),
                "n_tips": int(n_tips),
                "model_size": model_label,
                "status": "predicted",
            }
            row.update(predicted.iloc[0].to_dict())
            rows.append(row)

    return pd.DataFrame(rows)

def print_predictions_summary(predicted):
    print("\nPredictions summary:")
    for target in TARGET_COLS:
        lower = f"{target}_lower"
        median = target
        upper = f"{target}_upper"
        print(f"\n{target}:")
        print(f"  Median lower: {np.median(predicted[lower])}")
        print(f"  Median prediction: {np.median(predicted[median])}")
        print(f"  Median upper: {np.median(predicted[upper])}")
        print(f"  Std prediction: {np.std(predicted[median])}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a genomic tree and predict epidemiological parameters with prediction intervals."
    )
    parser.add_argument(
        "-ts", "--tree_scale", default=1, type=int,
        help="Date scale used in the supplied tree: 0 = days, 1 = years."
    )
    parser.add_argument("-t", "--tree", required=True, help="Tree file in .nwk format.")
    parser.add_argument(
        "-p", "--sampling_probability", required=True, type=float,
        help="Sampling probability p."
    )
    parser.add_argument(
        "--tree_type", default=OROV, type=str, choices=(OROV, TRANSMISSION),
        help="Type of tree: OROV = Oropouche-like trees, TRANSMISSION = perfectly resolved transmission trees."
    )
    parser.add_argument(
        "--keep_intermediate", action="store_true",
        help="Keep rescaled tree and encoded TSV files."
    )
    parser.add_argument(
        "--temporal", action="store_true",
        help="Run sequential predictions at regular time intervals from the root."
    )
    parser.add_argument(
        "--step_days", type=float, default=7.0,
        help="Temporal interval in days when --temporal is used (default: 7)."
    )
    parser.add_argument(
        "--model_dir", default=MODEL_DIR,
        help="Directory containing pretrained models."
    )

    args = parser.parse_args()

    base_name = os.path.splitext(args.tree)[0]
    rescaled_output = f"{base_name}_rescaled_trees.nwk"
    encoded_output = f"{base_name}_encoded_trees.tsv"
    prediction_output = f"{base_name}_predicted_parameters.csv"

    scaled_tree = genomic_tree_rescaling(
        original_tree=Tree(args.tree),
        tree_scale=args.tree_scale,
    )

    if args.temporal:
        prediction_output = f"{base_name}_temporal_predictions.csv"
        temporal_predictions = predict_every_n_days(
            original_tree=scaled_tree,
            sampling_probability=args.sampling_probability,
            step_days=args.step_days,
            model_dir=args.model_dir,
            tree_type=args.tree_type,
            include_final=True,
        )
        temporal_predictions.to_csv(prediction_output, index=False)
        print(temporal_predictions.to_string(index=False))
        print(f"\nSaved temporal predictions to: {prediction_output}")
        return

    model_info = select_temporal_model(
        n_tips=len(scaled_tree),
        model_dir=args.model_dir,
        tree_type=args.tree_type,
    )

    if model_info is None:
        raise ValueError(
            f"Tree has {len(scaled_tree)} tips, which is outside the trained ranges (30-350)."
        )

    _, model_json, model_weights = model_info

    # Save the rescaled tree before calling the encoder.
    scaled_tree.write(outfile=rescaled_output, format=1)

    encode_tree(
        tree_file=rescaled_output,
        encoded_file=encoded_output,
    )

    predicted = predict_parameters(
        encoded_file=encoded_output,
        sampling_probability=args.sampling_probability,
        model_json_file=model_json,
        model_weights_file=model_weights,
    )

    print_predictions_summary(predicted)

    predicted.to_csv(prediction_output, index=False)
    print(f"\nSaved predictions to: {prediction_output}")

    if not args.keep_intermediate:
        for path in (rescaled_output, encoded_output):
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    main()
