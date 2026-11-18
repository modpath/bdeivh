import io
import os
from glob import iglob

import numpy as np
import pandas as pd
from ete3 import Tree
from treesumstats import FeatureRegistry, FeatureManager
from treesumstats.balance_sumstats import BalanceFeatureCalculator
from treesumstats.basic_sumstats import BasicFeatureCalculator
from treesumstats.branch_sumstats import BranchFeatureCalculator
from treesumstats.event_time_sumstats import EventTimeFeatureCalculator
from treesumstats.ltt_sumstats import LTTFeatureCalculator
from treesumstats.resolution_sumstats import ResolutionFeatureCalculator
from treesumstats.subtree_sumstats import SubtreeFeatureCalculator
from treesumstats.transmission_chain_sumstats import TransmissionChainFeatureCalculator
from sklearn.preprocessing import StandardScaler

TARGET_AVG_BL = 1

CHAIN_LEN = 4
N_LTT_COORDINATES = 20

SCALING_FACTOR = 'sf'


def read_tree(tree_path):
    tree = None
    for f in (3, 2, 5, 0, 1, 4, 6, 7, 8, 9):
        try:
            tree = Tree(tree_path, format=f)
            break
        except:
            continue
    if not tree:
        raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(tree_path))
    return tree

def rescale_forest_to_avg_brlen(forest, target_avg_length):
    """
    Rescales the trees in the forest so that the average non-zero branch length becomes target_avg_length

    :param forest: list(ete3.Tree), forest of trees to be rescaled
    :param target_avg_length: float, the average non-zero non-root branch length to which we want to rescale the trees
    :return: float, resc_factor = tree_avg_length / target_avg_length
    """
    dist_sum, num = 0, 0
    for tree in forest:
        for node in tree.traverse():
            if node.dist > 0:
                if not node.is_root():
                    dist_sum += node.dist
                    num += 1
            else:
                node.dist = 0

    if 0 == dist_sum:
        num = 0
        for tree in forest:
            for node in tree.traverse():
                if node.dist > 0:
                    dist_sum += node.dist
                    num += 1

    avg_len = dist_sum / num

    resc_factor = avg_len / target_avg_length

    for tree in forest:
        for node in tree.traverse():
            node.dist /= resc_factor

    return resc_factor


def get_write_handle(path, temp_suffix=''):
    mode = 'wb' if path.endswith('.gz') or path.endswith('.xz') else 'w'
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path + temp_suffix, mode)
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path + temp_suffix, mode)
    return open(path + temp_suffix, mode)

FeatureRegistry.register(BasicFeatureCalculator())
FeatureRegistry.register(BranchFeatureCalculator())
FeatureRegistry.register(EventTimeFeatureCalculator())
FeatureRegistry.register(TransmissionChainFeatureCalculator(CHAIN_LEN, percentiles=[1, 5, 10, 25]))
FeatureRegistry.register(LTTFeatureCalculator(N_LTT_COORDINATES))
FeatureRegistry.register(BalanceFeatureCalculator())
FeatureRegistry.register(SubtreeFeatureCalculator())
FeatureRegistry.register(ResolutionFeatureCalculator())

BRLEN_STATS = ['brlen_inode_mean', 'brlen_inode_median', 'brlen_inode_var',
               'brlen_tip_mean', 'brlen_tip_median', 'brlen_tip_var',
               'brlen_inode_top_mean', 'brlen_inode_top_median', 'brlen_inode_top_var',
               'brlen_tip_top_mean', 'brlen_tip_top_median', 'brlen_tip_top_var',
               'brlen_inode_middle_mean', 'brlen_inode_middle_median', 'brlen_inode_middle_var',
               'brlen_tip_middle_mean', 'brlen_tip_middle_median', 'brlen_tip_middle_var',
               'brlen_inode_bottom_mean', 'brlen_inode_bottom_median', 'brlen_inode_bottom_var',
               'brlen_tip_bottom_mean', 'brlen_tip_bottom_median', 'brlen_tip_bottom_var',
               #
               'frac_brlen_inode_mean_by_brlen_tip_mean', 'frac_brlen_inode_median_by_brlen_tip_median',
               'frac_brlen_inode_var_by_brlen_tip_var',
               'frac_brlen_inode_top_mean_by_brlen_tip_top_mean', 'frac_brlen_inode_top_median_by_brlen_tip_top_median',
               'frac_brlen_inode_top_var_by_brlen_tip_top_var',
               'frac_brlen_inode_middle_mean_by_brlen_tip_middle_mean',
               'frac_brlen_inode_middle_median_by_brlen_tip_middle_median',
               'frac_brlen_inode_middle_var_by_brlen_tip_middle_var',
               'frac_brlen_inode_bottom_mean_by_brlen_tip_bottom_mean',
               'frac_brlen_inode_bottom_median_by_brlen_tip_bottom_median',
               'frac_brlen_inode_bottom_var_by_brlen_tip_bottom_var',
               ]
TIME_STATS = ['time_tip_normalized_mean', 'time_tip_normalized_min', 'time_tip_normalized_max',
              'time_tip_normalized_var',
              'time_tip_normalized_median',
              'time_inode_normalized_mean', 'time_inode_normalized_min', 'time_inode_normalized_max',
              'time_inode_normalized_var', 'time_inode_normalized_median']

CHAIN_STATS = ['n_4-chain_normalized',
               'brlen_sum_4-chain_mean', 'brlen_sum_4-chain_min', 'brlen_sum_4-chain_max', 'brlen_sum_4-chain_var',
               'brlen_sum_4-chain_median',
               'brlen_sum_4-chain_perc1', 'brlen_sum_4-chain_perc5', 'brlen_sum_4-chain_perc10',
               'brlen_sum_4-chain_perc25']

LTT_STATS = ['ltt_time0', 'ltt_time1', 'ltt_time2', 'ltt_time3', 'ltt_time4', 'ltt_time5', 'ltt_time6', 'ltt_time7',
             'ltt_time8', 'ltt_time9', 'ltt_time10', 'ltt_time11', 'ltt_time12', 'ltt_time13', 'ltt_time14',
             'ltt_time15',
             'ltt_time16', 'ltt_time17', 'ltt_time18', 'ltt_time19',
             'ltt_lineages0_normalized', 'ltt_lineages1_normalized', 'ltt_lineages2_normalized',
             'ltt_lineages3_normalized',
             'ltt_lineages4_normalized', 'ltt_lineages5_normalized', 'ltt_lineages6_normalized',
             'ltt_lineages7_normalized',
             'ltt_lineages8_normalized', 'ltt_lineages9_normalized', 'ltt_lineages10_normalized',
             'ltt_lineages11_normalized', 'ltt_lineages12_normalized', 'ltt_lineages13_normalized',
             'ltt_lineages14_normalized', 'ltt_lineages15_normalized', 'ltt_lineages16_normalized',
             'ltt_lineages17_normalized', 'ltt_lineages18_normalized', 'ltt_lineages19_normalized',
             #
             'time_lineages_max', 'time_lineages_max_top', 'time_lineages_max_middle', 'time_lineages_max_bottom',
             'lineages_max_normalized', 'lineages_max_top_normalized', 'lineages_max_middle_normalized',
             'lineages_max_bottom_normalized',
             #
             'lineage_slope_ratio',
             'lineage_slope_ratio_top',
             'lineage_slope_ratio_middle',
             'lineage_slope_ratio_bottom',
             #
             'lineage_start_to_max_slope_normalized', 'lineage_stop_to_max_slope_normalized',
             'lineage_start_to_max_slope_top_normalized', 'lineage_stop_to_max_slope_top_normalized',
             'lineage_start_to_max_slope_middle_normalized', 'lineage_stop_to_max_slope_middle_normalized',
             'lineage_start_to_max_slope_bottom_normalized', 'lineage_stop_to_max_slope_bottom_normalized'
             ]

BALANCE_STATS = ['colless_normalized',
                 'sackin_normalized',
                 'width_max_normalized', 'depth_max_normalized', 'width_depth_ratio_normalized',
                 'width_delta_normalized',
                 'frac_inodes_in_ladder', 'len_ladder_max_normalized',
                 'frac_inodes_imbalanced', 'imbalance_avg']

TOPOLOGY_STATS = ['frac_tips_in_2', 'frac_tips_in_3L', 'frac_tips_in_4L', 'frac_tips_in_4B', 'frac_tips_in_O',
                  'frac_tips_in_3U', 'frac_tips_in_4U', 'frac_tips_in_4U3U1', 'frac_tips_in_4U211',
                  'frac_inodes_with_sibling_inodes']

TIME_DIFF_STATS = ['time_diff_in_2_real_mean', 'time_diff_in_3L_real_mean', 'time_diff_in_3U_real_mean', 'time_diff_in_4U_real_mean', 'time_diff_in_I_real_mean',
                   'time_diff_in_2_real_min', 'time_diff_in_3L_real_min', 'time_diff_in_3U_real_min', 'time_diff_in_4U_real_min', 'time_diff_in_I_real_min',
                   'time_diff_in_2_real_max', 'time_diff_in_3L_real_max', 'time_diff_in_3U_real_max', 'time_diff_in_4U_real_max', 'time_diff_in_I_real_max',
                   'time_diff_in_2_real_var', 'time_diff_in_3L_real_var', 'time_diff_in_3U_real_var', 'time_diff_in_4U_real_var', 'time_diff_in_I_real_var',
                   'time_diff_in_2_real_median', 'time_diff_in_3L_real_median', 'time_diff_in_3U_real_median', 'time_diff_in_4U_real_median', 'time_diff_in_I_real_median',
                   #
                   'time_diff_in_2_random_mean', 'time_diff_in_3L_random_mean', 'time_diff_in_3U_random_mean', 'time_diff_in_4U_random_mean', 'time_diff_in_I_random_mean',
                   'time_diff_in_2_random_min', 'time_diff_in_3L_random_min', 'time_diff_in_3U_random_min', 'time_diff_in_4U_random_min', 'time_diff_in_I_random_min',
                   'time_diff_in_2_random_max', 'time_diff_in_3L_random_max', 'time_diff_in_3U_random_max', 'time_diff_in_4U_random_max', 'time_diff_in_I_random_max',
                   'time_diff_in_2_random_var', 'time_diff_in_3L_random_var', 'time_diff_in_3U_random_var', 'time_diff_in_4U_random_var', 'time_diff_in_I_random_var',
                   'time_diff_in_2_random_median', 'time_diff_in_3L_random_median', 'time_diff_in_3U_random_median', 'time_diff_in_4U_random_median', 'time_diff_in_I_random_median',
                   #
                   'time_diff_in_2_real_perc1', 'time_diff_in_2_real_perc5', 'time_diff_in_2_real_perc10',
                   'time_diff_in_2_real_perc25',
                   'time_diff_in_3L_real_perc1', 'time_diff_in_3L_real_perc5', 'time_diff_in_3L_real_perc10',
                   'time_diff_in_3L_real_perc25',
                   'time_diff_in_3U_real_perc1', 'time_diff_in_3U_real_perc5', 'time_diff_in_3U_real_perc10',
                   'time_diff_in_3U_real_perc25',
                   'time_diff_in_4U_real_perc1', 'time_diff_in_4U_real_perc5', 'time_diff_in_4U_real_perc10',
                   'time_diff_in_4U_real_perc25',
                   'time_diff_in_I_real_perc75', 'time_diff_in_I_real_perc90', 'time_diff_in_I_real_perc95',
                   'time_diff_in_I_real_perc99',
                   #
                   'time_diff_in_2_random_perc1', 'time_diff_in_2_random_perc5', 'time_diff_in_2_random_perc10',
                   'time_diff_in_2_random_perc25',
                   'time_diff_in_3L_random_perc1', 'time_diff_in_3L_random_perc5', 'time_diff_in_3L_random_perc10',
                   'time_diff_in_3L_random_perc25',
                   'time_diff_in_3U_random_perc1', 'time_diff_in_3U_random_perc5', 'time_diff_in_3U_random_perc10',
                   'time_diff_in_3U_random_perc25',
                   'time_diff_in_4U_random_perc1', 'time_diff_in_4U_random_perc5', 'time_diff_in_4U_random_perc10',
                   'time_diff_in_4U_random_perc25',
                   'time_diff_in_I_random_perc75', 'time_diff_in_I_random_perc90', 'time_diff_in_I_random_perc95',
                   'time_diff_in_I_random_perc99',
                   #
                   'time_diff_in_2_random_vs_real_frac_less', 'time_diff_in_3L_random_vs_real_frac_less',
                   'time_diff_in_3U_random_vs_real_frac_less', 'time_diff_in_4U_random_vs_real_frac_less',
                   'time_diff_in_I_random_vs_real_frac_more',
                   'time_diff_in_2_random_vs_real_pval_less', 'time_diff_in_3L_random_vs_real_pval_less',
                   'time_diff_in_3U_random_vs_real_pval_less', 'time_diff_in_4U_random_vs_real_pval_less',
                   'time_diff_in_I_random_vs_real_pval_more']

RESOLUTION_STATS = ['n_children_mean',
                    'n_children_var',
                    'frac_inodes_resolved',
                    'frac_inodes_resolved_non_zero']

STATS = (['n_tips', 'n_inodes'] \
        + BRLEN_STATS + TIME_STATS + CHAIN_STATS + LTT_STATS + BALANCE_STATS + TOPOLOGY_STATS
        + TIME_DIFF_STATS
        + RESOLUTION_STATS)

def forest2sumstat_df(forest, target_avg_brlen=TARGET_AVG_BL):
    """
    and returns a data frame, containing them along with the branch scaling factor.
    Rescales the input forest to have mean branch lengths of 1, calculates its summary statistics,

    :param forest: list(ete3.Tree) forest to encode
    :param target_avg_brlen: length of the average non-zero branch in the rescaled tree
    :return: pd.DataFrame containing the summary stats and the branch scaling factor
    """

    scaling_factor = rescale_forest_to_avg_brlen(forest, target_avg_length=target_avg_brlen)
    return pd.DataFrame.from_records([list(FeatureManager.compute_features(forest, *STATS))], columns=STATS)


def read_forest(tree_path):
    if os.path.exists(tree_path):
        with open(tree_path, 'r') as f:
            nwks = f.read().replace('\n', '').split(';')
    else:
        try:
            nwks = tree_path.replace('\n', '').split(';')
        except:
            pass
    if not nwks:
        raise ValueError('Could not find any trees (in newick or nexus format) in the file {}.'.format(tree_path))
    return [read_tree(nwk + ';') for nwk in nwks[:-1]]


def save_scaler_numpy(scaler, prefix, suffix=''):
    np.save(os.path.join(prefix, f'data_scaler{suffix}_mean.npy'), scaler.mean_, allow_pickle=False)
    np.save(os.path.join(prefix, f'data_scaler{suffix}_scale.npy'), scaler.scale_, allow_pickle=False)
    np.save(os.path.join(prefix, f'data_scaler{suffix}_var.npy'), scaler.var_, allow_pickle=False)
    with open(os.path.join(prefix, f'data_scaler{suffix}_n_samples_seen.txt'), 'w+') as f:
        f.write(f'{scaler.n_samples_seen_:d}')

def fit_scaler(paths):
    scaler = StandardScaler()
    for path in paths:
        df = pd.read_csv(path)
        X = df.loc[:, STATS].to_numpy(dtype=float, na_value=0)
        scaler.partial_fit(X)
    return scaler

def load_scaler_numpy(prefix):
    if os.path.exists(os.path.join(prefix, f'data_scaler_mean.npy')):
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(prefix, f'data_scaler_mean.npy'))
        scaler.scale_ = np.load(os.path.join(prefix, f'data_scaler_scale.npy'))
        scaler.var_ = np.load(os.path.join(prefix, f'data_scaler_var.npy'))
        with open(os.path.join(prefix, f'data_scaler_n_samples_seen.txt'), 'r') as f:
            scaler.n_samples_seen_ = int(f.read())
        return scaler
    return None



def check_sumstats(forest, log, scaler, threshold=5, mode='w'):
    sumstats = forest2sumstat_df(forest)
    X = sumstats.loc[:, STATS].to_numpy(dtype=float, na_value=0)
    X = scaler.transform(X)


    index = np.arange(len(X))
    with open(log, mode) as log_file:
        if mode != 'a':
            log_file.write("\tfeature\tz-score\n")
        for i in range(len(STATS)):
            mask = (X[:, i] < -threshold) | (X[:, i] > threshold)
            if np.any(mask):
                for _ in index[mask]:
                    log_file.write(f'{_}\t{STATS[i]:44s}\t{X[_, i] :.6f}\n')

def save_forests_as_sumstats(output, nwks=None, patterns=None, target_avg_brlen=TARGET_AVG_BL):
    """
    Rescale each tree given as input to have mean branch lengths of target_avg_brlen (by default 1),
    calculate its summary statistics,
    and save them into an output comma-separated table.

    :param patterns: patterns for obtaining input trees in newick format readable by glob.
    :param nwks: list of files containing input trees (potentially several per file) in newick format
    :param output: path to the output table (comma-separated)
    :param target_avg_brlen: length of the average non-zero branch in the rescaled tree
    :return: void, saves the results to the output file
    """

    def get_nwk_iterator():
        if patterns:
            for pattern in patterns:
                for nwk in iglob(pattern):
                    yield nwk
        if nwks:
            for nwk in nwks:
                yield nwk

    with get_write_handle(output, '.temp') as f:
        is_text = isinstance(f, io.TextIOBase)
        keys = None
        i = 0
        for nwk in get_nwk_iterator():
            try:
                forest = read_forest(nwk)
            except:
                continue

            for tree in forest:
                scaling_factor = rescale_forest_to_avg_brlen([tree], target_avg_length=target_avg_brlen)
                if keys is None:
                    keys = STATS
                    line = ','.join(keys)
                    line = f'{line}\n'
                    f.write(line if is_text else line.encode())

                line = ','.join(f'{v:.6f}' if v % 1 else f'{v:.0f}'
                                for v in FeatureManager.compute_features([tree], *STATS))
                line = f'{line}\n'
                f.write(line if is_text else line.encode())

                if 999 == (i % 1000):
                    print(f'saved {(i + 1):10.0f} trees/forests...')

                i += 1

    os.rename(output + '.temp', output)


if '__main__' == __name__:
    # ========PART 1: Fit a scaler to the summary statistics of the training data and save it to a file=====

    # DATA_DIR should be the path to the directory containing the training data (trees in newick format)
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets','50_150_train_merged_all')

    # TRAINING_DATA_PATTERN should be a pattern for the training data files (trees in newick format):
    # here I assumed that the training trees are in the DATA_DIR directory and have the .nwk extension
    TRAINING_DATA_PATTERN = os.path.join(DATA_DIR, "set_*","*.nwk")

    # TRAINING_DATA_ENCODED is the path to the output file where the summary statistics of the training data will be saved
    TRAINING_DATA_ENCODED = os.path.join(DATA_DIR, 'sumstats.csv.xz')

    # Below we calculate the summary statistics on the training data and save them to a compressed csv file:
    save_forests_as_sumstats(patterns=[TRAINING_DATA_PATTERN], output=TRAINING_DATA_ENCODED)


    # Below we fit a StandardScaler to the summary statistics of the training data and save it to the DATA_DIR directory:
    # this is done to calculate the mean and the variance for each summary statistic in the training data,
    # which will be used to z-score the summary statistics of new trees/forests.
    scaler = fit_scaler([TRAINING_DATA_ENCODED])
    save_scaler_numpy(scaler, DATA_DIR)
    # Essentially, after the above step, the summary statistics csv files can be deleted,
    # as the scaler contains all the information needed to z-score new summary statistics.



    # =========== PART 2: Check the summary statistics of a new tree against the training data =====
    # Once scalers fitter (the above code, to be run only once),
    # the below code can be used many times to check the summary statistics
    # of any number of new trees against the training data:

    # Replace with the tree path to the newick file you want to check.
    EMPIRICAL_TREE = os.path.join("datasets", "orov_clusters",'es1.nwk')
    LOG = EMPIRICAL_TREE.replace('.nwk', '.sumstats_log')
    tree = read_tree(EMPIRICAL_TREE)

    # Load the scaler fitted on the training data and check the summary statistics of the new tree against the training data:
    scaler = load_scaler_numpy(DATA_DIR)

    # Check the summary statistics of the new tree against the training data and save the results to a log file.
    # Threshold is set to 5 standard deviations away from the mean of the training data,
    # but can be changed to any other value.
    # If a summary statistic of the new tree is more than threshold standard deviations away from the mean of the training data,
    # it will be logged in the log file.
    check_sumstats(forest=[tree], log=LOG, scaler=scaler, threshold=5)