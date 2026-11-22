import multiprocessing
import os
import time
import numpy as np
from treesimulator.generator import generate
from treesimulator.mtbd_models import Model

TIMEOUT = int(10 * 60)

def random_float(rng: np.random.Generator, min_value: float = 0, max_value: float = 1, size=1) -> float:
    return min_value + rng.random(size=None if 1 == size else size) * (max_value - min_value)

def generate_tree(params, pid, results, rep):

    rng = np.random.default_rng(seed=int(time.time()) + (pid * rep))

    # Parameter Sampling

    """
    Reproductive numbers in humans and vectors (Rhh), R of vector-human (Rvh), and R human-vector (Rvh).
    The Rhh (R among humans) is assumed to fall between 0.5 and 5.0, while, the other two are allowed to assume larger values.
    """
    rhh = random_float(rng, params.min_RHH, params.max_RHH)
    rvh = random_float(rng, 0.5, 2 * rhh)
    rhv = rhh / rvh

    """
    Total Duration of the Infection in vectors.
    The transition rate and removal rate of the virus in vectors are transformed from the sampled values of the duration of each period.
    """
    d_v = random_float(rng, params.min_DV, params.max_DV)
    t_inc_v = random_float(rng, 1.0, d_v - 1.0)
    t_inf_v = d_v - t_inc_v
    muv = 1.0 / t_inc_v
    psiv = 1.0 / t_inf_v

    """
    Total Duration of the Infection in humans.
    The transition rate and removal rate of the virus in vectors are transformed from the sampled values of the duration of each period.
    """
    d_h = random_float(rng, params.min_DH, params.max_DH)
    t_inc_h = random_float(rng, 1.0, d_h - 1.0)
    t_inf_h = d_h - t_inc_h
    muh = 1.0 / t_inc_h
    psih = 1.0 / t_inf_h

    # Sampling probability
    rho = random_float(rng, params.min_RHO, params.max_RHO)

    # Transmission rates in vectors and humans
    laVH = rvh * psiv
    laHV = rhv * psih
    model = get_model(laVH=laVH, laHV=laHV, psiv=psiv, psih=psih, rho=rho, muv=muv, muh=muh)
    tips = int(rng.integers(params.min_tips, params.max_tips + 1))

    # Simulation
    n_generated = 0
    while n_generated < tips:
        epidemic = generate([model], max_tips=tips)
        if epidemic.sampled_tree:
            n_generated = len(epidemic.sampled_tree)

    pis = model.state_frequencies
    R = model.get_avg_R()
    d = model.get_avg_d()
    v2h_ratio = pis[:2].sum() / pis[2:].sum()
    results[pid] = epidemic.sampled_tree, (rvh, rhv, rhh, d_v, d_h, muv, muh, psiv, psih, rho, R, d, v2h_ratio)

def get_model(laVH, laHV, psiv, psih, rho, muv, muh) -> Model:
    states_vector = np.array(['ve','vi','he','hi'])
    la_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],  
        [0.0, 0.0, laVH, 0.0],  
        [0.0, 0.0, 0.0, 0.0],  
        [laHV, 0.0, 0.0, 0.0]])
    mu_matrix = np.array([
        [0.0, muv, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, muh],
        [0.0, 0.0, 0.0, 0.0]])
    psi_vector = np.array([0.0, psiv, 0.0, psih])
    p_vector = np.array([0.0, 0.0, 0.0, rho])
    return Model(
        states=states_vector,
        transmission_rates=la_matrix,
        transition_rates=mu_matrix,
        removal_rates=psi_vector,
        ps=p_vector)

if __name__ == "__main__":
    """
    This method generates n BDEIVH trees drawing parameters for each of them independently from user-supplied ranges,
    and saves the trees to a given newick file (--nwk), and parameter values to a given log file (--log).    
    """
    import argparse
    parser = argparse.ArgumentParser(description="Generates parameters for mtbd simulations.")
    parser.add_argument('--log', required=True, type=str, help="File where the parameter values will be saved in a comma-separated format.")
    parser.add_argument('--nwk', required=True, type=str, help="File where the generated tree(s) will be saved in newick format.")
    parser.add_argument('--min_RHH', default=1.0, type=float, help="Minimum value for the human-to-human reproduction number.")
    parser.add_argument('--max_RHH', default=5.0, type=float, help="Maximum value for the human-to-human reproduction number.")
    parser.add_argument('--min_DV', default=10.0, type=float, help="Minimum value for the duration of the infection in vectors.")
    parser.add_argument('--max_DV', default=22.0, type=float, help="Maximum value for the duration of the infection in vectors.")
    parser.add_argument('--min_DH', default=3.0, type=float, help="Minimum value for the duration of the infection in humans.")
    parser.add_argument('--max_DH', default=15.0, type=float, help="Maximum value for the duration of the infection in humans.")
    parser.add_argument('--min_RHO', default=0.01, type=float, help="Minimum value for the human sampling probability.")
    parser.add_argument('--max_RHO', default=0.50, type=float, help="Maximum value for the human sampling probability.")
    parser.add_argument('--min_tips', default=30, type=int, help="Minimum number of sampled tips for each generated tree.")
    parser.add_argument('--max_tips', default=50, type=int, help="Maximum number of sampled tips for each generated tree.")
    parser.add_argument('--n', default=1, type=int, help="Number of trees to generate")
    params = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    rep = 0
    for pid in range(params.n):
        while True:
            if pid in return_dict:
                print(f"Tree {pid} generated successfully.")
                break
            p = multiprocessing.Process(target=generate_tree, args=(params, pid, return_dict, rep))
            p.start()
            p.join(TIMEOUT)
            if p.is_alive():
                print(f"Timeout on pid {pid}, rep {rep}. Restarting...")
                p.terminate()
                p.join()
            rep += 1
    forest = []
    with open(params.log, 'w+') as f:
        f.write('Rvh,Rhv,Rhh,R,d_v,d_h,d_inc_v,d_inc_h,d_inf_v,d_inf_h,d,p,v2h_ratio,tips\n')
        for tree, (rvh, rhv, rhh, d_v, d_h, muv, muh, psiv, psih, rho, R, d, v2h_ratio) in return_dict.values():
            d_inc_v = 1 / muv
            d_inc_h = 1 / muh
            d_inf_v = 1 / psiv
            d_inf_h = 1 / psih
            f.write(f'{rvh},{rhv},{rhh},{R},{d_v},{d_h},{d_inc_v},{d_inc_h},{d_inf_v},{d_inf_h},{d},{rho},{v2h_ratio},{len(tree)}\n')
            forest.append(tree)

    os.makedirs(os.path.dirname(os.path.abspath(params.nwk)), exist_ok=True)
    with open(params.nwk, 'w+') as f:
        for tree in forest:
            f.write(tree.write(format=5, format_root_node=True))
            f.write('\n')