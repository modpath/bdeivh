import logging
import multiprocessing
import time
import numpy as np
from treesimulator import save_forest
from treesimulator.generator import generate
from treesimulator.mtbd_models import Model

TIMEOUT = int(10 * 60)

def random_float(rng: np.random.Generator, min_value: float = 0, max_value: float = 1, size=1) -> float:
    return min_value + rng.random(size=None if 1 == size else size) * (max_value - min_value)

def generate_tree(params, pid, results, rep):
    rng = np.random.default_rng(seed=int(time.time()) + (pid * rep))
    # Parameter Sampling
    rvh = random_float(rng, params.min_RVH, params.max_RVH)
    rhv = random_float(rng, params.min_RHV, params.max_RHV)
    muv = random_float(rng, params.min_MUV, params.max_MUV)
    muh = random_float(rng, params.min_MUH, params.max_MUH)
    psiv = random_float(rng, params.min_PSIV, params.max_PSIV)
    psih = random_float(rng, params.min_PSIH, params.max_PSIH)
    rho = random_float(rng, params.min_RHO, params.max_RHO)
    laVH = rvh * psiv
    laHV = rhv * psih
    model = get_model(laVH=laVH, laHV=laHV, psiv=psiv, psih=psih, rho=rho, muv=muv, muh=muh)
    tips = int(rng.integers(params.min_tips, params.max_tips + 1))
    # Simulation
    epidemic = generate([model], min_tips=tips, max_tips=tips)

    pis = model.state_frequencies
    R = model.get_avg_R()
    d = model.get_avg_d()
    v2h_ratio = pis[:2].sum() / pis[2:].sum()
    results[pid] = epidemic.sampled_forest[0], (rvh, rhv, muv, muh, psiv, psih, rho, R, d, v2h_ratio)

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
    parser.add_argument('--min_RVH', default=0.5, type=float, help="Minimum value for the vector-to-human reproduction number.")
    parser.add_argument('--max_RVH', default=5.0, type=float, help="Maximum value for the vector-to-human reproduction number.")
    parser.add_argument('--min_RHV', default=0.5, type=float, help="Minimum value for the human-to-vector reproduction number.")
    parser.add_argument('--max_RHV', default=5.0, type=float, help="Maximum value for the human-to-vector reproduction number.")
    parser.add_argument('--min_MUV', default=0.08, type=float, help="Minimum value for the vector incubation rate.")
    parser.add_argument('--max_MUV', default=0.17, type=float, help="Maximum value for the vector incubation rate.")
    parser.add_argument('--min_MUH', default=0.10, type=float, help='Minimum value for the human incubation rate.')
    parser.add_argument('--max_MUH', default=1.00, type=float, help="Maximum value for the human incubation rate.")
    parser.add_argument('--min_PSIV', default=0.05, type=float, help="Minimum value for the vector removal rate.")
    parser.add_argument('--max_PSIV', default=0.10, type=float, help="Maximum value for the vector removal rate.")
    parser.add_argument('--min_PSIH', default=0.20, type=float, help="Minimum value for the human removal rate.")
    parser.add_argument('--max_PSIH', default=0.50, type=float, help="Maximum value for the human removal rate.")
    parser.add_argument('--min_RHO', default=0.001, type=float, help="Minimum value for the human sampling probability.")
    parser.add_argument('--max_RHO', default=0.10, type=float, help="Maximum value for the human sampling probability.")
    parser.add_argument('--min_tips', default=100, type=int, help="Minimum number of sampled tips for each generated tree.")
    parser.add_argument('--max_tips', default=300, type=int, help="Maximum number of sampled tips for each generated tree.")
    parser.add_argument('--n', default=1, type=int, help="Number of trees to generate")
    params = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

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
        f.write('Rvh,Rhv,Rhh,R,d_inc_v,d_inc_h,d_inf_v,d_inf_h,d_v,d_h,d,p,v2h_ratio,tips\n')
        for tree, (rvh, rhv, muv, muh, psiv, psih, rho, R, d, v2h_ratio) in return_dict.values():
            d_inc_v = 1 / muv
            d_inc_h = 1 / muh
            d_inf_v = 1 / psiv
            d_inf_h = 1 / psih
            d_v = d_inc_v + d_inf_v
            d_h = d_inc_h + d_inf_h
            f.write(f'{rvh},{rhv},{rhv * rvh},{R},{d_inc_v},{d_inc_h},{d_inf_v},{d_inf_h},{d_v},{d_h},{d},{rho},{v2h_ratio},{len(tree)}\n')
            forest.append(tree)
    save_forest(forest, params.nwk)
