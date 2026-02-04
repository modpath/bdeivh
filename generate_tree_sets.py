import logging
import multiprocessing
import time
import re
import numpy as np
from treesimulator import save_forest
from treesimulator.generator import generate
from treesimulator.mtbd_models import Model

TIMEOUT = int(10 * 60)
R_VH = 'RVH'  
R_HV = 'RHV'          
U_V = 'UV'             
U_H = 'UH'              
PSI_V = 'PSIV'          
PSI_H = 'PSIH'          
RHO = 'RHO'            

def random_float(rng: np.random.Generator, min_value: float = 0, max_value: float = 1, size=1) -> float:
    return min_value + rng.random(size=None if 1 == size else size) * (max_value - min_value)

def generate_tree(params, pid, results, i, rep):
    rng = np.random.default_rng(seed=int(time.time()) + (i * rep))
    # Parameter Sampling
    rvh = random_float(rng, params.min_RVH, params.max_RVH)
    rhv = random_float(rng, params.min_RHV, params.max_RHV)
    uv = random_float(rng, params.min_UV, params.max_UV)
    uh = random_float(rng, params.min_UH, params.max_UH)
    psiv = random_float(rng, params.min_PSIV, params.max_PSIV)
    psih = random_float(rng, params.min_PSIH, params.max_PSIH)
    rho = random_float(rng, params.min_RHO, params.max_RHO)
    laVH = rvh * psiv
    laHV = rhv * psih
    model = get_model(laVH=laVH, laHV=laHV, psiv=psiv, psih=psih, rho=rho, uv=uv, uh=uh)
    tips = int(rng.integers(params.min_tips, params.max_tips + 1))
    # Simulation
    epidemic = generate([model], min_tips=tips, max_tips=tips)
    # Save exactly 10 items to match the unpacking at the end of the script
    results[pid] = epidemic.sampled_forest[0], (rvh, rhv, uv, uh, psiv, psih, rho)

def get_model(laVH, laHV, psiv, psih, rho, uv, uh) -> Model:
    states_vector = np.array(['ve','vi','he','hi'])
    la_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],  
        [0.0, 0.0, laVH, 0.0],  
        [0.0, 0.0, 0.0, 0.0],  
        [laHV, 0.0, 0.0, 0.0]])
    mu_matrix = np.array([
        [0.0, uv, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, uh],
        [0.0, 0.0, 0.0, 0.0]])
    psi_vector = np.array([0.0, psiv, 0.0, psih])
    p_vector = np.array([0.0, 0.0, 0.0, rho])
#    states_freq_vector = np.array([1,0.0,0.0,0.0])
    return Model(
        states=states_vector,
        transmission_rates=la_matrix,
        transition_rates=mu_matrix,
        removal_rates=psi_vector,
        ps=p_vector)
#        state_frequencies=states_freq_vector)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generates parameters for mtbd simulations.")
    parser.add_argument('--log', required=True, type=str)
    parser.add_argument('--nwk', required=True, type=str)    
    parser.add_argument('--min_RVH', default=0.5, type=float)
    parser.add_argument('--max_RVH', default=5.0, type=float)
    parser.add_argument('--min_RHV', default=0.5, type=float)
    parser.add_argument('--max_RHV', default=5.0, type=float)
    parser.add_argument('--min_UV', default=0.08, type=float)
    parser.add_argument('--max_UV', default=0.17, type=float)
    parser.add_argument('--min_UH', default=0.10, type=float)    
    parser.add_argument('--max_UH', default=1.00, type=float)
    parser.add_argument('--min_PSIV', default=0.05, type=float)
    parser.add_argument('--max_PSIV', default=0.10, type=float)
    parser.add_argument('--min_PSIH', default=0.20, type=float)
    parser.add_argument('--max_PSIH', default=0.50, type=float)  
    parser.add_argument('--min_RHO', default=0.001, type=float)
    parser.add_argument('--max_RHO', default=0.10, type=float)
    parser.add_argument('--min_tips', default=100, type=int)
    parser.add_argument('--max_tips', default=300, type=int)
    parser.add_argument('--model', choices=['MTBD'], default='MTBD', type=str)
    parser.add_argument('--n', default=1, type=int, help="number of trees to generate")
    params = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    # return_dict = {}
    # generate_tree(params, pid=0, results=return_dict, i=1, rep=1)

    # indices = [int(_) for _ in re.findall(r'[0-9]+', params.nwk)]
    # last_idx = indices[-1] if len(indices) > 0 else 0
    # second_last_idx = indices[-2] if len(indices) > 1 else 0
    # i = (last_idx + 1) + max(0, second_last_idx) * 128    #
    multiprocessing.set_start_method('spawn', force=True)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    rep = 0
    for pid in range(params.n):
        while True:
            if pid in return_dict:
                print(f"Tree {pid} generated successfully.")
                break
            p = multiprocessing.Process(target=generate_tree, args=(params, pid, return_dict, pid, rep))
            p.start()
            p.join(TIMEOUT)
            if p.is_alive():
                print(f"Timeout on pid {pid}, rep {rep}. Restarting...")
                p.terminate()
                p.join()
            rep += 1
    forest = []
    with open(params.log, 'w+') as f:
        f.write('RVH,RHV,UV,UH,PSIV,PSIH,RHO,tips\n')
        for tree, (rvh, rhv, uv, uh, psiv, psih, rho) in return_dict.values():
            f.write(f'{rvh},{rhv},{uv},{uh},{psiv},{psih},{rho},{len(tree)}\n')
            forest.append(tree)
    save_forest(forest, params.nwk)
