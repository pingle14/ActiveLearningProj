import argparse
import random
from query_strategies.core_set import CoreSet
from query_strategies.adjusted_fisher import AdjustedFisher
from query_strategies.bait import BAIT
from query_strategies.random_sampling import RandomSampling
import pandas as pd
import jax.random as random
import numpy as np
from tqdm import tqdm


# python run.py -n 100 -c 2 -s 10 -p 1000 -b 1 -i 100
def experiment(
    num_rounds=10,
    num_coeffs=5,
    initial_sample_sz=20,
    pool_sz=1000,  # todo redesign w 1 pt at a time ... effective poolsz = pool sz//budget
    budget=10,
    iter_per_algo=10,
    verbose=False,
):
    sampling_algos = {"Random", "BAIT", "Fisher", "CoreSet"}
    key = random.PRNGKey(9355442)
    _, key_coeff, _, _ = random.split(key, 4)
    # TRUE_coeff = np.asarray(random.normal(key_coeff, shape=(num_coeffs,)))
    # TRUE_coeff_same = np.asarray([TRUE_coeff[0] for _ in range(num_coeffs)])
    TRUE_coeff_same = np.asarray(
        [0 if i == 0 else 1 for i in range(num_coeffs)]
    )
    if verbose:
        print(f"TRUE_COEFFS: {TRUE_coeff_same}")
    step_keys = random.split(random.PRNGKey(0), num_rounds)

    for sampling_algo in sampling_algos:
        if verbose:
            print("#" * 30)
            print(f'{" "* 10}{sampling_algo}')
            print("#" * 30)

        realization_dfs = []
        for realization in tqdm(range(num_rounds)):
            if verbose:
                print(
                    f"{sampling_algo} REALIZATION: {realization}> key: {step_keys[realization][0]}"
                )
            kwargs = {
                "initial_sample_sz": initial_sample_sz,
                "pool_sz": pool_sz,
                "budget": budget,
                "iter": iter_per_algo,
                "true_coeff": TRUE_coeff_same,
                "given_key": step_keys[realization][0],
            }
            model = (
                CoreSet(**kwargs)
                if sampling_algo == "CoreSet"
                else AdjustedFisher(**kwargs)
                if sampling_algo == "Fisher"
                else BAIT(**kwargs)
                if sampling_algo == "BAIT"
                else RandomSampling(**kwargs)
                if sampling_algo == "Random"
                else None
            )

            if model:
                (
                    _,
                    _,
                    _,
                    diffs,
                ) = model.simulate()

                realization_df = pd.DataFrame()
                realization_df["diffs"] = [np.array(_) for _ in diffs]
                realization_df.reset_index(inplace=True)
                realization_df.rename(
                    columns={"index": "Iteration"}, inplace=True
                )
                realization_dfs.append(realization_df)

        distr_df = pd.concat(realization_dfs, axis=0)
        distr_df.to_csv(f"data/{sampling_algo}_distr.csv", index=False)


# ###############################
#     num_rounds=10,
#     num_coeffs=5,
#     initial_sample_sz=20,
#     pool_sz=1000,
#     budget=10,
#     iter_per_algo=10,
# ###############################
# ------------------- RUN ---------------------
parser = argparse.ArgumentParser(
    prog="BenchMark", description="Benchamarks stuff"
)
parser.add_argument(
    "-n",
    "--numRounds",
    action="store",
    help=f"Enter number of rounds to build distribution of vars (default=10)",
    type=int,
    required=False,
    default=10,
)
parser.add_argument(
    "-c",
    "--numCoeffs",
    action="store",
    help=f"Enter number of params/coefficients in linreg model (default=5)",
    type=int,
    required=False,
    default=5,
)
parser.add_argument(
    "-s",
    "--initSampleSz",
    action="store",
    help=f"Enter initial_sample_sz (num points to sample in 0th round) (default=20)",
    type=int,
    required=False,
    default=20,
)
parser.add_argument(
    "-p",
    "--poolSz",
    action="store",
    help=f"Enter num total points collected each night (default=1000)",
    type=int,
    required=False,
    default=1000,
)
parser.add_argument(
    "-b",
    "--budget",
    action="store",
    help=f"Enter num points can sample each subsequent night (default=10)",
    type=int,
    required=False,
    default=10,
)
parser.add_argument(
    "-i",
    "--itersPerRound",
    action="store",
    help=f"Enter num iterations each algo should take to converge each round (default=10)",
    type=int,
    required=False,
    default=10,
)
parser.add_argument(
    "-v",
    "--verbose",
    help=f"Bool to print stuff or not",
    type=bool,
    required=False,
    default=False,
)

args = vars(parser.parse_args())

num_rounds = int(args["numRounds"])
num_coeffs = int(args["numCoeffs"])
initial_sample_sz = int(args["initSampleSz"])
pool_sz = int(args["poolSz"])
budget = int(args["budget"])
iter_per_algo = int(args["itersPerRound"])
verbose = bool(args["verbose"])

if verbose:
    print("*" * 42)
    print("*" + " " * 10 + f"Benching with args: {args}")
    print("*" * 42)
experiment(
    num_rounds=num_rounds,
    num_coeffs=num_coeffs,
    initial_sample_sz=initial_sample_sz,
    pool_sz=pool_sz,
    budget=budget,
    iter_per_algo=iter_per_algo,
    verbose=verbose,
)
if verbose:
    print("DONE")
