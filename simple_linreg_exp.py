import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from jax import random
from query_strategies.core_set import CoreSet
from query_strategies.adjusted_fisher import AdjustedFisher
from query_strategies.bait import BAIT
from query_strategies.random_sampling import RandomSampling
from linreg_utils.data_gen import generate_linear_data
from linreg_utils.model import linear_model, linear_regression


def generate_rand_true_coeffs():
    key = random.PRNGKey(9355442)
    _, key_coeff, _, _ = random.split(key, 4)
    random_true_coeffs = np.asarray(random.normal(key_coeff, shape=(num_coeffs,)))
    return random_true_coeffs


def new_experiment(num_iters=100, pool_sz=1000, num_coeffs=2, initial_sample_sz=10):
    sampling_algos = {"Random", "BAIT", "Fisher", "CoreSet"}
    true_coeff = np.asarray([int(i != 0) for i in range(num_coeffs)])
    step_keys = random.split(random.PRNGKey(9355442), num_iters)

    kwargs = {
        "model_inference_fn": linear_model,
        "model_training_fn": linear_regression,
        "generate_data": generate_linear_data,
        "initial_sample_sz": initial_sample_sz,
        "pool_sz": pool_sz,
        "budget": budget,
        "iter": iter_per_algo,
        "true_coeff": true_coeff,
        "given_key": random.PRNGKey(9355442),
    }

    core_set_model = CoreSet(**kwargs)
    bait_model = BAIT(**kwargs)
    adj_fisher_model = AdjustedFisher(**kwargs)
    rand_model = RandomSampling(**kwargs)

    models = [
        rand_model,
        adj_fisher_model,
    ]  # , rand_model, , core_set_model, bait_model

    for i in tqdm(range(num_iters)):
        # Generate pool
        X, y, error, _ = generate_linear_data(
            initial_sample_sz if i == 0 else pool_sz,
            coeff=true_coeff,
            key=step_keys[i],
        )

        # Each algo choose a point
        for model in models:
            model.choose_sample(step_keys[i], X, y, error)
            estimated_coeffs = model.model_training_fn(model.labeled_X, model.labeled_y)
            model.current_params = estimated_coeffs

        if i % 100 == 0:
            # Each algo records point chosen
            for model in models:
                # Update running labels
                per_realization_labels = pd.DataFrame()
                per_realization_labels["labels"] = [
                    np.array(_)
                    for _ in model.labeled_X[
                        initial_sample_sz:
                    ]  # ignore initial random labels
                ]
                per_realization_labels["realization"] = 0
                per_realization_labels.reset_index(inplace=True)
                per_realization_labels.rename(
                    columns={"index": "Iteration"}, inplace=True
                )

                labeledX_df = per_realization_labels
                labeledX_df.to_csv(f"taurus_data/{model.name}_labeled.csv", index=False)


def corrected_experiement(
    num_rounds=10,
    num_coeffs=5,
    initial_sample_sz=20,
    pool_sz=100,
    budget=10,
    iter_per_algo=10,
    verbose=False,
): ...


# python simple_linreg_exp.py -n 100 -c 2 -s 10 -p 1000 -b 1 -i 100
def experiment(
    num_rounds=10,
    num_coeffs=5,
    initial_sample_sz=20,
    pool_sz=100,
    budget=10,
    iter_per_algo=10,
    verbose=False,
):
    sampling_algos = ["Fisher", "BAIT", "CoreSet", "Random"]  #
    true_coeff = np.asarray([0 if i == 0 else 1 for i in range(num_coeffs)])
    step_keys = random.split(random.PRNGKey(0), num_rounds)

    for sampling_algo in sampling_algos:
        if verbose:
            print("#" * 30)
            print(f'{" "* 10}{sampling_algo}')
            print("#" * 30)

        realization_param_diffs = []
        realization_chosen_labels = []
        for realization in tqdm(range(num_rounds)):
            if verbose:
                print(
                    f"{sampling_algo} REALIZATION: {realization}> key: {step_keys[realization][0]}"
                )
            kwargs = {
                "model_inference_fn": linear_model,
                "model_training_fn": linear_regression,
                "generate_data": generate_linear_data,
                "initial_sample_sz": initial_sample_sz,
                "pool_sz": pool_sz,
                "budget": budget,
                "iter": iter_per_algo,
                "true_coeff": true_coeff,
                "given_key": step_keys[realization][0],
            }
            model = (
                CoreSet(**kwargs)
                if sampling_algo == "CoreSet"
                else (
                    AdjustedFisher(**kwargs)
                    if sampling_algo == "Fisher"
                    else (
                        BAIT(**kwargs)
                        if sampling_algo == "BAIT"
                        else (
                            RandomSampling(**kwargs)
                            if sampling_algo == "Random"
                            else None
                        )
                    )
                )
            )

            if model:
                (
                    labeledX,
                    _,
                    _,
                    diffs,
                ) = model.simulate()

                # Update running param diffs
                if num_rounds > 1:
                    per_realization_param_diff = pd.DataFrame()
                    per_realization_param_diff["param_diffs"] = [
                        np.array(_) for _ in diffs
                    ]
                    per_realization_param_diff.reset_index(inplace=True)
                    per_realization_param_diff.rename(
                        columns={"index": "Iteration"}, inplace=True
                    )
                    realization_param_diffs.append(per_realization_param_diff)
                else:
                    # Update running labels
                    per_realization_labels = pd.DataFrame()
                    per_realization_labels["labels"] = [
                        np.array(_)
                        for _ in labeledX[
                            initial_sample_sz:
                        ]  # ignore initial random labels
                    ]
                    per_realization_labels["realization"] = realization
                    per_realization_labels.reset_index(inplace=True)
                    per_realization_labels.rename(
                        columns={"index": "Iteration"}, inplace=True
                    )
                    realization_chosen_labels.append(per_realization_labels)

        if num_rounds > 1:
            param_diffs_df = pd.concat(realization_param_diffs, axis=0)
            param_diffs_df.to_csv(
                f"data/{sampling_algo}_param_diff_s{initial_sample_sz}_b{budget}_p{pool_sz}_n{num_rounds}_i{iter_per_algo}_c{num_coeffs}.csv",
                index=False,
            )
        else:
            labeledX_df = pd.concat(realization_chosen_labels, axis=0)
            labeledX_df.to_csv(f"data/{sampling_algo}_labeled.csv", index=False)


# ------------------- RUN ---------------------
# ###############################
# num_rounds=10,
# num_coeffs=5,
# initial_sample_sz=20,
# pool_sz=1000,
# budget=10,
# iter_per_algo=10,
# ###############################

parser = argparse.ArgumentParser(prog="BenchMark", description="Benchamarks stuff")
parser.add_argument(
    "-n",
    "--numRounds",
    action="store",
    help="Enter number of rounds to build distribution of vars (default=10)",
    type=int,
    required=False,
    default=10,
)
parser.add_argument(
    "-c",
    "--numCoeffs",
    action="store",
    help="Enter number of params/coefficients in linreg model (default=5)",
    type=int,
    required=False,
    default=5,
)
parser.add_argument(
    "-s",
    "--initSampleSz",
    action="store",
    help="Enter initial_sample_sz (num points to sample in 0th round) (default=20)",
    type=int,
    required=False,
    default=20,
)
parser.add_argument(
    "-p",
    "--poolSz",
    action="store",
    help="Enter num total points collected each night (default=1000)",
    type=int,
    required=False,
    default=1000,
)
parser.add_argument(
    "-b",
    "--budget",
    action="store",
    help="Enter num points can sample each subsequent night (default=10)",
    type=int,
    required=False,
    default=10,
)
parser.add_argument(
    "-i",
    "--itersPerRound",
    action="store",
    help="Enter num iterations each algo should take to converge each round (default=10)",
    type=int,
    required=False,
    default=10,
)
parser.add_argument(
    "-l",
    "--longExperiment",
    help="Bool to run long experiement",
    action="store_true",
    required=False,
    default=False,
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Bool to print stuff or not",
    action="store_true",
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

# if bool(args['longExperiment']):
experiment(
    num_rounds=num_rounds,
    num_coeffs=num_coeffs,
    initial_sample_sz=initial_sample_sz,
    pool_sz=pool_sz,
    budget=budget,
    iter_per_algo=iter_per_algo,
    verbose=verbose,
)
# else:
#     new_experiment(num_iters=4000)

if verbose:
    print("DONE")
