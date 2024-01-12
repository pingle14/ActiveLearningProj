import jax.numpy as jnp
from jax import jit, random, grad
from abc import ABC, abstractmethod
from tqdm import tqdm
import time
from functools import partial


# Estimate variance of the estimated parameters analytically
@jit
def estimate_variance(err, residual):
    # Compute the weights
    weights = 1 / err
    A = jnp.sum(weights) / (jnp.sum(weights) ** 2 - jnp.sum(weights**2))

    # Compute the weighted measurements
    weighted_measurements = weights * (residual**2 - err**2)

    # Compute the variance estimator
    variance_estimator = A * jnp.sum(weighted_measurements)

    return variance_estimator


class Strategy(ABC):
    def __init__(
        self,
        model_inference_fn,
        model_training_fn,
        generate_data,
        name="Interface",
        pool_sz=100,
        initial_sample_sz=20,
        budget=1,
        iter=100,
        true_coeff=None,
        given_key=None,
    ):
        # Model and Grad
        self.model_inference_fn = model_inference_fn
        self.model_training_fn = model_training_fn
        self.grad_f = grad(model_inference_fn)
        self.generate_data = generate_data
        self.current_params = None

        # Active Learning Strategy Properties
        self.name = name
        self.initial_sample_sz = initial_sample_sz
        self.pool_sz = pool_sz
        self.budget = budget
        self.iter = iter
        self.true_coeff = true_coeff

        # Simulation Data
        self.labeled_X = None
        self.labeled_y = None
        self.error = None
        self.given_key = given_key

    @abstractmethod
    def update_sample(self, key, X, y, error):
        return

    def choose_sample(self, key):
        X, y, error, _ = self.generate_data(
            self.initial_sample_sz if self.labeled_X is None else self.pool_sz,
            coeff=self.true_coeff,
            key=key,
        )

        if self.labeled_X is None:
            # Init self.labeled: (initial sample)
            self.labeled_X = X
            self.labeled_y = y
            self.error = error
        else:
            self.update_sample(key, X, y, error)

    def estimate_variance(self, params, y, X, err):
        residual = y - self.model_inference_fn(params, X)
        return estimate_variance(err, residual)

    def simulate(self):
        param_diffs = []
        step_keys = random.split(random.PRNGKey(self.given_key), self.iter)
        sim_start = time.perf_counter()
        for i in tqdm(range(self.iter)):
            self.choose_sample(key=step_keys[i])
            estimated_coeffs = self.model_training_fn(
                self.labeled_X, self.labeled_y
            )
            self.current_params = estimated_coeffs
            param_diffs.append(jnp.absolute(estimated_coeffs - self.true_coeff))
        sim_end = time.perf_counter()
        sim_e2e = sim_end - sim_start
        print(f"\n*** E2E Time {self.name} = {sim_e2e}")
        return (
            self.labeled_X,
            self.labeled_y,
            self.error,
            param_diffs,
        )
