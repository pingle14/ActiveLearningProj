import jax.numpy as jnp
from jax import jit, random, grad
from query_strategies.data_gen import generate_data
from abc import ABC, abstractmethod
from tqdm import tqdm
import time


# functional form
@jit
def linear_model_f(params, X):
    return jnp.matmul(X, params)


# Perform linear regression using linear algebra
@jit
def linear_regression(X, y):
    X_transpose = jnp.transpose(X)
    X_transpose_X_inv = jnp.linalg.inv(jnp.matmul(X_transpose, X))
    coeff = jnp.matmul(jnp.matmul(X_transpose_X_inv, X_transpose), y)
    return coeff


def loss_fn_mse(y_trues, y_preds):
    assert y_trues.shape == y_preds.shape
    return jnp.mean(
        jnp.array(
            [
                jnp.power(y_true - y_pred, 2)
                for (y_true, y_pred) in zip(y_trues, y_preds)
            ]
        )
    )


# Estimate variance of the estimated parameters analytically
@jit
def estimate_variance(params, y, X, err):
    res = y - linear_model_f(params, X)

    # Compute the weights
    weights = 1 / err
    A = jnp.sum(weights) / (jnp.sum(weights) ** 2 - jnp.sum(weights**2))

    # Compute the weighted measurements
    weighted_measurements = weights * (res**2 - err**2)

    # Compute the variance estimator
    variance_estimator = A * jnp.sum(weighted_measurements)

    return variance_estimator


class Strategy(ABC):
    def __init__(
        self,
        true_coeff,
        name="Interface",
        model_f=linear_model_f,
        pool_sz=100,
        initial_sample_sz=20,
        budget=1,
        iter=100,
        given_key=None,
    ):
        self.grad_f = grad(model_f)
        self.name = name
        self.initial_sample_sz = initial_sample_sz
        self.pool_sz = pool_sz
        self.budget = budget
        self.iter = iter
        self.true_coeff = true_coeff
        self.given_key = given_key

        # Additional Vars to Init
        self.labeled_X = None
        self.labeled_y = None
        self.error = None
        self.current_params = None

    @abstractmethod
    def choose_sample(self, key):
        return

    def simulate(self):
        diffs = []

        step_keys = random.split(random.PRNGKey(self.given_key), self.iter)
        sim_start = time.perf_counter()
        for i in tqdm(range(self.iter)):
            self.choose_sample(key=step_keys[i])
            estimated_coeffs = linear_regression(self.labeled_X, self.labeled_y)
            if i % 5 == 0:
                # Dont re-train model every round .. do it periodically
                self.current_params = estimated_coeffs
                test_X, test_y, _, _ = generate_data(
                    self.initial_sample_sz if i == 0 else self.budget,
                    coeff=self.true_coeff,
                )
                y_preds = linear_model_f(estimated_coeffs, test_X)
            diffs.append(jnp.absolute(estimated_coeffs - self.true_coeff))  #
        sim_end = time.perf_counter()
        sim_e2e = sim_end - sim_start
        print(f"\n*** E2E Time {self.name} = {sim_e2e}")
        return (
            self.labeled_X,
            self.labeled_y,
            self.error,
            diffs,
        )
