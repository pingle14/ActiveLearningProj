import jax.numpy as jnp
from jax import random
from query_strategies.data_gen import generate_data
from query_strategies.strategy import Strategy


class RandomSampling(Strategy):
    def __init__(
        self,
        initial_sample_sz=20,
        pool_sz=100,
        budget=10,
        iter=10,
        true_coeff=None,
        given_key=None,
    ):
        super(RandomSampling, self).__init__(
            name="RandomSampling",
            initial_sample_sz=initial_sample_sz,
            pool_sz=pool_sz,
            budget=budget,
            iter=iter,
            true_coeff=true_coeff,
            given_key=given_key,
        )

    def choose_indices(self, key):
        return random.choice(key, a=self.pool_sz, shape=(self.budget,))

    def choose_sample(self, key):
        X, y, error, _ = generate_data(
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
            indices = self.choose_indices(key)
            sampled_feature_vectors = X[indices, :]
            self.labeled_X = (
                jnp.append(self.labeled_X, sampled_feature_vectors, axis=0)
                if self.budget > 1
                else jnp.append(
                    self.labeled_X,
                    sampled_feature_vectors.reshape((1, -1)),
                    axis=0,
                )
            )
            self.labeled_y = jnp.append(self.labeled_y, y[indices])
            self.error = jnp.append(self.error, error[indices])
