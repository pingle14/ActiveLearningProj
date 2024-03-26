import jax.numpy as jnp
from jax import vmap, jit
from query_strategies.strategy import Strategy
from datetime import datetime


# from sklearn.metrics import pairwise_distances
def pairwise_distances(X, Y):
    # Squared Euclidean distances
    XX = jnp.sum(X**2, axis=1, keepdims=True)
    YY = jnp.sum(Y**2, axis=1, keepdims=True)
    distances = XX + jnp.transpose(YY) - 2 * jnp.dot(X, Y.T)
    return distances


# Vectorize the function using vmap
pairwise_distances_vmap = vmap(pairwise_distances, in_axes=(0, None))


@jit
def calc_dist_sq(pt, cent):
    return sum(jnp.power(pt - cent, 2))


@jit
def find_nearest_cent(pt, centroids):
    # Find min dist(cent, pt)
    min_dist = jnp.min(jnp.array([calc_dist_sq(pt, _) for _ in centroids]))
    return min_dist


@jit
def update_nearest_cent(pt, current_dist, new_cent):
    new_dist = calc_dist_sq(pt, new_cent)
    return jnp.min(jnp.array([current_dist, new_dist]))


##################################################################
#                       CORE SET ALGO
# - Equivalent to minmax facility
# - GOAL: Choose b centers s.t. minmize max dist(pt, nearest center)
##################################################################
class CoreSet(Strategy):
    def __init__(
        self,
        model_inference_fn,
        model_training_fn,
        generate_data,
        initial_sample_sz=20,
        pool_sz=100,
        budget=10,
        iter=10,
        true_coeff=None,
        given_key=None,
    ):
        super(CoreSet, self).__init__(
            name="CoreSet",
            model_inference_fn=model_inference_fn,
            model_training_fn=model_training_fn,
            generate_data=generate_data,
            initial_sample_sz=initial_sample_sz,
            pool_sz=pool_sz,
            budget=budget,
            iter=iter,
            true_coeff=true_coeff,
            given_key=given_key,
        )
        self.vmap_find_nearest_cent = vmap(find_nearest_cent, in_axes=(0, None))
        self.vmap_update_nearest_cent = vmap(
            update_nearest_cent, in_axes=(0, 0, None)
        )

        # Step 2:
        self.max_num_outliers = 0
        self.nearest_dists = []

    def update_sample(self, key, X, y, error):
        self.nearest_dists = self.vmap_find_nearest_cent(X, self.labeled_X)
        collected_labels = []
        collected_errs = []
        for _ in range(self.budget):
            # Choose pt with max dist to nearest_cent
            new_pt_indx = jnp.argmax(self.nearest_dists)
            new_pt_X = (X[new_pt_indx]).reshape((1, -1))
            new_pt_y = y[new_pt_indx]
            self.labeled_X = jnp.append(self.labeled_X, new_pt_X, axis=0)
            collected_labels.append(new_pt_y)
            collected_errs.append(error[new_pt_indx])

            # Update distances
            self.nearest_dists = self.vmap_update_nearest_cent(
                X, self.nearest_dists, X[new_pt_indx]
            )

        self.labeled_y = jnp.append(
            self.labeled_y, jnp.array(collected_labels), axis=0
        )
        self.error = jnp.append(self.error, jnp.array(collected_errs), axis=0)
