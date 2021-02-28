import torch
import numpy as np
from numpy.core._multiarray_umath import ndarray
from torch import LongTensor as LT
import os
import sys
# try:
#     import tangent
# except:
#     pass
import warnings
from scipy.linalg import qr
from scipy import linalg
from loss_function_grad import calculate_cosineDist_gradient

warnings.filterwarnings("ignore")
sys.path.append('./..')


# ========================================
# the columns are the older basis vectors for the qr function ; thus transpose
# ========================================
def gramSchmidt(V):
    _basis, _ = qr(V.transpose(), mode='economic')
    return _basis.transpose()


# =====================================================================
# Combine Projected GD and Confidence weighted GD
# interaction_type :: Choices 'concat', 'mul' ; linear model
# =====================================================================
class onlineGD:
    def __init__(
            self,
            num_coeff,
            emb_dim,
            _gradient_fn = None,
            interaction_type = 'mul',
            learning_rate=0.2
    ):
        self.num_coeff = num_coeff
        self.coeff_mask: ndarray = np.zeros(num_coeff)
        self.learning_rate = learning_rate
        self.prior_grad_vectors = {
            k: [] for k in range(num_coeff)
        }
        self.W_cur = None
        self.emb_dim = emb_dim

        if _gradient_fn is None:
            print('ERROR no gradient calculator passed!!!')
            exit(2)
            self.gradient_fn = None
        else:
            self.gradient_fn = _gradient_fn
        self.W_orig = None
        self.interaction_type = interaction_type
        return

    def set_original_W(self, W):
        self.W_orig = W
        self.W_cur = W
        return

    # ------------------------------------
    # list_feature_mod_idx: A list of list of indices for each sample.
    # empty list for a sample means no explanation
    # signs
    # ------------------------------------
    def update_weight(
            self,
            label=[],
            list_feature_mod_idx=[],
            X=None
    ):
        update_mask = []
        W = self.W_cur
        num_coeff = self.num_coeff

        for _label, _feat_idx in zip(label, list_feature_mod_idx):
            _mask = self.coeff_mask.copy()
            # Update on the positive labels only
            if _label == 1:
                for _f in _feat_idx:
                    _mask[_f] = 1
            update_mask.append(_mask)
        update_mask = np.array(update_mask)
        num_samples = update_mask.shape[0]
        # tiled_W shape: [ Num_samples, num_coeff, coeff_dim ]
        tiled_W = np.tile(W.reshape([1, W.shape[0], W.shape[1]]), (num_samples, 1, 1))

        # ----------------------------
        # X is raw
        # convert to X_features where X[i_j] is the pairwise interaction

        num_inp_terms = X.shape[1]
        x_split = np.split(X, num_inp_terms, axis=1)
        x_split = [_.squeeze(1) for _ in x_split]
        x_features = []

        for i in range(num_inp_terms):
            for j in range(i + 1, num_inp_terms):
                if self.interaction_type == 'mul':
                    x_ij = x_split[i] * x_split[j]
                elif self.interaction_type == 'concat':
                    x_ij = np.concatenate ( [x_split[i],x_split[j]] , axis= -1 )
                x_features.append(x_ij)
        x_features = np.stack(x_features, axis=1)

        gradient_values = np.zeros(tiled_W.shape)
        for i in range(num_samples):
            for j in range(num_coeff):
                g = self.gradient_fn(tiled_W[i][j], x_features[i][j])
                g = update_mask[i][j] * g
                gradient_values[i][j] = g

        divisor = np.sum(update_mask, axis=0)
        divisor = np.reciprocal(divisor)
        divisor = np.where(divisor == np.inf, 0, divisor)
        divisor = divisor.reshape([-1, 1])
        # --------------------------------
        # Average gradients over the batch

        avg_gradients = np.multiply(np.sum(gradient_values, axis=0), divisor)

        # =================================
        # Calculate the projection of current gradient on each of the prior gradients for the same term
        # =================================
        coeff_update_flag = np.sum(update_mask, axis=0)
        coeff_update_flag = np.where(coeff_update_flag > 0, True, False)
        cur_gradient = avg_gradients
        sum_grad_projections = []

        # ==================================
        # Create orthonormal basis if and only more than 2 prior vectors available
        # ==================================
        for i in range(num_coeff):
            _x = cur_gradient[i]
            # IF no update needed, store 0
            if not coeff_update_flag[i]:
                g_proj_i = np.zeros(_x.shape)
                sum_grad_projections.append(g_proj_i)
                continue

            # Gram Scmidt process : get the bases
            bases = np.array(self.prior_grad_vectors[i])

            if bases.shape[0] > 1:
                bases = gramSchmidt(bases)
                g_proj_i = np.zeros(_x.shape)
                # Add up sum of all projections
                for orth_base in bases:
                    _g_proj = np.dot(_x, orth_base) / np.linalg.norm(orth_base) * orth_base
                    g_proj_i += _g_proj
            else:
                g_proj_i = _x
            sum_grad_projections.append(g_proj_i)
        # --------
        # Add up the multiple projections
        sum_grad_projections = np.array(sum_grad_projections)
        final_gradient = sum_grad_projections

        # Save avg_gradients
        for i in range(num_coeff):
            if coeff_update_flag[i]:
                self.prior_grad_vectors[i].append(avg_gradients[i])

                # Update the weights
        W = W - self.learning_rate * final_gradient
        self.W_cur = W
        return final_gradient, W