import pandas as pd
import os
import sys
import numpy as np
from torch import nn
import torch
from sklearn.base import TransformerMixin
from tqdm import tqdm
from torch import FloatTensor as FT
from torch.nn import functional as F
from sklearn.preprocessing import OneHotEncoder
'''
=================================
Linear classifier
Two components:
1 .Feature interactions
2. Binary feature weights for each of the entity_ids corresponding to Companies ( Serves as whitelisting/ blaacklisting)
=================================
'''

class linearClassifier_bEF(
    nn.Module,
    TransformerMixin
):

    def __init__(
            self,
            num_domains,
            emb_dim,
            LR=0.001,
            num_epochs=250,
            batch_size=32,
            interaction_type='concat',
            force_reg=False,
            L2_reg_lambda=0.0001
    ):
        super(linearClassifier_bEF, self).__init__()
        self.interaction_type = interaction_type
        self.emb_dim = emb_dim
        self.K = int(num_domains * (num_domains - 1) / 2)
        self.num_domains = num_domains
        self.force_reg = force_reg
        self.L2_lambda = L2_reg_lambda

        if interaction_type == 'concat':
            w_data = torch.normal(mean=0, std=1, size=[self.K, emb_dim * 2])
            torch.nn.init.xavier_uniform(w_data)
            self.W = nn.parameter.Parameter(
                data=w_data
            )
        elif interaction_type == 'mul':
            w_data = torch.normal(mean=0, std=1, size=[self.K, emb_dim])
            torch.nn.init.xavier_uniform(w_data)
            self.W = nn.parameter.Parameter(
                data=w_data
            )

        self.opt = torch.optim.Adam(
            [self.W],
            lr=LR,
            weight_decay=self.L2_lambda
        )

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        return

    # -------------
    # Main function to be called when training the model
    # Fits on the interaction features only
    # -------------
    def fit(self, X, y, log_interval=100):
        self._train(X, y, log_interval=log_interval)
        return

    # ---------------------------------------------------------
    # Prediction using interaction features
    # ---------------------------------------------------------
    def predict(self, X):
        self.eval()
        return self.score_sample(X)

    def score_sample(self, X):
        return self.forward(FT(X)).cpu().data.numpy()

    # ---------------------------------------------------------
    # Prediction using interaction features + binary features
    # ---------------------------------------------------------
    def predict_bEF(self, X_binary, X_interaction):
        self.eval()
        s1 = self.score_sample(X_interaction)

        s2 = self.score_sample_bEF(X_binary)
        return s1 + s2

    # ---------------------------
    # Externally set the weights
    # ----------------------------
    def update_W(self, new_W):
        self.W.data = torch.from_numpy(new_W).float()
        return

    # -------------
    # X has shape [ N, nd, emb_dm ]
    # y has shape [N]
    # -------------
    def forward(self, x):

        _x_ = torch.chunk(x, self.num_domains, dim=1)
        _x_ = [_.squeeze(1) for _ in _x_]
        terms = []
        k = 0
        for i in range(self.num_domains):
            for j in range(i + 1, self.num_domains):
                if self.interaction_type == 'concat':
                    x1x2 = torch.cat([_x_[i], _x_[j]], axis=-1)
                else:
                    x1x2 = _x_[i] * _x_[j]
                _ij = torch.matmul(x1x2, self.W[k])
                terms.append(_ij)
                k += 1
        wx = torch.stack(terms, dim=-1)

        sum_wx = torch.sum(wx, dim=-1)
        return sum_wx

    def train_iter(self, x, y, reg=False):

        self.opt.zero_grad()
        sum_wx = self.forward(FT(x))
        # Regression style MSE loss function
        y = FT(y)
        loss = F.smooth_l1_loss(
            sum_wx,
            y,
            reduction='none'
        )

        if self.force_reg:
            l2_reg = torch.sum(torch.norm(self.W.data, dim=-1))
            loss += self.L2_lambda * l2_reg
        loss = torch.mean(loss)
        loss.backward()
        self.opt.step()
        return loss

    # ============================
    # Assume that the labels are +1, -1
    # ============================
    def _train(self, X, y, log_interval=100):
        self.train()
        # there are 2 labels
        # +1 and -1
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == -1)[0]

        bs = self.batch_size
        for e in tqdm(range(self.num_epochs)):
            p_idx = np.random.choice(pos_idx, size=bs // 2)
            n_idx = np.random.choice(neg_idx, size=bs // 2)
            _x_p = X[p_idx]
            _x_n = X[n_idx]
            _x = np.vstack([_x_p, _x_n])
            _y = np.hstack([
                np.ones([bs // 2]),
                np.ones([bs // 2]) * -1
            ])
            _loss = self.train_iter(_x, _y, reg=True)
            _loss = _loss.cpu().data.numpy().mean()
            if e % log_interval == 0:
                print('Step {} Loss {:.4f}'.format(e + 1, _loss))
        return

    # ==============================
    # Train the model on positive samples only
    # ==============================
    def fit_on_pos(self, X, y, n_epochs=None, log_interval=250):
        pos_idx = np.where(y == 1)[0]
        bs = self.batch_size
        if n_epochs is None:
            n_epochs = self.num_epochs // 10
        for e in tqdm(range(n_epochs)):
            p_idx = np.random.choice(pos_idx, size=bs)
            _x = X[p_idx]
            _y = np.ones([bs])
            _loss = self.train_iter(_x, _y)
            _loss = _loss.cpu().data.numpy().mean()
            if e % log_interval == 0:
                print('Step {} Loss {:.4f}'.format(e + 1, _loss))
        return

    def predict_score_op(self, X):
        self.eval()
        res_y = self.forward(FT(X))
        return res_y.cpu().data.numpy()

    def setup_binaryFeatures(
            self,
            domain_dims,
            binaryF_domains
    ):
        self.domain_dims = domain_dims
        self.total_entity_count = sum(domain_dims.values())
        self.domain_oneHotEncoders = []

        for dom, dim in self.domain_dims.items():
            self.domain_oneHotEncoders += [OneHotEncoder().fit(np.arange(dim).reshape([-1, 1]))]
        self.valid_binaryF_domains = np.zeros([self.num_domains])

        _tmpidx = 0
        for dom, dim in self.domain_dims.items():
            if dom in binaryF_domains:
                self.valid_binaryF_domains[_tmpidx] = 1
        _tmpidx += 1

        # -------------------------------
        # Setup the initial weights
        # This scaling factor :: 0.025 is important  - since it assigns importance.
        # -------------------------------
        self.binary_W = np.ones(self.total_entity_count) * 0.1

        # This variable stores which of the entities have been marked 1 ( as in occurring as relevant )
        # Initially everything is 0
        self.entity_flag = [np.zeros(dim) for dom, dim in domain_dims.items()]
        return

    # -----------------------
    # label_flag is 0 or 1  : per sample
    # -----------------------
    def update_binary_VarW(
            self,
            X,
            label_flag
    ):

        label_flag = np.array(label_flag).reshape(-1)
        for d_idx in range(self.num_domains):
            if self.valid_binaryF_domains[d_idx] == 1:
                _entity_idx = X[:, d_idx].reshape(-1)  # _entity_idx is a slice along column
                e_idx = []
                # For each sample check if it is labelled 1
                # Get the non zero entries

                for j in range(label_flag.shape[0]):
                    if label_flag[j] == 0:
                        continue
                    e_idx.append(_entity_idx[j])

                self.entity_flag[d_idx][e_idx] = 1

        return

    # --------------------
    # X_binary: shape [batch, num_domains]
    # --------------------
    def score_sample_bEF(self, X_binary):
        # X_binary is in the form of [a,b,..]
        # Where a ,b, c ... are the entity ids (non-serialized)
        # Convert to one hot

        X_binary_ohe = []
        X_binary = np.array(X_binary)
        for d_idx in range(self.num_domains):
            _x_bd = np.array(
                self.domain_oneHotEncoders[d_idx].transform(X_binary[:, d_idx].reshape(-1, 1)).todense()
            )
            _x_bd = _x_bd * self.valid_binaryF_domains[d_idx]
            X_binary_ohe.append(_x_bd)

        X_binary_ohe = np.concatenate(X_binary_ohe, axis=1)

        # --------------------
        # Multiply by weight
        # X_binary_ohe : shape [ batch, #total entities ]
        # binary_W : shape : [ batch, #total entities ]
        # entity_validation_flag : [ batch, #total entities ]
        # --------------------
        entity_validation_flag = np.concatenate(
            [np.array(_).reshape(1, -1) for _ in self.entity_flag], axis=1
        )

        binary_W = self.binary_W * entity_validation_flag
        wx = np.sum(X_binary_ohe * binary_W, axis=1)
        return wx

# ----------------------------------------------------------
# X1 = np.random.normal(
#     loc=-2, scale=1.5, size=[1000, 5, 16]
# )
# X2 = np.random.normal(
#     loc=2, scale=1, size=[1000, 5, 16]
# )
# X = np.vstack([X1, X2])
# y = np.hstack([np.ones(1000), np.ones(1000) * -1])
#
# print(y.shape)
# obj = linearClassifier_bEF(
#     num_domains=5,
#     emb_dim=16,
#     num_epochs=10000,
#     batch_size=256,
#     interaction_type='concat',
#     force_reg=False,
#     L2_reg_lambda=0.0025
# )
# obj.fit(X, y)
# obj.fit_on_pos(X, y, 100)
#
# print(obj.score_sample(X[:10]))
# print(y[:10])
#
# print([np.linalg.norm(w) for w in obj.W.cpu().data.numpy()])
