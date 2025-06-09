import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import OrderedDict
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from utils.train_utils import CSVBatchLogger_ISR
from utils.eval_utils import env2group, measure_group_accs, measure_group_accs_transformed
import gc
# from memory_profiler import profile
import cProfile




from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
import matplotlib.pyplot as plt

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError(f"Unknown type: {type(tensor)}")


def feature_transform(Z: np.ndarray, u: np.ndarray, d_spu: int = 1, scale: float = 0) -> np.ndarray:
    scales = np.ones(Z.shape[1])
    scales[:d_spu] = scale
    # print(Z.shape, u.shape, scales.shape)
    Z = Z @ u @ np.diag(scales)
    return Z


def check_labels(labels) -> int:
    classes = np.unique(labels)
    n_classes = len(classes)
    assert np.all(classes == np.arange(n_classes)), f"Labels must be 0, 1, 2, ..., {n_classes - 1}"
    return n_classes


def estimate_means(zs, ys, gs, n_envs, n_classes) -> dict:
    Zs, Ys = {}, {}
    for e in range(n_envs):
        Zs[e] = zs[gs == e]
        Ys[e] = ys[gs == e]
    Mus = {}
    for label in range(n_classes):
        means = {}
        for e in range(n_envs):
            means[e] = np.mean(Zs[e][Ys[e] == label], axis=0)
        Mus[label] = np.vstack(list(means.values()))
    return Mus


def estimate_covs(zs, ys, gs, n_envs, n_classes) -> dict:
    Zs, Ys = {}, {}
    for e in range(n_envs):
        Zs[e] = zs[gs == e]
        Ys[e] = ys[gs == e]
    Covs = {label: {} for label in range(n_classes)}
    for label in range(n_classes):
        for e in range(n_envs):
            Covs[label][e] = np.cov(Zs[e][Ys[e] == label].T)
    return Covs


def check_clf(clf, n_classes):
    if isinstance(clf, LogisticRegression) or isinstance(clf, RidgeClassifier) or isinstance(clf, SGDClassifier):
        if n_classes == 2:
            assert 1 <= clf.coef_.shape[0] <= 2, f"The output dim of a binary classifier must be 1 or 2"
        else:
            assert clf.coef_.shape[0] == n_classes, f"The output dimension of the classifier must be {n_classes}."
        return clf
    elif isinstance(clf, torch.nn.Linear):
        weight = clf.weight.detach().data.cpu().numpy()
        bias = clf.bias.detach().data.cpu().numpy()
    elif isinstance(clf, dict):
        weight, bias = to_numpy(clf['weight']), to_numpy(clf['bias'])
    else:
        raise ValueError(f"Unknown classifier type: {type(clf)}")

    assert weight.shape[0] == len(
        bias), f"The output dimension of weight should match bias: {weight.shape[0]} vs {len(bias)}"
    sklearn_clf = LogisticRegression()
    sklearn_clf.n_classes = n_classes
    sklearn_clf.classes_ = np.arange(n_classes)
    sklearn_clf.coef_ = weight
    sklearn_clf.intercept_ = bias
    assert sklearn_clf.coef_.shape[0] == n_classes, f"The output dimension of the classifier must be {n_classes}."

    return sklearn_clf



class HISRClassifier:
    default_clf_kwargs = dict(C=1, max_iter=10000, random_state=0)

    def __init__(self, version: str = 'mean', hessian_approx_method = "exact", pca_dim: int = -1, d_spu: int = -1, spu_scale: float = 0,
                 chosen_class=None, clf_type: str = 'LogisticRegression',  clf_kwargs: dict = None,
                 ):
        self.version = version
        self.hessian_approx_method = hessian_approx_method
        self.pca_dim = pca_dim
        self.d_spu = d_spu
        self.spu_scale = spu_scale



        self.clf_kwargs = HISRClassifier.default_clf_kwargs if clf_kwargs is None else clf_kwargs
        self.clf_type = clf_type
        self.chosen_class = chosen_class
        self.Us = {}  # stores computed projection matrices
        assert self.clf_type in ['LogisticRegression', 'RidgeClassifier', 'SGDClassifier'], \
            f"Unknown classifier type: {self.clf_type}"
        self.loss_fn = nn.CrossEntropyLoss() if self.clf_type == 'LogisticRegression' else nn.MSELoss()
        # self.loss_fn = nn.BCEWithLogitsLoss()
    def set_params(self, **params):
        for name, val in params.items():
            setattr(self, name, val)

    def fit(self, features, labels, envs, chosen_class: int = None, d_spu: int = None, given_clf=None,
            spu_scale: float = None):

        # estimate the stats (mean & cov) and fit a PCA if requested
        self.fit_data(features, labels, envs)

        if chosen_class is None:
            assert self.chosen_class is not None, "chosen_class must be specified if not given in the constructor"
            chosen_class = self.chosen_class

        if self.version == 'mean':
            self.fit_isr_mean(chosen_class=chosen_class, d_spu=d_spu)
        elif self.version == 'cov':
            self.fit_isr_cov(chosen_class=chosen_class, d_spu=d_spu)
        else:
            raise ValueError(f"Unknown ISR version: {self.version}")

        self.fit_clf(features, labels, given_clf=given_clf, spu_scale=spu_scale,alpha=10e-5, beta=10e-5)
        return self

    def fit_data(self, features, labels, envs, n_classes=None, n_envs=None):
        # estimate the mean and covariance of each class per environment
        self.n_classes = check_labels(labels)
        self.n_envs = check_labels(envs)
        self.n_groups =self.n_classes * self.n_envs
        if n_classes is not None: assert self.n_classes == n_classes
        if n_envs is not None: assert self.n_envs == n_envs

        # fit a PCA if requested
        if self.pca_dim > 0:
            self.pca = PCA(n_components=self.pca_dim).fit(features)
            features = self.pca.transform(features)
        else:
            self.pca = None
        self.means = estimate_means(features, labels, envs, self.n_envs, self.n_classes)
        self.covs = estimate_covs(features, labels, envs, self.n_envs, self.n_classes)
        return features

    def fit_isr_mean(self, chosen_class: int, d_spu: int = None):
        d_spu = self.d_spu if d_spu is None else d_spu
        assert d_spu < self.n_envs
        assert 0 <= chosen_class < self.n_classes
        # We project features into a subspace, and d_spu is the dimension of the subspace
        # Wew derive theoretically in the paper that the projection dimension of ISR-Mean
        # is at most n_envs-1
        if d_spu <= 0: self.d_spu = self.n_envs - 1

        key = ('mean', chosen_class, self.d_spu)
        if key in self.Us:
            return self.Us[key]

        # Estimate the empirical mean of each class

        # This PCA is just a helper function to obtain the projection matrix
        helper_pca = PCA(n_components=self.d_spu).fit(self.means[chosen_class])
        # The projection matrix has dimension (orig_dim, d_spu)
        # The SVD is just to pad the projection matrix with columns (the dimensions orthogonal
        # to the projection subspace) that makes the matrix a full-rank square matrix.
        U_proj = helper_pca.components_.T

        self.U = np.linalg.qr(U_proj, mode='complete')[0].real
        # The first d_spu dimensions of U correspond to spurious features, which we will
        # discard or reduce. The remaining dimensions are of the invariant feature subspace that
        # the algorithm identifies (not necessarily to be the real invariant features).

        # If we want to discard the spurious features, we can simply reduce the first d_spu
        # dimensions of U to zeros. However, this may hurt the performance of the algorithm sometimes,
        # so we can use the following strategy: rescale of the first d_spu dimensions with
        # factor between 0 and 1. This rescale factor is spu_scale that is chosen by the user.
        # print('\neig(U):', np.real(np.linalg.eigvals(self.U)))
        # print('singular vals:', s)
        self.Us[key] = self.U
        return self.U

    def fit_isr_cov(self, chosen_class: int, d_spu: int = None):
        self.d_spu = d_spu if d_spu is not None else self.d_spu
        assert self.d_spu > 0, "d_spu must be provided for ISR-Cov"
        # TODO: implement ISR-Cov for n_envs > 2
        assert self.n_envs == 2, "ISR-Cov is only implemented for binary env so far"

        key = ('cov', chosen_class, self.d_spu)
        if key in self.Us:
            return self.Us[key]

        env_pair = [0, 1]
        cov_0 = self.covs[chosen_class][env_pair[0]]
        cov_1 = self.covs[chosen_class][env_pair[1]]
        cov_diff = cov_1 - cov_0
        D = cov_diff.shape[0]

        # take square root of cov_diff such that the resulting matrix has non-negative eigenvalues
        # the largest d_spu eigenvalues correspond to the spurious feature subspace
        # we only need compute the eigenvectors of these d_spu dimensions (save computation cost)

        cov_sqr = cov_diff @ cov_diff
        w, U_proj = scipy.linalg.eigh(cov_sqr, subset_by_index=[D - self.d_spu, D - 1])
        assert w.min() >= 0
        order = np.flip(np.argsort(w).flatten())
        U_proj = U_proj[:, order]

        # trivially call SVD to fill the rest columns the (D-d_spu)-dim subspace orthogonal to the
        # spurious feature subspace

        self.U = np.linalg.svd(U_proj, full_matrices=True)[0]

        self.Us[key] = self.U

        return self.U



    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad





    def hessian_pen(self, x, logits, envs):
        env_hessians = []
        envs_indices_unique = envs.unique()
        for e in envs_indices_unique:
            idx = (envs == e).nonzero().squeeze()
            if idx.numel() == 0:
                breakpoint()
                continue

            logits_env = logits[idx]
            x_env = x[idx]
            hessian = self.hessian(x_env, logits_env)
            env_hessians.append(hessian)

        avg_hessian = torch.mean(torch.stack(env_hessians), dim=0)
        hess_pen = 0
        for env_idx, hessian in zip(envs_indices_unique, env_hessians):
            # hessian_pytorch = env_hessians_pytorch[env_idx]
            idx = (envs == env_idx).nonzero().squeeze()
            if idx.numel() == 0:
                breakpoint()
                continue

            # Compute the Frobenius norm of the difference between the Hessian for this environment and the average Hessian
            hessian_diff = hessian - avg_hessian
            hessian_reg = torch.norm(hessian_diff, p='fro') ** 2
            num_envs = len(envs_indices_unique)
            hess_pen += hessian_reg / num_envs

        return hess_pen


    def hessian(self, x, logits):
        """
        Compute the Hessian of the cross-entropy loss for n-class classification.

        Args:
            x (torch.Tensor): Input features with shape [batch_size, num_features].
            logits (torch.Tensor): Output logits with shape [batch_size, num_classes].

        Returns:
            torch.Tensor: Hessian with shape [num_classes, num_features, num_features].
        """
        batch_size, d = x.shape  # Shape: [batch_size, d]
        num_classes = logits.shape[1]  # Number of classes
        dC = num_classes * d  # Total number of parameters in the flattened gradient

        # Compute probabilities
        p = F.softmax(logits, dim=1)  # Shape: [batch_size, num_classes]
        #
        # # Compute p_k(1-p_k) for diagonal blocks and -p_k*p_l for off-diagonal blocks
        # # Diagonal part
        p_diag = p * (1 - p)  # Shape: [batch_size, num_classes]
        # # Off-diagonal part
        p_off_diag = -p.unsqueeze(2) * p.unsqueeze(1)  # Shape: [batch_size, num_classes, num_classes]
        # # Fill the diagonal part in off-diagonal tensor
        indices = torch.arange(num_classes)
        p_off_diag[:, indices, indices] = p_diag
        # Outer product of x
        X_outer = torch.einsum('bi,bj->bij', x, x)  # Shape: [batch_size, d, d]
        H = torch.einsum('bkl,bij->bklij', p_off_diag, X_outer)  # Shape: [batch_size, num_classes, num_classes, d, d]

        H = H.sum(0).reshape(dC, dC)  # Shape: [dC, dC]
        H /= batch_size
        H /= dC
        return H

    def grad_pen(self, x, logits, y, envs):
        env_gradients = []
        for e in range(self.n_envs):
            idx = (envs == e).nonzero().squeeze()
            if idx.numel() == 0:
                continue
            elif idx.dim() == 0:
                num_samples = 1
            else:
                num_samples = len(idx)
            y_env = y[idx]
            logits_env = logits[idx]
            x_env = x[idx]
            grad_w = self.gradient(x_env, logits_env, y_env)
            env_gradients.append(grad_w)

        avg_gradient = torch.mean(torch.stack(env_gradients), dim=0)

        # avg_grad_minus_grad_bar_2_sq = torch.mean(torch.stack([(grad - avg_gradient).norm(2) ** 2 for grad in env_gradients]))
        sum_grad_minus_grad_bar_2_sq = 0
        for e in range(self.n_envs):
            sum_grad_minus_grad_bar_2_sq += (env_gradients[e] - avg_gradient).norm(2) ** 2
        avg_grad_minus_grad_bar_2_sq = sum_grad_minus_grad_bar_2_sq / self.n_envs

        return avg_grad_minus_grad_bar_2_sq



    def gradient(self, x, logits, y):
        # Ensure logits are in proper shape
        p = F.softmax(logits, dim=-1)

        # Generate one-hot encoding for y
        y_onehot = torch.zeros_like(p)
        try :
            y_onehot.scatter_(1, y.long().unsqueeze(-1), 1)
        except:
            breakpoint()
            y_onehot.scatter_(1, y.unsqueeze(-1), 1)
        # y_onehot.scatter_(1, y.long().unsqueeze(-1), 1)

        # multiclasses
        grad_w = torch.matmul((p - y_onehot).T, x) / x.size(0)

        # Stack the gradients for both classes
        # grad_w2 = torch.stack([grad_w_class1, grad_w_class0], dim=0)
        # assert torch.allclose(grad_w, grad_w2), "Gradient computation is incorrect"

        dC = grad_w.shape[0] * grad_w.shape[1]
        grad_w /= dC
        return grad_w

    def calc_hessian_diag(self, model, loss_grad, repeat=1000):
        diag = []
        gg = torch.cat([g.flatten() for g in loss_grad])
        assert gg.requires_grad, "Gradient tensor does not require gradient"
        for _ in range(repeat):
            z = 2*torch.randint_like(gg, high=2)-1
            loss = torch.dot(gg,z)
            assert loss.requires_grad, "Surrogate loss does not require gradient"
            Hz = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
            Hz = torch.cat([torch.flatten(g) for g in Hz])
            diag.append(z*Hz)
        return sum(diag)/len(diag)

    def hutchinson_loss(self, logits, x, y, env_indices, alpha, beta, stats ={}):
        envs = []
        for edx in range(self.n_envs):
            idx = (env_indices == edx).nonzero().squeeze()
            if idx.numel() == 0:
                continue
            elif idx.dim() == 0:
                num_samples = 1
            else:
                num_samples = len(idx)
            y_env = y[idx]
            logits_env = logits[idx]
            env_fraction = len(idx) / len(env_indices)
            stats.update({f'env_frac:{edx}': env_fraction})
            env = {}
            env['nll'] = self.loss_fn(logits_env.squeeze(), y_env.long())
            stats.update({f'erm_loss_env:{edx}': env['nll'].item()})
            env['sadg'], env['grad'] = self.compute_sadg_penalty(logits, y)
            envs.append(env)

        train_nll = torch.stack([env['nll'] for env in envs]).mean()
        mean_grad = autograd.grad(train_nll, self.clf.parameters(), create_graph=True, retain_graph=True)
        mean_hessian = self.calc_hessian_diag(mean_grad, repeat=300)
        flatten_mean_grad = self._flatten_grad(mean_grad)

        loss = train_nll.clone()

        sadg_penalty_list = []
        all_flatten_grads = [self._flatten_grad(env['grad']) for env in envs]
        all_hessians = [self.calc_hessian_diag(env['grad'], repeat=300) for env in envs]


        # alpha beta reversed in HGP/Hugtchinson
        self.penalty_alpha = beta # for hessian
        self.penalty_beta = alpha # for gradient

        if len(envs) > 0:
            for i in range(len(all_flatten_grads)):
                sadg_penalty_list.append(
                    self.penalty_alpha * (all_hessians[i] - mean_hessian.detach()).pow(2).sum() + self.penalty_beta * (
                                all_flatten_grads[i] - flatten_mean_grad.detach()).pow(2).sum())

            N = len(sadg_penalty_list)
            sadg_penalty = torch.stack(sadg_penalty_list).sum() / len(envs)
        else:
            sadg_penalty = torch.stack([self.penalty_alpha * torch.flatten(all_hessians[0]).pow(2).sum(),
                                        self.penalty_beta * envs[0]['sadg']]).sum()


        stats['erm_loss'] = loss.item()
        loss += sadg_penalty
        stats['total_loss'] = loss.item()
        stats['hessian_loss'] = sadg_penalty.item() # abbreviation for hessian + gradient penalty

        self.update_count += 1
        return loss, sadg_penalty, stats


    def calc_hessian_diag(self, loss_grad, repeat=50):
        diag = []
        gg = torch.cat([g.flatten() for g in loss_grad])
        for _ in range(repeat):
            z = 2 * torch.randint_like(gg, high=2) - 1
            loss = torch.dot(gg, z)
            Hz = autograd.grad(loss, self.clf.parameters(), retain_graph=True, create_graph=True)
            Hz = torch.cat([torch.flatten(g) for g in Hz])
            diag.append(z * Hz)
        return sum(diag) / len(diag)

    def compute_sadg_penalty(self, logits, y):
        gradient_norm = []
        numels = []
        loss = self.loss_fn(logits.squeeze(), y.long())
        grads = autograd.grad(loss, self.clf.parameters(), create_graph=True, retain_graph=True)
        for grad in grads:
            grad = grad + 1e-16
            gradient_norm.append(torch.norm(grad, p=2))
            numels.append(torch.numel(grad))
        gradient_loss = torch.norm(torch.stack(gradient_norm), p=2)
        return gradient_loss, grads

    def hgp_loss(self, logits, x, y, env_indices, alpha, beta, stats = {}):
        envs = []
        for edx in range(self.n_envs):
            idx = (env_indices == edx).nonzero().squeeze()
            if idx.numel() == 0:
                continue
            elif idx.dim() == 0:
                num_samples = 1
            else:
                num_samples = len(idx)
            y_env = y[idx]
            logits_env = logits[idx]
            env_fraction = len(idx) / len(env_indices)
            stats.update({f'env_frac:{edx}': env_fraction})
            env = {}
            env['nll'] = self.loss_fn(logits_env.squeeze(), y_env.long())
            stats.update({f'erm_loss_env:{edx}': env['nll'].item()})
            env['sadg'], env['grad'] = self.compute_sadg_penalty(logits, y)
            envs.append(env)

        train_nll = torch.stack([env['nll'] for env in envs]).mean()
        # start  = time.time()
        mean_grad = autograd.grad(train_nll, self.clf.parameters(), create_graph=True, retain_graph=True)
        flatten_mean_grad = self._flatten_grad(mean_grad)
        norm_of_mean_grad = flatten_mean_grad.pow(2).sum().sqrt()
        norm_of_mean_grad = norm_of_mean_grad + 1e-16
        grad_of_norm_of_mean_grad = autograd.grad(norm_of_mean_grad, self.clf.parameters(), create_graph=True,
                                                  retain_graph=True)
        flatten_grad_of_norm_of_mean_grad = self._flatten_grad(grad_of_norm_of_mean_grad)
        mean_hessian_grad = torch.mul(norm_of_mean_grad, flatten_grad_of_norm_of_mean_grad)

        loss = train_nll.clone()

        sadg_penalty_list = []
        all_flatten_grads = [self._flatten_grad(env['grad']) for env in envs]

        grads_of_norm_of_grad = [
            autograd.grad(env['sadg'], self.clf.parameters(), create_graph=True, retain_graph=True) for env in
            envs]
        all_flatten_grads_of_norm_of_grad = [self._flatten_grad(grad_of_norm_of_grad) for grad_of_norm_of_grad in
                                             grads_of_norm_of_grad]

        hessian_grad = [torch.mul(envs[k]['sadg'], f_grad) for k, f_grad in
                        enumerate(all_flatten_grads_of_norm_of_grad)]


        # alpha beta reversed in HGP/Hugtchinson
        self.penalty_alpha = beta # for hessian
        self.penalty_beta = alpha # for gradient

        if len(envs) > 0:
            for i in range(len(all_flatten_grads)):
                sadg_penalty_list.append(self.penalty_alpha * (hessian_grad[i] - mean_hessian_grad.detach()).pow(
                    2).sum() + self.penalty_beta * (all_flatten_grads[i] - flatten_mean_grad.detach()).pow(2).sum())

            N = len(sadg_penalty_list)
            sadg_penalty = torch.stack(sadg_penalty_list).sum() / len(envs)
        else:
            sadg_penalty = torch.stack([self.penalty_alpha * torch.flatten(hessian_grad[0]).pow(2).sum(),
                                        self.penalty_beta * envs[0]['sadg']]).sum()



        stats['erm_loss'] = loss.item()
        loss += sadg_penalty
        stats['total_loss'] = loss.item()
        stats['hessian_loss'] = sadg_penalty.item() # abbreviation for hessian + gradient penalty

        self.update_count += 1
        return loss, sadg_penalty, stats


    def compute_pytorch_hessian(self, model, x, y):
        batch_size = x.shape[0]
        for param in model.parameters():
            param.requires_grad = True

        logits = model(x).squeeze()

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, y.long())

        # First order gradients
        grads = torch.autograd.grad(loss, model.linear.weight, create_graph=True)[0]

        hessian = []
        for i in range(grads.size(1)):
            row = torch.autograd.grad(grads[0][i], model.linear.weight, create_graph=True, retain_graph=True)[0]
            hessian.append(row)

        return torch.stack(hessian).squeeze()


    def hessian_pen_old(self, x, logits, envs):
        unique_envs = envs.unique()
        num_envs = len(unique_envs)
        H_H_f = torch.zeros(num_envs, num_envs, device=x.device)
        for e1 in range(num_envs):
            for e2 in range(e1, num_envs):
                mask1 = envs == unique_envs[e1]
                mask2 = envs == unique_envs[e2]
                x_env1 = x[mask1]
                x_env2 = x[mask2]
                logits_env1 = logits[mask1]
                logits_env2 = logits[mask2]
                p1 = F.softmax(logits_env1, dim=1)
                p2 = F.softmax(logits_env2, dim=1)
                diag1 = torch.diag_embed(p1)
                diag2 = torch.diag_embed(p2)
                off_diag1 = torch.einsum('bi,bj->bij', p1, p1)
                off_diag2 = torch.einsum('bi,bj->bij', p2, p2)
                diff1 = diag1 - off_diag1
                diff2 = diag2 - off_diag2
                prob_trace_1_2 = torch.einsum('bik,cjk->bcij', diff1, diff2).diagonal(dim1=-2, dim2=-1).sum(-1)
                X_outer1 = torch.einsum('bi,bj->bij', x_env1, x_env1)
                X_outer2 = torch.einsum('bi,bj->bij', x_env2, x_env2)
                # x_traces_1_2 = torch.einsum('bik,cjk->bcij', X_outer1, X_outer2).diagonal(dim1=-2, dim2=-1).sum(-1)
                x_traces_1_2 = torch.zeros(x_env1.shape[0], x_env2.shape[0], device=x.device)
                for i in range(x_env1.shape[0]):
                    for j in range(x_env2.shape[0]):
                        x_traces_1_2[i, j] = torch.matmul(X_outer1[i], X_outer2[j]).trace()

                H_H_f[e1, e2] = torch.sum(prob_trace_1_2 * x_traces_1_2).sum(dim=-1).sum(dim=-1) / (
                            mask1.sum() * mask2.sum())
                H_H_f[e2, e1] = H_H_f[e1, e2]

        f_norm_env = H_H_f.diagonal()
        shared_term = H_H_f.sum() / (num_envs ** 2)
        individual_term = 2 * H_H_f.sum(dim=1) / num_envs
        sum_h_minus_h_bar_sq = torch.sum(f_norm_env + shared_term - individual_term) / num_envs

        dC = x.shape[1] * logits.shape[1]

        sum_h_minus_h_bar_sq /= (dC) ** 2
        return f_norm_env, sum_h_minus_h_bar_sq

    def hessian_pen_old(self, x, logits, envs):
        p = F.softmax(logits, dim=1)
        diag = torch.diag_embed(p)
        batch_size = x.shape[0]
        off_diag = torch.einsum('bi,bj->bij', p, p)
        diff = diag - off_diag
        prob_trace = torch.einsum('bik,cjk->bcij', diff, diff).diagonal(dim1=-2, dim2=-1).sum(-1)
        X_outer = torch.einsum('bi,bj->bij', x, x)
        # x_traces = torch.einsum('bik,cjk->bcij', X_outer, X_outer).diagonal(dim1=-2, dim2=-1).sum(-1)
        x_traces = torch.zeros(batch_size, batch_size, device=x.device)
        for i in range(batch_size):
            for j in range(i, batch_size):
                x_traces[i, j] = torch.matmul(X_outer[i], X_outer[j]).trace()
                x_traces[j, i] = x_traces[i, j]

        # Create a boolean mask for each environment
        env_ids = torch.arange(self.n_envs).unsqueeze(1)  # Shape (num_envs, 1)

        # Compare env_indices with envs to create the masks tensor
        masks = env_ids == envs

        # Compute product of prob_trace and x_traces
        product_matrix = prob_trace * x_traces

        # Precompute the denominators using broadcasting
        denoms = masks.sum(1).unsqueeze(1) * masks.sum(1).unsqueeze(0)

        # Expand masks for all pairwise environment combinations using broadcasting
        mask1_expanded = masks.unsqueeze(1).unsqueeze(3)  # Shape (num_envs, 1, num_samples, 1)
        mask2_expanded = masks.unsqueeze(0).unsqueeze(2)  # Shape (1, num_envs, 1, num_samples)

        # Compute all pairwise masks by logical and
        pairwise_masks = mask1_expanded & mask2_expanded  # Shape (num_envs, num_envs, num_samples, num_samples)

        # Apply the pairwise masks to the product_matrix and sum over the last two dimensions
        masked_products = pairwise_masks * product_matrix.unsqueeze(0).unsqueeze(0)  # Broadcast product_matrix
        H_H_f = masked_products.sum(dim=-1).sum(dim=-1) / denoms

        f_norm_env = H_H_f.diagonal()

        # Compute the shared terms
        shared_term = H_H_f.sum() / (self.n_envs ** 2)

        # Compute the sum for each environment, divided by num_envs
        individual_term = 2 * H_H_f.sum(dim=1) / self.n_envs

        # Calculate the full expression in a vectorized form
        avg_h_minus_h_bar_sq = torch.mean(f_norm_env + shared_term - individual_term)

        sum_h_minus_h_bar_sq = 0
        for e in range(self.n_envs):
            sum_h_minus_h_bar_sq += f_norm_env[e] + shared_term - individual_term[e]
        avg_h_minus_h_bar_sq = sum_h_minus_h_bar_sq / self.n_envs

        # normalize by the dimmension of the hessian
        avg_h_minus_h_bar_sq /= (x.shape[1] * self.n_classes) ** 2

        return f_norm_env, avg_h_minus_h_bar_sq



    def exact_hessian_loss_old(self, logits, x, y, env_indices, alpha=10e-5, beta=10e-5, stats = {}):
        env_gradients = []
        # env_hessians = torch.zeros(self.n_envs, x.shape[1] * self.n_classes,  x.shape[1] * self.n_classes)
        env_hessians = []
        env_indices_unique = env_indices.unique()
        for env_idx in env_indices_unique:
            idx = (env_indices == env_idx).nonzero().squeeze()
            if idx.numel() == 0:
                env_gradients.append(torch.zeros(1,device=x.device))
                # env_hessians[env_idx] = torch.zeros(x.shape[1] * self.n_classes, x.shape[1] * self.n_classes)
                env_hessians.append(torch.zeros(x.shape[1] * self.n_classes, x.shape[1] * self.n_classes))
                continue
            elif x[idx].dim() == 1:
                yhat_env = logits[idx].view(1, -1)
            else:
                yhat_env = logits[idx]
            # # Gradient and Hessian Computation assumes negative log loss
            # # grads = self.gradient(model, x[idx], y[idx])
            # get grads, hessian of loss with respect to parameters, and those to be backwarded later
            # loss.backward(retain_graph=True)
            # grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            grads, hessian = 0, 0
            if alpha != 0:
                x_env = x[idx]
                y_env = y[idx]
                yhat_env = yhat_env[0] if isinstance(yhat_env, tuple) else yhat_env
                grads = self.gradient(x_env, yhat_env, y_env)

            # for checking the correctness of the gradient and hessian computation
            # hessian = self.compute_pytorch_hessian(model, x[idx], y[idx])

            if beta != 0:
                x_env = x[idx]
                yhat_env = yhat_env[0] if isinstance(yhat_env, tuple) else yhat_env
                hessian = self.hessian(x_env, yhat_env)

            env_gradients.append(grads)
            # env_hessians[env_idx] = hessian
            env_hessians.append(hessian)


        avg_gradient, avg_hessian = 0, 0
        if alpha != 0:
            weight_gradients = [g for g in env_gradients]
            avg_gradient = torch.mean(torch.stack(weight_gradients), dim=0)


        # avg_gradient = torch.mean(torch.stack(env_gradients), dim=0)
        if beta != 0:
            # avg_hessian = torch.mean(env_hessians, dim=0)
            avg_hessian = torch.mean(torch.stack(env_hessians), dim=0)

        erm_loss = 0
        hess_loss = 0
        grad_loss = 0

        for env_idx, (grads, hessian) in enumerate(zip(env_gradients, env_hessians)):
            # hessian_original = env_hessians_original[env_idx]
            idx = (env_indices == env_idx).nonzero().squeeze()
            if idx.numel() == 0:
                continue
            elif idx.dim() == 0:
                num_samples = 1
            else:
                num_samples = len(idx)
            y_env = y[idx]
            logits_env = logits[idx]
            env_fraction = len(idx) / len(env_indices)
            stats.update({f'env_frac:{env_idx}': env_fraction})
            loss = self.loss_fn(logits_env.squeeze(), y_env.long())
            # Compute the 2-norm of the difference between the gradient for this environment and the average gradient
            grad_diff_norm, grad_reg = 0, 0
            hessian_diff_norm, hessian_reg = 0, 0
            if alpha != 0:
                grad_diff_norm = torch.norm(grads - avg_gradient, p=2)
                grad_reg = alpha * grad_diff_norm ** 2
            if beta != 0:
                # Compute the Frobenius norm of the difference between the Hessian for this environment and the average Hessian
                hessian_diff = hessian - avg_hessian
                # hessian_diff_original = hessian_original - avg_hessian_original
                hessian_diff_norm = torch.norm(hessian_diff, p='fro')
                hessian_reg = beta * hessian_diff_norm ** 2

            stats.update({f'erm_loss_env:{env_idx}': loss.item()})
            stats.update({f'grad_penalty_env:{env_idx}': grad_diff_norm.item() if alpha != 0 else 0})
            stats.update({f'hessian_penalty_env:{env_idx}': hessian_diff_norm.item() if beta != 0 else 0})
            erm_loss += loss / self.n_envs
            grad_loss += grad_reg / self.n_envs
            hess_loss += hessian_reg / self.n_envs
        total_loss = erm_loss + grad_loss + hess_loss


        stats['total_loss'] = total_loss.item()
        stats['erm_loss'] = erm_loss.item()
        stats['hessian_loss'] = hess_loss.item() if beta != 0 else 0
        stats['grad_loss'] = grad_loss.item() if alpha != 0 else 0
        return total_loss, erm_loss, grad_loss, hess_loss, stats



    def exact_hessian_loss(self, logits, x, y, env_indices, alpha=10e-5, beta=10e-5, stats = {}):
        env_erm = torch.zeros(self.n_envs, device = x.device)
        for e in range(self.n_envs):
            idx = (env_indices == e).nonzero().squeeze()
            if idx.numel() == 0:
                continue
            elif idx.dim() == 0:
                num_samples = 1
            else:
                num_samples = len(idx)
            y_env = y[idx]
            logits_env = logits[idx]
            env_fraction = len(idx) / len(env_indices)
            stats.update({f'env_frac:{e}': env_fraction})
            loss = self.loss_fn(logits_env.squeeze(), y_env.long())
            env_erm[e] = loss
            stats.update({f'erm_loss_env:{e}': loss.item()})
            # Compute the 2-norm of the difference between the gradient for this environment and the average gradient

        grad_pen, hess_pen = 0, 0
        if alpha != 0:
            grad_pen = self.grad_pen(x, logits, y, env_indices)

        if beta != 0:
            hess_pen = self.hessian_pen(x, logits, env_indices)


        erm_loss = torch.mean(env_erm)
        total_loss = erm_loss + alpha * grad_pen + beta * hess_pen

        stats['total_loss'] = total_loss.item()
        stats['erm_loss'] = erm_loss.item()
        stats['hessian_loss'] = hess_pen.item() if isinstance(hess_pen, torch.Tensor) else hess_pen
        stats['grad_loss'] = grad_pen.item() if isinstance(grad_pen, torch.Tensor) else grad_pen

        return total_loss, erm_loss, grad_pen, hess_pen, stats

    def compute_fishr_penalty(self, all_logits, all_y, all_env_indices):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, all_env_indices)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y.long()).sum()
        with backpack(BatchGrad()):
            # loss.backward(
            #     inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            # )
            loss.backward(
                # inputs=list(self.classifier.parameters()),
                retain_graph=True, create_graph=True
            )


        # compute individual grads for all samples across all domains simultaneously

        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )

        # dict_grads = OrderedDict(
        #     [
        #         (name, weights.grad.clone().view(weights.grad.size(0), -1))
        #         for name, weights in self.classifier.named_parameters()
        #     ]
        # )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, env_indices):
        # grads var per domain
        # len_minibatches = [len(env_indices[env_indices == domain_id]) for domain_id in range(self.n_envs)]
        grads_var_per_domain = [{} for _ in range(self.n_envs)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id in range(self.n_envs):
                idx = (env_indices == domain_id).nonzero().squeeze()
                env_grads = _grads[idx]
                # all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.n_envs):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def l2_between_dicts(self, dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (
                torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
                torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        ).pow(2).mean()

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.n_envs)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.n_envs):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.n_envs


    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def coral_loss(self, logits, x, y, env_indices, args, stats = {}):
        self.kernel_type = 'mean_cov'
        erm_loss = 0
        penalty = 0
        nmb = self.n_envs
        #
        features = x
        classifs = logits
        targets = y
        for e in range(self.n_envs):
            idx = (env_indices == e).nonzero().squeeze()
            if idx.numel() == 0:
                continue
            elif idx.dim() == 0:
                num_samples = 1
            else:
                num_samples = len(idx)
        # for i in range(nmb):
        #     objective += F.cross_entropy(classifs[i], targets[i])
            erm_loss += F.cross_entropy(classifs[idx], targets[idx].long())

            for e2 in range(e + 1, nmb):
                idx_e2 = (env_indices == e2).nonzero().squeeze()
                penalty += self.mmd(features[idx], features[idx_e2])


        erm_loss /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        # self.optimizer.zero_grad()
        # (objective + (self.hparams['mmd_gamma'] * penalty)).backward()
        # self.optimizer.step()

        total_loss = erm_loss + args.mmd_gamma * penalty

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return total_loss, erm_loss, penalty, stats





    def fishr_loss(self, logits, x, y, env_indices, args, stats = {}):
        # all_nll = F.cross_entropy(logits, y.long())
        # all_nll2 = self.loss_fn(logits.squeeze(), y.long())
        penalty_weight, penalty = 0, 0
        env_nlls = []
        for e in range(self.n_envs):
            idx = (env_indices == e).nonzero().squeeze()
            if idx.numel() == 0:
                continue
            elif idx.dim() == 0:
                num_samples = 1
            else:
                num_samples = len(idx)
            y_env = y[idx]
            logits_env = logits[idx]
            env_nll = F.cross_entropy(logits_env, y_env.long())
            env_nlls.append(env_nll)
            env_fraction = len(idx) / len(env_indices)
            stats.update({f'env_frac:{e}': env_fraction})
            stats.update({f'erm_loss_env:{e}': env_nll.item()})

        all_nll = torch.mean(torch.stack(env_nlls))
        penalty = self.compute_fishr_penalty(logits, y, env_indices)

        if self.update_count >= args.penalty_anneal_iters:
            penalty_weight = args.lam
            if self.update_count == args.penalty_anneal_iters != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                # init optimizer
                self.optimizer = optim.SGD(self.classifier.parameters(), lr=0.001)
        self.update_count += 1
        stats.update({'ema': args.ema})
        stats.update({'anneal_iters': args.penalty_anneal_iters})
        stats.update({'fishr_penalty_weight': penalty_weight})
        stats.update({'fishr_penalty': penalty})

        objective = all_nll + penalty_weight * penalty
        stats.update({'erm_loss': all_nll.item()})
        stats.update({'total_loss': objective.item()})
        return objective, all_nll, penalty, penalty_weight, stats




    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def groupdro_loss(self, logits, x, y, env_indices, args, stats = {}):
        penalty_weight, penalty = 0, 0
        env_nlls = torch.zeros(self.n_envs).to(self.device)
        self.q = torch.ones(self.n_envs).to(self.device)
        for e in range(self.n_envs):
            idx = (env_indices == e).nonzero().squeeze()
            if idx.numel() == 0:
                continue
            elif idx.dim() == 0:
                num_samples = 1
            else:
                num_samples = len(idx)
            x_env = x[idx]
            y_env = y[idx]
            env_nll = F.cross_entropy(logits[idx], y_env.long())
            self.q[e] *= (args.groupdro_eta * env_nll.data).exp()
            env_nlls[e] = env_nll
            env_fraction = len(idx) / len(env_indices)
            stats.update({f'env_frac:{e}': env_fraction})
            stats.update({f'erm_loss_env:{e}': env_nll.item()})
        self.q /= self.q.sum()
        loss = torch.dot(env_nlls, self.q)
        stats.update({'groupdro_eta': args.groupdro_eta})
        stats.update({'total_loss': loss.item()})

        return loss, stats


    def validation_fishr_loss(self, epoch, val_x, val_y, val_envs_indices, csv_logger, args):
        self.clf.eval()
        transformed_val_x = self.transform(val_x)
        if not isinstance(transformed_val_x, torch.Tensor):
            transformed_val_x = torch.tensor(transformed_val_x).float()
        if not isinstance(val_y, torch.Tensor):
            val_y = torch.tensor(val_y).float()
        if not isinstance(val_envs_indices, torch.Tensor):
            val_envs_indices = torch.tensor(val_envs_indices).int()
        datasets = TensorDataset(transformed_val_x, val_y, val_envs_indices)
        dataloader = DataLoader(datasets, batch_size=500, shuffle=True)
        val_total_loss = 0
        val_erm_loss = 0
        val_penalty = 0
        for val_x_batch, val_y_batch, val_envs_indices_batch in dataloader:
            stats = {}
            val_x_batch, val_y_batch, val_envs_indices_batch = val_x_batch.to(self.device), val_y_batch.to(self.device), val_envs_indices_batch.to(self.device)
            val_logits = self.clf(val_x_batch)
            val_loss_batch, val_erm_loss_batch, val_penalty_batch, _, _ = self.fishr_loss(val_logits, val_x_batch, val_y_batch, val_envs_indices_batch, args, stats = stats)
            val_total_loss += val_loss_batch
            val_erm_loss += val_erm_loss_batch
            val_penalty += val_penalty_batch


        val_total_loss /= len(dataloader)
        val_erm_loss /= len(dataloader)
        val_penalty /= len(dataloader)
        group_indices = env2group(val_envs_indices, val_y, self.n_envs)
        group_accs, worst_acc, worst_group = measure_group_accs(self, val_x, val_y, group_indices,
                                                                include_avg_acc=True)

        stats['epoch'] = epoch
        stats['anneal_iters']= args.penalty_anneal_iters
        stats['total_loss'] = val_total_loss.item() if isinstance(val_penalty, torch.Tensor) else val_penalty
        stats['erm_loss'] = val_erm_loss.item() if isinstance(val_penalty, torch.Tensor) else val_penalty
        stats['fishr_penalty_weight'] = args.lam
        stats['fishr_penalty'] = val_penalty.item() if isinstance(val_penalty, torch.Tensor) else val_penalty
        stats['worst_acc'] = worst_acc
        stats['worst_group'] = worst_group
        stats.update(group_accs)
        csv_logger.log(epoch, 0, stats)
        csv_logger.flush()

    def validation_groupdro_loss(self, epoch, val_x, val_y, val_envs_indices, csv_logger, args):
        self.clf.eval()
        transformed_val_x = self.transform(val_x)
        if not isinstance(transformed_val_x, torch.Tensor):
            transformed_val_x = torch.tensor(transformed_val_x).float()
        if not isinstance(val_y, torch.Tensor):
            val_y = torch.tensor(val_y).float()
        if not isinstance(val_envs_indices, torch.Tensor):
            val_envs_indices = torch.tensor(val_envs_indices).int()
        datasets = TensorDataset(transformed_val_x, val_y, val_envs_indices)
        dataloader = DataLoader(datasets, batch_size=500, shuffle=True)
        val_total_loss = 0

        for val_x_batch, val_y_batch, val_envs_indices_batch in dataloader:
            stats = {}
            val_x_batch, val_y_batch, val_envs_indices_batch = val_x_batch.to(self.device), val_y_batch.to(self.device), val_envs_indices_batch.to(self.device)
            with torch.no_grad():
                val_logits = self.clf(val_x_batch)
                val_loss_batch, _ = self.groupdro_loss(val_logits, val_x_batch, val_y_batch, val_envs_indices_batch, args, stats = stats)
                val_total_loss += val_loss_batch

        val_total_loss /= len(dataloader)

        group_indices = env2group(val_envs_indices, val_y, self.n_envs)
        group_accs, worst_acc, worst_group = measure_group_accs(self, val_x, val_y, group_indices,
                                                                include_avg_acc=True)
        stats['epoch'] = epoch
        stats['groupdro_eta'] = args.groupdro_eta
        stats['total_loss'] = val_total_loss.item() if isinstance(val_total_loss, torch.Tensor) else val_total_loss
        stats['worst_acc'] = worst_acc
        stats['worst_group'] = worst_group

        stats.update(group_accs)
        csv_logger.log(epoch, self.update_count, stats)
        csv_logger.flush()
    def validation_coral_loss(self, epoch, val_x, val_y, val_envs_indices, csv_logger, args):
        self.clf.eval()
        transformed_val_x = self.transform(val_x)
        if not isinstance(transformed_val_x, torch.Tensor):
            transformed_val_x = torch.tensor(transformed_val_x).float()
        if not isinstance(val_y, torch.Tensor):
            val_y = torch.tensor(val_y).float()
        if not isinstance(val_envs_indices, torch.Tensor):
            val_envs_indices = torch.tensor(val_envs_indices).int()
        datasets = TensorDataset(transformed_val_x, val_y, val_envs_indices)
        dataloader = DataLoader(datasets, batch_size=500, shuffle=True)
        val_total_loss = 0
        val_erm_loss = 0
        val_penalty = 0
        for val_x_batch, val_y_batch, val_envs_indices_batch in dataloader:
            stats = {}
            val_x_batch, val_y_batch, val_envs_indices_batch = val_x_batch.to(self.device), val_y_batch.to(self.device), val_envs_indices_batch.to(self.device)
            with torch.no_grad():
                val_logits = self.clf(val_x_batch)
                val_loss_batch, val_erm_loss_batch, val_penalty_batch, _ = self.coral_loss(val_logits, val_x_batch, val_y_batch, val_envs_indices_batch, args, stats = stats)
                val_total_loss += val_loss_batch
                val_erm_loss += val_erm_loss_batch
                val_penalty += val_penalty_batch

        val_total_loss /= len(dataloader)
        val_erm_loss /= len(dataloader)
        val_penalty /= len(dataloader)

        group_indices = env2group(val_envs_indices, val_y, self.n_envs)
        group_accs, worst_acc, worst_group = measure_group_accs(self, val_x, val_y, group_indices,
                                                                include_avg_acc=True)
        stats['epoch'] = epoch
        stats['coral_mmd_gamma'] = args.mmd_gamma
        stats['total_loss'] = val_total_loss.item()
        stats['erm_loss'] = val_erm_loss.item()
        stats['coral_penalty'] = val_penalty.item() if isinstance(val_penalty, torch.Tensor) else val_penalty
        stats['worst_acc'] = worst_acc
        stats['worst_group'] = worst_group

        stats.update(group_accs)
        csv_logger.log(epoch, self.update_count, stats)
        csv_logger.flush()

    def validation_hgp_loss(self,  epoch, val_x, val_y, val_envs_indices, csv_logger, args):
        self.clf.eval()
        transformed_val_x = self.transform(val_x)
        if not isinstance(transformed_val_x, torch.Tensor):
            transformed_val_x = torch.tensor(transformed_val_x).float()
        if not isinstance(val_y, torch.Tensor):
            val_y = torch.tensor(val_y).float()
        if not isinstance(val_envs_indices, torch.Tensor):
            val_envs_indices = torch.tensor(val_envs_indices).int()
        datasets = TensorDataset(transformed_val_x, val_y, val_envs_indices)
        dataloader = DataLoader(datasets, batch_size=500, shuffle=True)
        val_total_loss = 0
        val_hess_grad_loss = 0
        for val_x_batch, val_y_batch, val_envs_indices_batch in dataloader:
            stats = {}
            val_x_batch, val_y_batch, val_envs_indices_batch = val_x_batch.to(self.device), val_y_batch.to(self.device), val_envs_indices_batch.to(self.device)
            # with torch.no_grad():
            val_logits = self.clf(val_x_batch)
            val_loss_batch, val_hess_grad_loss_batch, _ = self.hgp_loss(val_logits, val_x_batch, val_y_batch, val_envs_indices_batch, alpha=args.alpha, beta=args.beta, stats = stats)
            val_total_loss += val_loss_batch

        val_total_loss /= len(dataloader)
        val_hess_grad_loss /= len(dataloader)

        group_indices = env2group(val_envs_indices, val_y, self.n_envs)

        group_accs, worst_acc, worst_group = measure_group_accs(self, val_x, val_y, group_indices,
                                                                include_avg_acc=True)
        stats['epoch'] = epoch
        stats['grad_alpha'] = args.alpha
        stats['hess_beta'] = args.beta
        stats['total_loss'] = val_total_loss.item()
        stats['hessian_loss'] = val_hess_grad_loss.item() if isinstance(val_hess_grad_loss, torch.Tensor) else val_hess_grad_loss
        stats['worst_acc'] = worst_acc
        stats['worst_group'] = worst_group

        stats.update(group_accs)
        csv_logger.log(epoch, 0, stats)
        csv_logger.flush()
    def validation_hutchinson_loss(self,  epoch, val_x, val_y, val_envs_indices, csv_logger, args):
        self.clf.eval()
        transformed_val_x = self.transform(val_x)
        if not isinstance(transformed_val_x, torch.Tensor):
            transformed_val_x = torch.tensor(transformed_val_x).float()
        if not isinstance(val_y, torch.Tensor):
            val_y = torch.tensor(val_y).float()
        if not isinstance(val_envs_indices, torch.Tensor):
            val_envs_indices = torch.tensor(val_envs_indices).int()
        datasets = TensorDataset(transformed_val_x, val_y, val_envs_indices)
        dataloader = DataLoader(datasets, batch_size=500, shuffle=True)
        val_total_loss = 0
        val_hess_grad_loss = 0
        for val_x_batch, val_y_batch, val_envs_indices_batch in dataloader:
            stats = {}
            val_x_batch, val_y_batch, val_envs_indices_batch = val_x_batch.to(self.device), val_y_batch.to(self.device), val_envs_indices_batch.to(self.device)
            # with torch.no_grad():
            val_logits = self.clf(val_x_batch)
            val_loss_batch, val_hess_grad_loss_batch, _ = self.hutchinson_loss(val_logits, val_x_batch, val_y_batch, val_envs_indices_batch, alpha=args.alpha, beta=args.beta, stats = stats)
            val_total_loss += val_loss_batch

        val_total_loss /= len(dataloader)
        val_hess_grad_loss /= len(dataloader)

        group_indices = env2group(val_envs_indices, val_y, self.n_envs)

        group_accs, worst_acc, worst_group = measure_group_accs(self, val_x, val_y, group_indices,
                                                                include_avg_acc=True)
        stats['epoch'] = epoch
        stats['grad_alpha'] = args.alpha
        stats['hess_beta'] = args.beta
        stats['total_loss'] = val_total_loss.item()
        stats['hessian_loss'] = val_hess_grad_loss.item() if isinstance(val_hess_grad_loss, torch.Tensor) else val_hess_grad_loss
        stats['worst_acc'] = worst_acc
        stats['worst_group'] = worst_group

        stats.update(group_accs)
        csv_logger.log(epoch, 0, stats)
        csv_logger.flush()

    def validation_hessian_loss(self,  epoch, val_x, val_y, val_envs_indices, csv_logger, args):
        self.clf.eval()
        transformed_val_x = self.transform(val_x)
        if not isinstance(transformed_val_x, torch.Tensor):
            transformed_val_x = torch.tensor(transformed_val_x).float()
        if not isinstance(val_y, torch.Tensor):
            val_y = torch.tensor(val_y).float()
        if not isinstance(val_envs_indices, torch.Tensor):
            val_envs_indices = torch.tensor(val_envs_indices).int()
        datasets = TensorDataset(transformed_val_x, val_y, val_envs_indices)
        dataloader = DataLoader(datasets, batch_size=500, shuffle=True)
        val_total_loss = 0
        val_erm_loss = 0
        val_hess_loss = 0
        val_grad_loss = 0
        for val_x_batch, val_y_batch, val_envs_indices_batch in dataloader:
            stats = {}
            val_x_batch, val_y_batch, val_envs_indices_batch = val_x_batch.to(self.device), val_y_batch.to(self.device), val_envs_indices_batch.to(self.device)
            with torch.no_grad():
                val_logits = self.clf(val_x_batch)
                val_loss_batch, val_erm_loss_batch, val_hess_loss_batch, val_grad_loss_batch, _ = self.exact_hessian_loss(val_logits, val_x_batch, val_y_batch, val_envs_indices_batch, alpha=args.alpha, beta=args.beta, stats = stats)
                val_total_loss += val_loss_batch
                val_erm_loss += val_erm_loss_batch
                val_hess_loss += val_hess_loss_batch
                val_grad_loss += val_grad_loss_batch

        val_total_loss /= len(dataloader)
        val_erm_loss /= len(dataloader)
        val_hess_loss /= len(dataloader)
        val_grad_loss /= len(dataloader)

        group_indices = env2group(val_envs_indices, val_y, self.n_envs)

        group_accs, worst_acc, worst_group = measure_group_accs(self, val_x, val_y, group_indices,
                                                                include_avg_acc=True)
        stats['epoch'] = epoch
        stats['anneal_iters']= args.penalty_anneal_iters
        stats['grad_alpha'] = args.alpha
        stats['hess_beta'] = args.beta
        stats['total_loss'] = val_total_loss.item()
        stats['erm_loss'] = val_erm_loss.item()
        stats['hessian_loss'] = val_hess_loss.item() if isinstance(val_hess_loss, torch.Tensor) else val_hess_loss
        stats['grad_loss'] = val_grad_loss.item() if isinstance(val_grad_loss, torch.Tensor) else val_grad_loss
        stats['worst_acc'] = worst_acc
        stats['worst_group'] = worst_group

        stats.update(group_accs)
        csv_logger.log(epoch, 0, stats)
        csv_logger.flush()




    def fit_hessian_clf(self, x, y, envs_indices, args, approx_type = "exact", alpha = 1e-4, beta = 1e-4):
        # Create the model based on the model type
        num_iterations = self.clf_kwargs['max_iter']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_classes = len(np.unique(y))

        if self.clf_type == 'LogisticRegression':
            self.clf = self.LogisticRegression(x.shape[1], num_classes)
        elif self.clf_type == 'RidgeClassifier':
            self.clf = self.RidgeRegression(x.shape[1])
        elif self.clf_type == 'SGDClassifier':
            self.clf = self.SGDClassifier(x.shape[1])
        else:
            raise ValueError(f"Unknown model type: {self.clf_type}")
        self.clf = self.clf.to(self.device)
        self.optimizer = optim.SGD(self.clf.parameters(), lr=0.001)
        self.optimizer.zero_grad()
        if approx_type == "fishr":
            self.classifier = extend(self.clf)
            self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
            self.ema_per_domain = [
                MovingAverage(ema=args.ema, oneminusema_correction=True)
                for _ in range(self.n_envs)
            ]

        # Transform the data to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).float()
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y).float()
        if not isinstance(envs_indices, torch.Tensor):
            envs_indices = torch.tensor(envs_indices).int()

        dataset = TensorDataset(x, y, envs_indices)
        print("Starting training on", self.device)

        if approx_type in ['exact','control','fishr','coral','hgp','hutchinson']:
            dataloader = DataLoader(dataset, batch_size=500, shuffle=True)
            pbar = tqdm(range(num_iterations), desc='Hessian iter', leave=False)
            self.update_count = 0
            self.log_every = 100
            for epoch in pbar:
                for batch_idx, (x_batch, y_batch, envs_indices_batch) in enumerate(dataloader):
                    stats = {}
                    group_indices_batch = env2group(envs_indices_batch, y_batch, self.n_envs)
                    group_accs, worst_acc, worst_group = measure_group_accs_transformed(self.clf, x_batch, y_batch,
                                                                                        group_indices_batch,
                                                                                        include_avg_acc=True)
                    x_batch, y_batch, envs_indices_batch = x_batch.to(self.device), y_batch.to(self.device), envs_indices_batch.to(self.device)
                    group_count = torch.bincount(group_indices_batch.to(torch.int64), minlength=self.n_groups)
                    group_frac = group_count.float() / len(group_indices_batch)
                    env_count = torch.bincount(envs_indices_batch, minlength=self.n_envs)
                    env_frac = env_count.float() / len(envs_indices_batch)
                    logits = self.clf(x_batch)
                    stats['epoch'] = epoch
                    stats['batch'] = batch_idx
                    stats['step'] = self.update_count
                    if approx_type == "exact":
                        alpha, beta = 0, 0
                        if self.update_count >= args.penalty_anneal_iters:
                            alpha = args.alpha
                            beta = args.beta
                            if self.update_count == args.penalty_anneal_iters != 0:
                                self.optimizer = optim.SGD(self.clf.parameters(), lr=0.001)
                        stats['grad_alpha'] = alpha
                        stats['hess_beta'] = beta
                        stats['anneal_iters'] = args.penalty_anneal_iters
                        total_loss, erm_loss, grad_loss, hess_loss, stats = self.exact_hessian_loss(logits, x_batch, y_batch, envs_indices_batch, alpha, beta, stats)
                        self.update_count += 1
                    elif approx_type == "fishr":
                        total_loss, erm_loss, penalty, penalty_weight, stats = self.fishr_loss(logits, x_batch, y_batch,envs_indices_batch, args, stats = stats)
                    elif approx_type == "coral":
                        total_loss, erm_loss, penalty, stats = self.coral_loss(logits, x_batch, y_batch, envs_indices_batch, args, stats = stats)
                        stats['coral_mmd_gamma'] = args.mmd_gamma
                        self.update_count += 1
                    elif approx_type == "hgp":
                        alpha, beta = 0, 0
                        if self.update_count >= args.penalty_anneal_iters:
                            alpha = args.alpha
                            beta = args.beta
                            if self.update_count == args.penalty_anneal_iters != 0:
                                self.optimizer = optim.SGD(self.clf.parameters(), lr=0.001)
                        stats['grad_alpha'] = alpha
                        stats['hess_beta'] = beta
                        total_loss, hess_grad_pen, stats = self.hgp_loss(logits, x_batch, y_batch, envs_indices_batch, alpha, beta, stats = stats)
                    elif approx_type == "hutchinson":
                        total_loss, hess_grad_pen, stats = self.hutchinson_loss(logits, x_batch, y_batch, envs_indices_batch, alpha, beta, stats = stats)
                    elif approx_type == "groupdro":
                        total_loss, stats = self.groupdro_loss(logits, x_batch, y_batch, envs_indices_batch, args, stats = stats)
                        self.update_count += 1

                    stats.update(group_accs)
                    for group_idx in range(self.n_groups):
                        stats[f'group_count:{group_idx}'] = group_count[group_idx].item()
                        stats[f'group_frac:{group_idx}'] = group_frac[group_idx].item()
                    stats['worst_acc'] = worst_acc
                    stats['worst_group'] = worst_group
                    self.train_csv_logger.log(epoch,batch_idx,stats)
                    self.train_csv_logger.flush()
                    total_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # if self.update_count >= args.penalty_anneal_iters:
                    #     self.clf = self.clf.to('cpu')
                    #     # self.clf = model
                    #     return self.clf

                if approx_type == "fishr":
                    pbar.set_postfix(loss=total_loss.item(), erm_loss=erm_loss.item(), penalty=penalty, penalty_weight=penalty_weight)
                    self.validation_fishr_loss(epoch, self.val_x, self.val_y, self.val_envs_indices,
                                                 self.val_csv_logger, args)
                    self.validation_fishr_loss(epoch, self.test_x, self.test_y, self.test_envs_indices,
                                                 self.test_csv_logger, args)
                elif approx_type == "coral":
                    pbar.set_postfix(loss=total_loss.item(), erm_loss=erm_loss.item(), penalty=penalty)
                    self.validation_coral_loss(epoch, self.val_x, self.val_y, self.val_envs_indices,
                                                 self.val_csv_logger, args)
                    self.validation_coral_loss(epoch, self.test_x, self.test_y, self.test_envs_indices,
                                                 self.test_csv_logger, args)
                elif approx_type == "groupdro":
                    pbar.set_postfix(loss=total_loss.item())
                    self.validation_groupdro_loss(epoch, self.val_x, self.val_y, self.val_envs_indices,
                                                 self.val_csv_logger, args)
                    self.validation_groupdro_loss(epoch, self.test_x, self.test_y, self.test_envs_indices,
                                                 self.test_csv_logger, args)

                elif approx_type == "hgp":
                    pbar.set_postfix(loss=total_loss.item(), hess_grad_pen=hess_grad_pen)
                    self.validation_hgp_loss(epoch, self.val_x, self.val_y, self.val_envs_indices,
                                                 self.val_csv_logger, args)
                    self.validation_hgp_loss(epoch, self.test_x, self.test_y, self.test_envs_indices,
                                                 self.test_csv_logger, args)
                elif approx_type == "hutchinson":
                    pbar.set_postfix(loss=total_loss.item())
                    self.validation_hutchinson_loss(epoch, self.val_x, self.val_y, self.val_envs_indices,
                                                 self.val_csv_logger, args)
                    self.validation_hutchinson_loss(epoch, self.test_x, self.test_y, self.test_envs_indices,
                                                 self.test_csv_logger, args)

                else:
                    pbar.set_postfix(loss=total_loss.item(), erm_loss=erm_loss.item(),beta = beta, hess_loss=hess_loss, alpha = alpha, grad_loss=grad_loss)
                    self.validation_hessian_loss(epoch, self.val_x, self.val_y, self.val_envs_indices,self.val_csv_logger, args)
                    self.validation_hessian_loss(epoch, self.test_x, self.test_y, self.test_envs_indices, self.test_csv_logger, args)


        self.clf = self.clf.to('cpu')
        # self.clf = model
        return self.clf




    def fit_clf(self, features=None, labels=None, envs = None, args = None, given_clf=None, sample_weight=None,
                hessian_approx = None, alpha = 10e-5, beta = 10e-5, spu_scale = None, train_csv_logger = None,
        val_csv_logger = None, test_csv_logger = None,
                        val_zs= None, val_ys= None, val_es = None, val_gs = None, test_zs = None, test_ys = None, test_es = None, test_gs = None):
        self.train_csv_logger = train_csv_logger
        self.val_csv_logger = val_csv_logger
        self.test_csv_logger = test_csv_logger
        self.val_x = val_zs
        self.val_y = val_ys
        self.val_envs_indices = val_es
        self.val_groups_indices = val_gs
        self.test_x = test_zs
        self.test_y = test_ys
        self.test_envs_indices = test_es
        self.test_groups_indices = test_gs
        if not hessian_approx:
            if given_clf is None:
                assert features is not None and labels is not None
                # remove 'gradient_hyperparam' and 'hessian_hyperparam' from clf_kwargs
                self.clf_kwargs.pop('gradient_hyperparam', None)
                self.clf_kwargs.pop('hessian_hyperparam', None)
                self.clf = getattr(linear_model, self.clf_type)(**self.clf_kwargs)
                features = self.transform(features, )
                self.clf.fit(features, labels, sample_weight=sample_weight)
            else:
                self.clf = check_clf(given_clf, n_classes=self.n_classes)
                self.clf.coef_ = self.clf.coef_ @ self.U
            return self.clf
        else:
            assert features is not None and labels is not None
            features = self.transform(features, )
            return self.fit_hessian_clf(features, labels, envs, args,  approx_type=self.hessian_approx_method,
                                        alpha=alpha, beta=beta)

    def transform(self, features, ):
        if self.pca is not None:
            features = self.pca.transform(features)
        new_zs = feature_transform(features, u=self.U,
                                   d_spu=self.d_spu, scale=self.spu_scale)
        return new_zs

    def predict(self, features):
        zs = self.transform(features)

        return self.clf.predict(zs)

    def score(self, features, labels):
        zs = self.transform(features)

        return self.clf.score(zs, labels)

    def fit_transform(self, features, labels, envs, chosen_class, given_clf=None):
        self.fit(features, labels, envs, chosen_class, given_clf)
        return self.transform(features)

    # build custom module for logistic regression
    class LogisticRegression(torch.nn.Module):
        # build the constructor
        def __init__(self, n_inputs, n_outputs, bias = False):
            super().__init__()
            self.linear = torch.nn.Linear(n_inputs, n_outputs, bias = bias)

        # make predictions
        def forward(self, x):
            # Just return the logits (raw scores). Softmax will be applied in the loss function.
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x).float()
            # print(x.shape)
            return self.linear(x)

        def predict(self, x):
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

        def score(self, X, y, sample_weight=None):
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    class RidgeRegression(torch.nn.Module):
        def __init__(self, n_inputs):
            super(RidgeClassifier, self).__init__()
            self.linear = torch.nn.Linear(n_inputs, 1)

        def forward(self, x):
            y_pred = self.linear(x)
            return y_pred

        def score(self, X, y, sample_weight=None):
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    class SGDClassifier(torch.nn.Module):
        def __init__(self, n_inputs):
            super(SGDClassifier, self).__init__()
            self.linear = torch.nn.Linear(n_inputs, 1)

        def forward(self, x):
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred

        def score(self, X, y, sample_weight=None):
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data

