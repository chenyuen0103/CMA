import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
import scipy
import random
import json
import gc
from tqdm import tqdm

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class HISR():
    def __init__(self, dim_inv, fit_method='cov',hessian_approx_method = "HUT", l2_reg=0.01,
                 verbose=False, regression=False, spu_proj=False,
                 logistic_regression=False,
                 hparams=None, num_iterations=1000,
                 ):
        self.dim_inv = dim_inv
        self.fit_method = fit_method
        self.hessian_approx_method = hessian_approx_method
        self.l2_reg = l2_reg
        self.verbose = verbose
        self.regression = regression
        self.spu_proj = spu_proj
        self.logistic_regression = logistic_regression
        self.hparams = hparams
        self.num_iterations = num_iterations
        assert not regression, 'Regression is not supported yet for ISR'

    def extract_class_data(self, envs, label, ):
        data = []
        for env_idx, (env_data, env_label) in enumerate(envs):
            class_data = env_data[env_label == label] if label >= 0 else env_data
            data.append(np.array(class_data))

        return data

    def cal_Mu(self, envs_data, env_idxes=None):
        # calculate the mean of each env
        n_env = len(envs_data)
        n_dim = envs_data[0].shape[-1]
        Mu = np.zeros((n_env, n_dim))
        env_idxes = np.arange(n_env) if env_idxes is None else env_idxes
        for env_idx in env_idxes:
            env_data = envs_data[env_idx]
            Mu[env_idx] = np.mean(env_data, axis=0)
        return Mu

    def cal_Cov(self, envs_data, n_selected_envs=None, env_idxes=None):
        # calculate the covariance of each env
        n_env = len(envs_data)
        n_dim = envs_data[0].shape[-1]
        if env_idxes is None:
            if n_selected_envs is not None:
                env_idxes = np.random.choice(n_env, size=n_selected_envs, replace=False)
            else:
                env_idxes = np.arange(n_env)

        envs_data = [envs_data[i] for i in env_idxes]
        n_env = len(envs_data)
        covs = np.zeros((n_env, n_dim, n_dim))
        for env_idx, env_data in enumerate(envs_data):
            covs[env_idx] = np.cov(env_data.T)
        return covs

    def fit(self, envs, fit_method=None, n_env=None, env_idxes=None,
            extracted_class=1, fit_clf=True, regression=None, spu_proj=None,
            return_proj_mat=False, dim_spu=None):
        regression = regression if regression is not None else self.regression
        extracted_class = -1 if regression else extracted_class
        label_space = [-1] if regression else [0, 1]
        spu_proj = self.spu_proj if spu_proj is None else spu_proj
        data = self.extract_class_data(envs, label=extracted_class)
        n_env = len(envs) if n_env is None else n_env
        n_dim = data[0].shape[-1]
        self.dim = n_dim
        self.dim_spu = n_dim - self.dim_inv if dim_spu is None else dim_spu

        if fit_method is None:
            fit_method = self.fit_method
        if fit_method == 'mean':
            Mu = self.cal_Mu(data, env_idxes=env_idxes)
            P = self.fit_mean(Mu)
        elif fit_method == 'cov':
            Cov = self.cal_Cov(data, env_idxes=env_idxes)
            P = self.fit_cov(Cov)
        elif fit_method == 'cov-flag':
            cov_projs = []
            for i in range(n_env):
                for label in label_space:
                    for j in range(i + 1, n_env):
                        proj_mat = self.fit(envs, fit_method='cov', n_env=None, env_idxes=[i, j],
                                            extracted_class=label, return_proj_mat=True, fit_clf=False)
                        cov_projs.append(proj_mat)
            concat_projs = np.concatenate([proj for proj in cov_projs], axis=1)
            P = np.linalg.svd(concat_projs, full_matrices=True)[0]  # compute the flag-mean

        elif fit_method == 'mean-flag':
            # should focus on the spurious dims, and find the rest dimensions at the end
            mean_projs = []
            for label in [0, 1]:
                proj_mat = self.fit(envs, fit_method='mean', n_env=None, extracted_class=label, return_proj_mat=True,
                                    fit_clf=False, spu_proj=True, dim_spu=min(n_env - 1, n_dim - self.dim_inv))
                mean_projs.append(proj_mat)
                # print("proj_mat.shape", proj_mat.shape)
            concat_projs = np.concatenate([proj for proj in mean_projs], axis=1)
            # print("concat_projs.shape", concat_projs.shape)
            P = np.linalg.svd(concat_projs, full_matrices=True)[0]  # compute the flag-mean
            # print("P.shape", P.shape)
            P = P[:, np.flip(np.arange(P.shape[1]))]
            # print("P.shape", P.shape)
            self.dim_spu = n_dim - self.dim_inv

        else:
            raise ValueError(f'fit_method = {fit_method} is not supported')

        self.P = P  # projection matrix
        if spu_proj:
            proj_mat = P[:, -self.dim_spu:]
        else:
            proj_mat = P[:, :self.dim_inv]
        self.proj_mat = proj_mat

        if fit_clf:
            self.clf = self.fit_subspace_clf(envs, proj_mat, spu_proj=spu_proj)
        if return_proj_mat:
            return proj_mat

    def fit_cov(self, covs):

        d = len(covs[0])
        E = len(covs)

        pos_coefs = np.ones(E // 2)
        neg_coefs = -1 * np.ones(E - E // 2)
        coefs = np.concatenate([pos_coefs, neg_coefs])
        coefs -= np.mean(coefs)
        np.random.shuffle(coefs)

        Cov = np.zeros((d, d))
        for i in range(len(coefs)):
            Cov += coefs[i] * covs[i]
        eigenvals, P = np.linalg.eigh(Cov)
        inv_order = np.argsort(np.abs(eigenvals))
        P[:, :] = P[:, inv_order]
        k = self.dim_inv
        self.Cov = Cov
        self.P = P
        return P

    def fit_mean(self, Mu):
        B, n_env, n_dim = Mu, len(Mu), len(Mu[0])
        E_b = np.mean(B, axis=0)
        B_zm = B - E_b
        Cov = B_zm.T @ B_zm / (len(B_zm) - 1)
        _, P = np.linalg.eigh(Cov)
        k = self.dim_inv
        self.Mu = Mu
        self.P = P

        return P

    def fit_subspace_clf(self, envs, proj_mat, spu_proj=False):
        features = []
        labels = []
        envs_indices = []
        for i, (feature, label) in enumerate(envs):
            features.append(feature)
            labels.append(label)

            # Add the current environment index for each feature-label pair in this environment
            envs_indices.extend([i] * len(feature))

        zs = np.concatenate(features, axis=0)
        ys = np.concatenate(labels, axis=0)

        if self.logistic_regression:
            if self.regression:
                clf = Ridge(alpha=self.l2_reg, max_iter=self.num_iterations)
            else:
                clf = LogisticRegression(C=1 / self.l2_reg, max_iter=self.num_iterations)
        else:
            task = 'regression' if self.regression else ''
            clf = ERM(self.dim_inv, 1, task, regression=self.regression, hparams=self.hparams,
                      num_iterations=self.num_iterations)
            self.hparams = clf.hparams

        if spu_proj:
            print(proj_mat.shape)
            proj_mat = scipy.linalg.null_space(proj_mat.T)
            self.proj_mat = proj_mat
        zs_proj = zs @ (proj_mat)
        clf.fit(zs_proj, ys, envs_indices, approx_type=self.hessian_approx_method)
        return clf

    def predict(self, x):
        return self.clf.predict(x @ self.proj_mat)

    def score(self, x, y):
        return self.clf.score(x @ self.proj_mat, y)


class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, task, hparams="default", num_iterations=10000):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.task = task
        self.num_iterations = num_iterations

        # network architecture
        self.network = torch.nn.Linear(in_features, out_features)

        # loss
        if self.task == "regression":
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.BCEWithLogitsLoss()

        # hyper-parameters
        if hparams == "default":
            self.hparams = {k: v[0] for k, v in self.HPARAMS.items()}
        elif hparams == "random":
            self.hparams = {k: v[1] for k, v in self.HPARAMS.items()}
        else:
            self.hparams = json.loads(hparams)

        # callbacks
        self.callbacks = {}
        for key in ["errors"]:
            self.callbacks[key] = {
                "train": [],
                "validation": [],
                "test": []
            }


class ERM(Model):
    def __init__(self, in_features, out_features, task, hparams="default", regression=False, num_iterations=10000):
        self.regression = regression
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10 ** random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10 ** random.uniform(-6, -2))
        self.num_iterations = num_iterations

        super().__init__(in_features, out_features, task, hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def calc_hessian_diag(self, model, loss_grad, repeat=1000):
        diag = []
        gg = torch.cat([g.flatten() for g in loss_grad])
        assert gg.requires_grad, "Gradient tensor does not require gradient"
        for _ in range(repeat):
            z = 2*torch.randint_like(gg, high=2)-1
            loss = torch.dot(gg,z)
            assert loss.requires_grad, "Surrogate loss does not require gradient"
            Hz = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(name)
            Hz = torch.cat([torch.flatten(g) for g in Hz])
            diag.append(z*Hz)
        return sum(diag)/len(diag)

    def hutchinson_loss(self, model, x_batch, y_batch, envs_indices_batch, alpha=10e-5, beta=10e-5):
        # total_loss = torch.tensor(0.0, requires_grad=True)
        total_loss = torch.tensor(0.0, requires_grad=True, device=x_batch.device)
        env_gradients = []
        env_hessian_diag = []
        for env_idx in envs_indices_batch.unique():
            model.zero_grad()

            idx = (envs_indices_batch == env_idx).nonzero().squeeze()
            loss = self.loss(model(x_batch[idx]).squeeze(), y_batch[idx])
            assert loss.requires_grad, "Original loss does not require gradient"

            # get gradient of loss with respect to parameters
            loss.backward(retain_graph=True)

            # Gradient
            # grads = self.get_grads(model)
            # grads = [param.grad.clone().detach().requires_grad_(True) for param in model.parameters()]
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            original_grads = [grad.clone().requires_grad_(True) for grad in grads]

            # flatten gradient
            flatten_grad = self._flatten_grad(grads)
            flatten_original_grad = self._flatten_grad(original_grads)
            env_gradients.append(flatten_grad)

            # ||Gradient||
            grad_norm = torch.sqrt(sum(grad.norm(2) ** 2 for grad in flatten_grad))
            grad_norm.backward(retain_graph=True)

            # Diag of Hessian, the arguments are: model, gradient of the current environment, number of mc steps to take
            Hg = self.calc_hessian_diag(model, flatten_original_grad, repeat = 150)
            env_hessian_diag.append(Hg)

        average_Hg = torch.mean(torch.stack(env_hessian_diag), dim=0)
        average_gradient = torch.mean(torch.stack(env_gradients), dim=0)


        for env_idx, (grad, hessian_diag) in enumerate(zip(env_gradients, env_hessian_diag)):
            idx = (envs_indices_batch == env_idx).nonzero().squeeze()
            loss = self.loss(model(x_batch[idx]).squeeze(), y_batch[idx])

            grad_reg = sum((grad - avg_grad).norm(2) ** 2 for grad, avg_grad in zip(grads, average_gradient))
            hg_reg = sum((hg - avg_hg).norm(2) ** 2 for hg, avg_hg in zip(hessian_diag, average_Hg))

            total_loss = total_loss + (loss + alpha * hg_reg + beta * grad_reg)

        n_unique_envs = len(envs_indices_batch.unique())
        total_loss = total_loss / n_unique_envs

        return total_loss


    def hgp_loss(self, model, x_batch, y_batch, envs_indices_batch, alpha=10e-5, beta=10e-5):
        torch.autograd.set_detect_anomaly(True)
        total_loss = torch.tensor(0.0, requires_grad=True)


        env_gradients = []
        env_hgp = []
        for env_idx in envs_indices_batch.unique():
            model.zero_grad()
            idx = (envs_indices_batch == env_idx).nonzero().squeeze()
            loss = self.loss(model(x_batch[idx]).squeeze(), y_batch[idx])
            # get gradient of loss with respect to parameters
            loss.backward(retain_graph=True)

            # Gradient
            # grads = self.get_grads(model)
            grads = [param.grad.clone().detach().requires_grad_(True) for param in model.parameters()]
            env_gradients.append(grads)

            # ||Gradient||
            grad_norm = torch.sqrt(sum(grad.norm(2) ** 2 for grad in grads))
            grad_norm.backward(retain_graph=True)

            # gradient of ||Gradient||
            grads_of_grad_norm = [param.grad.clone().detach() for param in model.parameters()]

            # Approx. hessian-gradient-product
            hessian_gradient_product = [grad_norm.clone().detach() * param for param in grads_of_grad_norm]
            env_hgp.append(hessian_gradient_product)
        # Compute average gradient and hessian
        avg_gradient = [torch.mean(torch.stack([grads[i] for grads in env_gradients]), dim=0) for i in
                        range(len(env_gradients[0]))]
        avg_hgp = [torch.mean(torch.stack([hess[i] for hess in env_hgp]), dim=0) for i in
                   range(len(env_hgp[0]))]

        for env_idx, (grads, hessian_gradient_product) in enumerate(zip(env_gradients, env_hgp)):
            idx = (envs_indices_batch == env_idx).nonzero().squeeze()
            loss = self.loss(model(x_batch[idx]).squeeze(), y_batch[idx])

            grad_reg = sum((grad - avg_grad).norm(2) ** 2 for grad, avg_grad in zip(grads, avg_gradient))
            hgp_reg = sum((hgp - avg_hgp).norm(2) ** 2 for hgp, avg_hgp in zip(hessian_gradient_product, avg_hgp))

            total_loss = total_loss + (loss + alpha * hgp_reg + beta * grad_reg)

        n_unique_envs = len(envs_indices_batch.unique())
        total_loss = total_loss / n_unique_envs

        return total_loss


    def fit(self, x, y, envs_indices, approx_type = "HGP", alpha = 10e-5, beta = 10e-5):
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)
        envs_indices = torch.Tensor(envs_indices).to(device)
        model = self.network.to(device)
        # for epoch in tqdm(range(self.num_iterations), total = self.num_iterations, desc = "Training"):
        for epoch in range(self.num_iterations):
            # Initialize Python scalar to accumulate loss from each environment
            # Initialize total_loss as a zero tensor
            # torch.autograd.set_detect_anomaly(True)
            if approx_type == "HGP":
                total_loss = self.hgp_loss(model, x, y, envs_indices, alpha, beta)
            elif approx_type == "HUT":
                total_loss = self.hutchinson_loss(model, x, y, envs_indices, alpha, beta)
            else:
                raise ValueError(f"Unknown hessian approximation type: {approx_type}")
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()  # Reset gradients to zero for the next iteration
            torch.cuda.empty_cache()
        x = x.to("cpu")
        y = y.to("cpu")
        envs_indices = envs_indices.to("cpu")
        torch.cuda.empty_cache()

    def predict(self, x):
        return self.network(x.float())

    def score(self, x, y):
        return 1 - self.network(x.float()).gt(0).float().squeeze(1).ne(y).float().mean().item()


