import torch
import torch.nn.functional as F
from tqdm import tqdm


# Set the default CUDA device to GPU 2
# torch.cuda.set_device(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)



class LossComputer:
    def __init__(self, criterion, is_robust, dataset, alpha=None, gamma=0.1, adj=None, min_var_weight=0, step_size=0.01,
                 normalize_loss=False, btl=False):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = dataset.n_groups
        self.group_counts = dataset.group_counts().cuda()
        # self.group_counts = dataset.group_counts().to(device)
        self.group_frac = self.group_counts / self.group_counts.sum()
        self.group_str = dataset.group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
            # self.adj = torch.from_numpy(adj).float().to(device)
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()
            # self.adj = torch.zeros(self.n_groups).float().to(device)
        if is_robust:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda() / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        # self.adv_probs = torch.ones(self.n_groups).to(device) / self.n_groups
        # self.exp_avg_loss = torch.zeros(self.n_groups).to(device)
        # self.exp_avg_initialized = torch.zeros(self.n_groups).byte().to(device)
        self.avg_group_gradient_norm = 0
        self.avg_group_hessian_norm= 0
        self.avg_hessian_aligned_loss = 0
        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat, 1) == y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (1 - self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights
    def compute_pytorch_hessian(self, model, x, y):
        batch_size = x.shape[0]
        for param in model.parameters():
            param.requires_grad = True

        logits = model(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        logits = logits.squeeze()
        loss = self.criterion2(logits, y.long())

        # First order gradients
        # grads = torch.autograd.grad(loss, model.linear.weight, create_graph=True)[0]
        grads = torch.autograd.grad(loss, [param for param in model.parameters() if param.requires_grad], create_graph=True, retain_graph=True, allow_unused=True)

        # Flatten all gradients to a single vector (for a full Hessian)
        grad_vector = torch.cat([grad.view(-1) for grad in grads  if grad is not None])

        # Initialize the Hessian matrix
        hessian = []
        for i in tqdm(range(len(grad_vector))):
            # Compute gradients with respect to each element of the gradient vector
            # row_grads = torch.autograd.grad(grad_vector[i], [param for param in model.parameters() if param.requires_grad], create_graph=True, retain_graph=True, allow_unused=True)
            row_grads = torch.autograd.grad(grad_vector[i],
                                            [param for param in model.parameters() if param.requires_grad],
                                            create_graph=False, retain_graph=True if i < len(grad_vector) - 1 else False,
                                            allow_unused=True)
            # Flatten and append to the Hessian
            row = torch.cat([g.reshape(-1) for g in row_grads])
            hessian.append(row)
            # del row_grads, row  # Free up these variables
            # torch.cuda.empty_cache()  # Free up CUDA memory

        # Convert list of rows into a full Hessian tensor
        hessian = torch.stack(hessian)

        return hessian

    def hessian(self,x, logits):
        '''This function computes the hessian of the Cross Entropy with respect to the model parameters using the analytical form of hessian.'''
        # for params in model.parameters():
        #     params.requires_grad = True

        # logits = model(x)
        # logits = logits[0] if isinstance(logits, tuple) else logits
        prob = F.softmax(logits, dim=1).clone()[:, 1]

        if x.dim() == 1:
            x = x.view(1, -1)
        # hessian_list_class0 = [prob[i] * (1 - prob[i]) * torch.ger(x[i], x[i]) for i in range(batch_size)]
        batch_size = x.shape[0]
        hessian_list_class0 = [prob[i] * (1 - prob[i]) * torch.ger(x[i], x[i]) for i in range(batch_size)]

        hessian_w_class0 = sum(hessian_list_class0) / batch_size

        # Hessian for class 1 is just the negative of the Hessian for class 0
        hessian_w_class1 = -hessian_w_class0


        # Stacking the Hessians for both classes
        hessian_w = torch.stack([hessian_w_class0, hessian_w_class1])
        return hessian_w


    def gradient(self, x, logits, y):
        # for param in model.parameters():
        #     param.requires_grad = True

        # Compute logits and
        # probabilities
        # logits = model(x)
        # logits = logits[0] if isinstance(logits, tuple) else logits
        if logits.dim() == 1:
            p = F.softmax(logits, dim=0)
        else:
            p = F.softmax(logits, dim=1)


        y_onehot = torch.zeros_like(p)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)
        # Check if p is 1D and if so, reshape it to 2D
        if len(p.shape) == 1:
            p = p.unsqueeze(0)

        # Ensure y_onehot is 2D: (batch_size, num_classes)
        num_classes = 2  # or whatever the number of classes is in your problem
        if len(y_onehot.shape) == 1:
            y_onehot = y_onehot.unsqueeze(0)

        # Now, scatter should work without errors
        y_onehot = y_onehot.scatter(1, y.unsqueeze(1).long(), 1)


        if x.dim() == 1:
            x = x.view(1, -1)
        grad_w_class1 = torch.matmul((y_onehot[:, 1] - p[:, 1]).unsqueeze(0), x) / x.size(0)
        grad_w_class0 = torch.matmul((y_onehot[:, 0] - p[:, 0]).unsqueeze(0), x) / x.size(0)

        # Stack the gradients for both classes
        grad_w = torch.cat([grad_w_class1, grad_w_class0], dim=0)
        return grad_w

    def exact_hessian_loss(self, logits, x, y, envs_indices, grad_alpha=10e-5, hess_beta=10e-5):
        total_loss = torch.tensor(0.0, requires_grad=True)
        self.criterion2 = torch.nn.CrossEntropyLoss()
        # empty list of lentgh = self.n_groups
        env_gradients = []
        env_hessians = []
        # compute per-sample and per-group losses
        # yhat = model(x)

        # For logging purposes
        per_sample_losses = self.criterion(logits, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, envs_indices)

        group_acc, group_count = self.compute_group_avg((torch.argmax(logits, 1) == y).float(), envs_indices)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None


        # Compute the gradient and hessian for each environment
        for env_idx in range(self.n_groups):
            # model.zero_grad()
            idx = (envs_indices == env_idx).nonzero().squeeze()
            if idx.numel() == 0:
                env_gradients.append(torch.zeros(1))
                env_hessians.append(torch.zeros(1))
                continue
            elif x[idx].dim() == 1:
                yhat_env = logits[idx].view(1, -1)
            else:
                yhat_env = logits[idx]
            # Assuming the first element of the tuple is the output you need
            yhat_env = yhat_env[0] if isinstance(yhat_env, tuple) else yhat_env
            # grads = self.gradient(model, x[idx], y[idx])
            grads = self.gradient(x[idx], yhat_env, y[idx])
            # hessian = self.compute_pytorch_hessian(model, x[idx], y[idx])
            hessian = self.hessian(x[idx], yhat_env)
            env_gradients.append(grads)
            env_hessians.append(hessian)



        # Compute average gradient and hessian
        # avg_gradient = [torch.mean(torch.stack([grads[i] for grads in env_gradients]), dim=0) for i in
        #                 range(len(env_gradients[0]))]

        weight_gradients = [g[0] for g in env_gradients if g.dim() > 1]


        avg_gradient = torch.mean(torch.stack(weight_gradients), dim=0)
        filtered = [h for h in env_hessians if h.dim() > 2]
        # avg_gradient = torch.mean(torch.stack(env_gradients), dim=0)
        avg_hessian = torch.mean(torch.stack(filtered), dim=0)

        erm_loss = 0
        accum_hess_loss = 0
        accum_grad_loss = 0

        gradient_norms = torch.zeros(len(env_gradients)).cuda()
        hessian_norms = torch.zeros(len(env_hessians)).cuda()
        for env_idx, (grads, hessian) in enumerate(zip(env_gradients, env_hessians)):
            idx = (envs_indices == env_idx).nonzero().squeeze()
            if idx.numel() == 0:
                continue
            elif idx.dim() == 0:
                num_samples = 1
            else:
                num_samples = len(idx)
            yhat = logits[idx]
            loss = self.criterion2(yhat.squeeze(), y[idx].long())
            # if torch.isnan(loss):
            #     loss = 0
            #     continue
            # Compute the 2-norm of the difference between the gradient for this environment and the average gradient

            gradient_norm = torch.norm(grads[0], p=2)
            gradient_norms[env_idx] = gradient_norm
            grad_diff_norm = torch.norm(grads[0] - avg_gradient, p=2)
            # Compute the Frobenius norm of the difference between the Hessian for this environment and the average Hessian
            hessian_norm = torch.norm(hessian, p='fro')
            hessian_norms[env_idx] = hessian_norm
            hessian_diff = hessian - avg_hessian
            hessian_diff_norm = torch.norm(hessian_diff, p='fro')



            grad_loss = grad_alpha * grad_diff_norm ** 2
            hessian_loss = hess_beta * hessian_diff_norm ** 2

            total_loss = total_loss + (loss + hessian_loss + grad_loss) * num_samples/len(y)
            erm_loss = erm_loss + loss * num_samples/len(x)
            accum_grad_loss = accum_grad_loss + grad_loss * num_samples/len(x)
            accum_hess_loss = accum_hess_loss + hessian_loss * num_samples/len(x)

        # not weighted
        # total_loss = total_loss / self.n_groups
        # erm_loss = erm_loss / self.n_groups
        # accum_hess_loss = accum_hess_loss / self.n_groups
        # accum_grad_loss = accum_grad_loss / self.n_groups





        # print("Loss:", total_loss.item(), "; Hessian Reg:",  alpha * hessian_reg.item(), "; Gradient Reg:", beta * grad_reg.item())



        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights, gradient_norm=gradient_norms,
                          hessian_norm=hessian_norms, hessian_aligned_loss = total_loss)

        return total_loss, erm_loss, accum_hess_loss, accum_grad_loss

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        # group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_group_gradient_norm = torch.zeros(self.n_groups).cuda()
        self.avg_group_hessian_norm = torch.zeros(self.n_groups).cuda()

        # self.processed_data_counts = torch.zeros(self.n_groups).to(device)
        # self.update_data_counts = torch.zeros(self.n_groups).to(device)
        # self.update_batch_counts = torch.zeros(self.n_groups).to(device)
        # self.avg_group_loss = torch.zeros(self.n_groups).to(device)
        # self.avg_group_acc = torch.zeros(self.n_groups).to(device)

        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None, gradient_norm=0, hessian_norm=0, hessian_aligned_loss=0):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss


        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc
        self.avg_group_gradient_norm = prev_weight * self.avg_group_acc + curr_weight * gradient_norm
        self.avg_group_hessian_norm = prev_weight * self.avg_group_acc + curr_weight * hessian_norm


        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (1 / denom) * actual_loss
        self.avg_hessian_aligned_loss = (self.batch_count / denom) * self.avg_hessian_aligned_loss + (1 / denom) * hessian_aligned_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc


    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'avg_grad_norm_group:{idx}'] = self.avg_group_gradient_norm[idx].item()
            stats_dict[f'avg_hessian_norm_group:{idx}'] = self.avg_group_hessian_norm[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        # stats_dict['hessian_aligned_loss'] = self.avg_hessian_aligned_loss.item()
        if hasattr(self.avg_hessian_aligned_loss, 'item'):
            stats_dict['hessian_aligned_loss'] = self.avg_hessian_aligned_loss.item()
        else:
            stats_dict['hessian_aligned_loss'] = self.avg_hessian_aligned_loss
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        # logger.write(f'Hessian aligned loss: {self.avg_hessian_aligned_loss.item():.3f}  \n')
        hessian_loss_value = self.avg_hessian_aligned_loss.item() if hasattr(self.avg_hessian_aligned_loss,
                                                                             'item') else self.avg_hessian_aligned_loss
        logger.write(f'Hessian aligned loss: {hessian_loss_value:.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx] / torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n'
                f'grad norm = {self.avg_group_gradient_norm[group_idx]:.3f}\n'
                f'hessian norm = {self.avg_group_hessian_norm[group_idx]:.3f}\n')

        logger.flush()
