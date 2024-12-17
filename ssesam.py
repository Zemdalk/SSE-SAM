import torch
from collections import defaultdict


class SAM():
    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)
        
    @torch.no_grad()
    def first_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
        
    @torch.no_grad()
    def second_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()
        
class ImbSAM:
    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    @torch.no_grad()
    def first_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grad_normal = self.state[p].get("grad_normal")
            if grad_normal is None:
                grad_normal = torch.clone(p).detach()
                self.state[p]["grad_normal"] = grad_normal
            grad_normal[...] = p.grad[...]
        self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            # eps.mul_(self.rho)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def third_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
            p.grad.add_(self.state[p]["grad_normal"])
        self.optimizer.step()
        self.optimizer.zero_grad()

class SSESAM:
    def __init__(self, optimizer, model, head_rho, tail_rho, gamma=0, total_epochs=200):
        self.optimizer = optimizer
        self.model = model
        self.rho_is = [head_rho, tail_rho]
        self.state = defaultdict(dict)
        self.total_epochs = total_epochs
        if gamma > 0 and gamma < 1:
            self.epoch_count = 0
            self.cut_epoch = total_epochs * gamma

    @torch.no_grad()
    # n_i: index of class
    def compute_and_add_epsilon(self, n_i):
        grads = []
        for p in self.model.parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16

        # Compute epsilon_i
        for p in self.model.parameters():
            if p.grad is None:
                continue

            # set/create eps
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps

            eps[...] = p.grad[...]
            eps.mul_(self.rho_is[n_i] / grad_norm)
            self.state[p]["eps"] = eps
            p.add_(eps)
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def compute_grad_sum_and_restore_p(self):
        for p in self.model.parameters():
            if p.grad is None:
                continue

            # Compute grad_sum
            grad_sum = self.state[p].get("grad_sum")
            if grad_sum is None:
                grad_sum = torch.clone(p.grad).detach()
            else:
                grad_sum[...] += p.grad[...]
            self.state[p]["grad_sum"] = grad_sum

            # Restore parameters
            p.sub_(self.state[p]["eps"])
        self.optimizer.zero_grad()

    @torch.no_grad()
    def update(self):
        for p in self.model.parameters():
            p.grad = torch.clone(self.state[p].get("grad_sum")).detach()
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.state.clear()

    @torch.no_grad()
    def update_rho(self):
        self.epoch_count += 1
        if self.cut_epoch > 0 and self.epoch_count >= self.cut_epoch:
            self.rho_is[0] = 0 # Set rho_head = 0