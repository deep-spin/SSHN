# author: vlad niculae and andre f. t. martins
# license: simplified bsd

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from entmax import sparsemax, entmax15, normmax_bisect 
from lpsmap import TorchFactorGraph, Budget, SequenceBudget


class Flatten(object):

    def __call__(self, tensor):
        return torch.flatten(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def entmax(Z, alpha, dim):
    """only exact cases; raise otherwise; for toy experiments"""
    if alpha == 1:
        return torch.softmax(Z, dim=dim)
    elif alpha == 1.5:
        return entmax15(Z, dim=dim)
    elif alpha == 2:
        return sparsemax(Z, dim=dim)
    else:
        raise NotImplementedError()

def SparseMAP_exactly_k(scores, k=2):
    marginals = torch.zeros_like(scores)
    for j in range(scores.shape[1]):
        fg = TorchFactorGraph()
        u = fg.variable_from(scores[:, j])
        fg.add(Budget(u, k, force_budget=True))
        fg.solve(verbose=0)
        marginals[:, j] = u.value[:]
    return marginals

def SparseMAP_sequence_exactly_k(scores, edge_score, k=2):
    n = scores.shape[0]
    transition = torch.zeros((n+1,2,2))
    transition.data[1:n, 0, 0] = edge_score
    # Only one state in the beginning and in the end for start / stop symbol.
    transition = transition.reshape(-1)[2:-2]
    marginals = torch.zeros_like(scores)
    for j in range(scores.shape[1]):
        s = torch.zeros((n, 2))
        s[:, 0] = scores[:, j]
        fg = TorchFactorGraph()
        u = fg.variable_from(s)
        fg.add(SequenceBudget(u, transition, k, force_budget=True))
        fg.solve(verbose=0)
        marginals[:, j] = u.value[:, 0]
    return marginals

class NormmaxBisectFunction(Function):
    @classmethod
    def _gp(cls, x, alpha):
        return x ** (alpha - 1)

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y ** (1 / (alpha - 1))

    @classmethod
    def _p(cls, X, alpha):
        return cls._gp_inv(torch.clamp(X, min=0), alpha)

    @classmethod
    def forward(cls, ctx, X, alpha=2, dim=-1, n_iter=50):

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)

        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        ctx.alpha = alpha
        ctx.dim = dim
        d = X.shape[dim]

        max_val, _ = X.max(dim=dim, keepdim=True)

        # Note: when alpha < 1, tau_lo > tau_hi. This still works since dm < 0.
        # With alpha >= 1, ||p||_alpha <= 1 and ||p||_alpha >= d**((1-alpha)/alpha), therefore
        # pi = (si - tau)**(1/(alpha-1)) ||p||_alpha
        # tau = si - (pi / ||p||_\alpha) ** (alpha-1)
        # tau = smax - (pmax / ||p||_\alpha) ** (alpha-1)
        # pmax / ||p||_\alpha <= 1 / (d**(1-alpha)/alpha) = d**((alpha-1)/alpha)
        # Also pmax / ||p||_\alpha <= 1 (since pmax = ||p||_inf)
        # pmax / ||p||_\alpha >= (1/d) / 1 = d**(-1)
        # Therefore
        # tau_min = smax - 1 ** (alpha-1)
        # tau_max = smax - (1/d) ** (alpha-1)  
        # Same as entmax!!
        tau_lo = max_val - cls._gp(1, alpha) # 1
        tau_hi = max_val - cls._gp(1 / d, alpha) # (1/d)**(alpha-1)

        f_lo = (cls._p(X - tau_lo, alpha) ** alpha).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):

            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls._p(X - tau_m, alpha) # [X - tau]_+ ** (1/(alpha-1))
            f_m = (p_m ** alpha).sum(dim) - 1

            mask = (f_m >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        #p_m = p_m + 1e-12
        p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)
        ctx.save_for_backward(p_m)

        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        Y, = ctx.saved_tensors

        a = torch.where(Y > 0, Y, Y.new_zeros(1))
        b = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))
        
        dX = dY * b
        q = dX.sum(ctx.dim).unsqueeze(ctx.dim)
        dX -= q * a
        q = (dY * a).sum(ctx.dim).unsqueeze(ctx.dim)
        dX -= q * (b - b.sum(ctx.dim).unsqueeze(ctx.dim) * a)
        dX *= ((a ** ctx.alpha).sum(ctx.dim).unsqueeze(ctx.dim) ** ((ctx.alpha - 1) / ctx.alpha))
        dX /= (ctx.alpha - 1)

        return dX, None, None, None, None

def normmax_bisect(X, alpha=2, dim=-1):
    """alpha-normmax: normalizing sparse transform (a la softmax and entmax).

    Solves the optimization problem:

        max_p <x, p> - ||p||_alpha    s.t.    p >= 0, sum(p) == 1.

    where ||.||_alpha is the alpha-norm, with custom alpha >= 1,
    using a bisection (root finding, binary search) algorithm.

    This function is differentiable with respect to X.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    alpha : float or torch.Tensor
        Tensor of alpha parameters (> 1) to use. If scalar
        or python float, the same value is used for all rows, otherwise,
        it must have shape (or be expandable to)
        alpha.shape[j] == (X.shape[j] if j != dim else 1).

    dim : int
        The dimension along which to apply alpha-normmax.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return normmax_bisect(X, alpha=alpha, dim=dim)


def tsallis_unif(n, alpha, dtype=torch.double):
    if alpha == 1:
        return math.log(n)
    return (1 - n**(1-alpha)) / (alpha * (alpha-1))


def tsallis(p, alpha, dim=-1):
    if alpha == 1:
        return torch.special.entr(p).sum(dim)
    else:
        return ((p - p ** alpha) / (alpha * (alpha-1))).sum(dim=dim)

def normmaxentropy(p, alpha, dim=-1):

    return 1 - torch.sqrt((p**2).sum(dim=-1))

def normmax_unif(n, alpha, dim=-1):
    return 1 - math.sqrt(n)/n

def energy(Q, X, alpha=1.0, beta=1.0, normmax = False):
    # Q: tensor dim (m, d)
    # X: tensor dim (n, d)
    n = X.shape[0]

    # m is batch dim, so bXQ[i] = bXq_i
    bXQ = beta * Q @ X.T
    if normmax:
        phat = normmax_bisect(bXQ, alpha, -1)
        fy_term_gold = -normmax_unif(n, alpha) - bXQ.mean(dim=-1)
        fy_term_pred = -normmaxentropy(phat, alpha) - (bXQ * phat).sum(dim=-1)
    else:
        phat = entmax(bXQ, alpha, dim = -1)
        fy_term_gold = -tsallis_unif(n, alpha) - bXQ.mean(dim=-1)
        fy_term_pred = -tsallis(phat, alpha) - (bXQ * phat).sum(dim=-1)
    
    fy = fy_term_gold - fy_term_pred
    mx = X.mean(dim=0)
    q_nrmsq = (Q**2).sum(dim=-1)
    Msq = (X ** 2).sum(dim=-1).max()

    return -fy/beta - Q @ mx + q_nrmsq/2 + Msq/2

class HopfieldNet(nn.Module):

    def __init__(self, 
                 in_features,
                 alpha=1.0,
                 beta=1.0,
                 max_iter=128,
                 sparsemap = False,
                 normmax = False,
                 factor = "budget",
                 k = 2,
                 return_p = False):
        
        super(HopfieldNet, self).__init__()
        self.sparsemap = sparsemap
        self.normmax = normmax
        self.X = in_features
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.return_p = return_p
        self.factor = factor

    def _energy(self, Q):

        return energy(Q, self.X, self.alpha, self.beta)
        
    def _run(self, Q, eps=1e-6):
        """Synchronousl update

        Args:
            x (torch.Tensor): inputs
            eps (float): Defaults to 1e-6.

        """

        for _ in range(self.max_iter):
            if self.sparsemap:
                if self.factor == "budget":
                    p = SparseMAP_exactly_k(self.beta * self.X.mm(Q), self.k)
                else:
                    p = SparseMAP_sequence_exactly_k(self.beta * self.X.mm(Q), 100, self.k)
            elif self.normmax:
                p = normmax_bisect(self.beta * self.X.mm(Q), alpha=self.alpha, dim=0)
                
            else:
                p = entmax(self.beta * self.X.mm(Q), self.alpha, dim=0)
            Q = self.X.T @ p
        
        if self.return_p:
            return Q, p
        else:
            return Q

    def forward(self, Q):
        
        return self._run(Q)