"""
Core calculations for the note "Partial rebalancing of a delta hedge under fixed and
proportional transaction costs: a three-phase solution".

Used by make_figures.py (figures and numbers in main.tex) and partial_rebalancing.ipynb.

Notation (as in the paper)
    g, u   hedge error in shares (u = |g| > 0 is the current error)
    x      shares traded, 0 <= x <= u
    a, b   fixed cost per trade ($) and proportional cost (fraction of traded value)
    h      horizon over which the error is held (years)
    lam    Lagrange multiplier of the standard-deviation constraint (dimensionless)
    gamma  variance-penalty coefficient (1/$)
    rho(g) = k1 |g| + k2 g^2 / 2,  k1 = lam S sigma sqrt(h),  k2 = gamma S^2 sigma^2 h
"""
import numpy as np
from scipy.stats import norm

# -----------------------------------------------------------------------------
# Parameters used in the paper
# -----------------------------------------------------------------------------
EXP_PARAMS = dict(S=49.0, u=0.40, mu=0.05, tau=1.0, sigma=0.20, r=0.05, a=0.02)   # Section 4
ONE_PERIOD = dict(S=49.0, sigma=0.20, h=1 / 52, n=100_000, gap=0.40, a=20.0,
                  lam=0.036, gamma=1.33e-7)                                          # Section 7
HULL = dict(S0=49.0, K=50.0, r=0.05, mu=0.13, sigma=0.20, T=20 / 52, n=100_000)     # Section 8
SIM = dict(n_paths=10_000, n_steps=100, a=20.0, b=0.001, h=1 / 52, gamma=1e-5, seed=7)


# -----------------------------------------------------------------------------
# Section 4: exponential risk functional
# -----------------------------------------------------------------------------
def rho_exp(g, S, mu, tau, sigma, **_):
    return S * np.exp(g * mu * tau - 0.5 * g**2 * sigma**2 * tau + g * sigma * np.sqrt(tau))


def drho_exp(g, S, mu, tau, sigma, **_):
    return rho_exp(g, S, mu, tau, sigma) * (mu * tau - g * sigma**2 * tau + sigma * np.sqrt(tau))


def exp_interval(p=EXP_PARAMS):
    """Lemma 2 interval for b, with costs compounded to expiry by exp(r tau)."""
    scale = p["S"] * np.exp(p["r"] * p["tau"])
    return drho_exp(0.0, **p) / scale, drho_exp(p["u"], **p) / scale


def best_f_exp(b, p=EXP_PARAMS, n_grid=40_001):
    """Grid maximisation of the net benefit under the exponential functional."""
    f = np.linspace(0, p["u"], n_grid)
    N = (rho_exp(p["u"], **p) - rho_exp(p["u"] - f, **p)
         - np.where(f > 0, p["a"] + p["S"] * b * f, 0.0) * np.exp(p["r"] * p["tau"]))
    i = N.argmax()
    return f[i] if N[i] > 0 else 0.0


# -----------------------------------------------------------------------------
# Sections 5-6: combined functional and closed-form solution
# -----------------------------------------------------------------------------
def coeffs(S, sigma, h, lam, gamma):
    """k1 ($/share) and k2 ($/share^2) of rho(g) = k1|g| + k2 g^2 / 2."""
    return lam * S * sigma * np.sqrt(h), gamma * S**2 * sigma**2 * h


def net(x, u, S, b, a, k1, k2):
    """Net benefit N(x) = rho(u) - rho(u - x) - c(x), with N(0) = 0."""
    return (k1 + k2 * u) * x - 0.5 * k2 * x**2 - np.where(x > 0, a + b * S * x, 0.0)


def decide(u, S, b, a, k1, k2):
    """Optimal number of shares to trade for a hedge error u >= 0 (Theorem 1)."""
    c = b * S                              # marginal cost per share
    if c <= k1:                            # marginal cost <= rho'(0+): full candidate
        x = u
    elif k2 > 0 and c < k1 + k2 * u:       # rho'(0+) < c < rho'(u): interior candidate
        x = u - (c - k1) / k2
    else:                                  # c >= rho'(u): no trade
        return 0.0
    return x if net(x, u, S, b, a, k1, k2) > 0 else 0.0


decide_v = np.vectorize(decide)


def region(x, u):
    """0 = none, 1 = partial, 2 = full."""
    return 2 if np.isclose(x, u) else (0 if x == 0 else 1)


REGION_NAMES = ["none", "partial", "full"]


def boundaries(u, S, k1, k2):
    """Proposition 1: region boundaries in the (b, a) plane and the triple point."""
    b0 = k1 / S
    a_c = 0.5 * k2 * u**2
    full_partial = lambda a: np.full_like(np.asarray(a, float), b0)
    partial_none = lambda a: (k1 + k2 * u - np.sqrt(2 * k2 * np.asarray(a, float))) / S
    full_none = lambda a: (k1 + 0.5 * k2 * u - np.asarray(a, float) / u) / S
    return dict(b0=b0, a_c=a_c, full_partial=full_partial, partial_none=partial_none,
                full_none=full_none)


# -----------------------------------------------------------------------------
# Section 8: simulation of the complete hedge
# -----------------------------------------------------------------------------
def bs_price_delta(S, K, r, sigma, tau):
    """Black-Scholes call price and delta."""
    tau = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2), norm.cdf(d1)


def decide_paths(u, S, b, a, sigma, h, lam, gamma):
    """Vectorised decide() over paths (u and S are arrays)."""
    k1, k2 = coeffs(S, sigma, h, lam, gamma)
    c = b * S
    full = c <= k1
    w = np.where(k2 > 0, (c - k1) / np.where(k2 > 0, k2, 1.0), np.inf)
    x = np.where(full, u, np.clip(u - w, 0.0, u))
    gain = (k1 + k2 * u) * x - 0.5 * k2 * x**2 - (a + b * S * x)
    return np.where((x > 0) & (gain > 0), x, 0.0)


def simulate(rule, n_steps=SIM["n_steps"], lam=None, gamma=SIM["gamma"], a=SIM["a"],
             b=SIM["b"], h=SIM["h"], n_paths=SIM["n_paths"], seed=SIM["seed"], hp=HULL):
    """
    Short hp['n'] calls, hedged with shares, cash accruing at r.
    rule = 'full': trade to delta at each of n_steps dates.
    rule = 'rule': apply Theorem 1 at each of n_steps dates with (lam, gamma, h).
    Returns mean trades per path, mean PV of transaction costs, std and mean of PV of P&L.
    """
    rng = np.random.default_rng(seed)
    S0, K, r, mu, sg, T, n = (hp[k] for k in ["S0", "K", "r", "mu", "sigma", "T", "n"])
    dt = T / n_steps
    S = np.full(n_paths, S0)
    prem = bs_price_delta(S0, K, r, sg, T)[0]
    cash = np.full(n_paths, prem * n)          # premium received
    shares = np.zeros(n_paths)
    trades = np.zeros(n_paths)
    tc = np.zeros(n_paths)
    for i in range(n_steps):
        _, d = bs_price_delta(S, K, r, sg, T - i * dt)
        gap = d * n - shares
        x = np.abs(gap) if rule == "full" else decide_paths(np.abs(gap), S, b, a, sg, h, lam, gamma)
        trade = np.sign(gap) * x
        c = np.where(x > 0, a + b * S * x, 0.0)
        cash -= trade * S + c
        shares += trade
        trades += x > 0
        tc += c
        cash *= np.exp(r * dt)
        S = S * np.exp((mu - 0.5 * sg**2) * dt + sg * np.sqrt(dt) * rng.standard_normal(n_paths))
    disc = np.exp(-r * T)
    pnl = (cash + shares * S - n * np.maximum(S - K, 0)) * disc
    return dict(trades=trades.mean(), cost=(tc * disc).mean(), std=pnl.std(), mean=pnl.mean(),
                prem=prem * n)
