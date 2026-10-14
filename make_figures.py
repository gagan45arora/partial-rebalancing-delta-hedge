"""
Reproduces every figure and every number quoted in main.tex.

    python make_figures.py

Writes figures/*.pdf and results.tex (LaTeX macros read by main.tex).
The plotting functions are also used by partial_rebalancing.ipynb.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import partial_rebalancing as pr

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#d9d9d9"
STYLE = {
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 2, "legend.frameon": False,
}
B_EXAMPLES = [0.0005, 0.00105, 0.00112, 0.0013]
LAM_MULTS = [0.0, 0.5, 0.9, 1.1, 2.0]
FULL_DATES = [5, 10, 20, 50, 100]


def fmt(v, p=0):
    s = f"{abs(v):,.{p}f}".replace(",", "{,}")
    return f"$-${s}" if v < 0 and s.strip("0.{,}") else s


# -----------------------------------------------------------------------------
# Calculations
# -----------------------------------------------------------------------------
def one_period():
    """Section 7: coefficients, thresholds and the four examples."""
    p = pr.ONE_PERIOD
    u = p["gap"] * p["n"]
    k1, k2 = pr.coeffs(p["S"], p["sigma"], p["h"], p["lam"], p["gamma"])
    bd = pr.boundaries(u, p["S"], k1, k2)
    examples = []
    for b in B_EXAMPLES:
        x = pr.decide(u, p["S"], b, p["a"], k1, k2)
        gain = float(pr.net(x, u, p["S"], b, p["a"], k1, k2)) if x > 0 else 0.0
        examples.append((b, x / p["n"], pr.REGION_NAMES[pr.region(x, u)], gain))
    return dict(p=p, u=u, k1=k1, k2=k2, b0=bd["b0"], b_hi=(k1 + k2 * u) / p["S"],
                a_c=bd["a_c"], bd=bd, examples=examples)


def check_closed_form(op, n_grid=400_001):
    """Grid maximisation of N reproduces Theorem 1 over a range of b."""
    p, u, k1, k2 = op["p"], op["u"], op["k1"], op["k2"]
    xg = np.linspace(0, u, n_grid)
    worst = 0.0
    for b in np.linspace(0.0002, 0.0016, 29):
        x_cf = pr.decide(u, p["S"], b, p["a"], k1, k2)
        N = pr.net(xg, u, p["S"], b, p["a"], k1, k2)
        x_bf = xg[N.argmax()] if N.max() > 0 else 0.0
        ok = abs(x_cf - x_bf) < 1.0 or abs(N.max() - pr.net(x_cf, u, p["S"], b, p["a"], k1, k2)) < 1e-6
        assert ok, (b, x_cf, x_bf)
        worst = max(worst, abs(x_cf - x_bf))
    return worst


def simulations():
    """Section 8: zero-cost check, full rebalancing on fixed schedules, and the rule."""
    zero = [(m, pr.simulate("full", n_steps=m, a=0.0, b=0.0)) for m in [5, 20, 80]]
    lam_star = pr.SIM["b"] / (pr.HULL["sigma"] * np.sqrt(pr.SIM["h"]))   # b = b0
    full = [(m, pr.simulate("full", n_steps=m)) for m in FULL_DATES]
    rule = [(k, pr.simulate("rule", lam=k * lam_star)) for k in LAM_MULTS]
    return dict(zero=zero, lam_star=lam_star, full=full, rule=rule)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def plot_first_guess():
    b_lo, b_hi = pr.exp_interval()
    b_axis = np.linspace(0.0, 0.30, 1501)
    f_opt = np.array([pr.best_f_exp(b) for b in b_axis])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(100 * b_axis, f_opt, color=C1)
    ax.axvspan(100 * b_lo, 100 * b_hi, color=C2, alpha=0.25, lw=0)
    ax.set_xlabel(r"Proportional cost $b$ (%)")
    ax.set_ylabel(r"Optimal $f^*$ (shares per option)")
    ax.annotate("interval of Lemma 2", (100 * b_hi, 0.3), xytext=(12, 0), textcoords="offset points")
    fig.tight_layout()
    return fig


CASE_B = [("Case 1", 0.0005), ("Case 2", 0.00105), ("Case 3", 0.0013)]


def plot_cases(op):
    """Lemma 2: N(x) in the three cases, annotated with the initial and final slopes."""
    p, u, k1, k2 = op["p"], op["u"], op["k1"], op["k2"]
    x = np.linspace(0, u, 2001)[1:]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    for ax, (name, b) in zip(axes, CASE_B):
        beta = b * p["S"]
        n0, nu = (k1 + k2 * u) - beta, k1 - beta
        ax.plot(x / p["n"], pr.net(x, u, p["S"], b, p["a"], k1, k2), color=C1)
        ax.plot([0], [-p["a"]], "o", color="black", ms=4)
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_title(rf"{name}: $b={100 * b:.3f}\%$" "\n"
                     rf"$\mathcal{{N}}'(0^+)={n0:+.4f}$, $\mathcal{{N}}'(u)={nu:+.4f}$", fontsize=9)
        ax.set_xlabel(r"$x/n$ (shares per option)")
    axes[0].set_ylabel(r"$\mathcal{N}(x)$ (\$)")
    fig.tight_layout()
    return fig


def plot_net(op):
    p, u, k1, k2, n = op["p"], op["u"], op["k1"], op["k2"], op["p"]["n"]
    f = np.linspace(0, u, 1001) / n
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for (b, fs, name, gain), col in zip(op["examples"], [C1, C2, C3, C4]):
        ax.plot(f, pr.net(f * n, u, p["S"], b, p["a"], k1, k2), color=col,
                label=rf"$b={100 * b:.3f}\%$: {name}")
        if fs > 0:
            ax.plot(fs, gain, "o", color=col, ms=6)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"Correction $f = x/n$ (shares per option)")
    ax.set_ylabel(r"Net benefit $\mathcal{N}$ (\$)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_fstar(op):
    p, u, n = op["p"], op["u"], op["p"]["n"]
    b_axis = np.linspace(0, 0.002, 2001)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for mult, col in zip([1, 5, 25], [C1, C2, C3]):
        k1m, k2m = pr.coeffs(p["S"], p["sigma"], p["h"], p["lam"], p["gamma"] * mult)
        ax.plot(100 * b_axis, pr.decide_v(u, p["S"], b_axis, p["a"], k1m, k2m) / n, color=col,
                label=rf"$\gamma = {mult}\,\gamma_0$")
    ax.axvline(100 * op["b0"], color="gray", lw=1, ls="--")
    ax.set_xlabel(r"Proportional cost $b$ (%)")
    ax.set_ylabel(r"Optimal $f^*$ (shares per option)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_regions(op):
    p, u, k1, k2, bd = op["p"], op["u"], op["k1"], op["k2"], op["bd"]
    b_map = np.linspace(0.0, 0.0016, 321)
    a_map = np.linspace(0.0, 400.0, 321)
    Z = np.array([[pr.region(pr.decide(u, p["S"], bb, aa, k1, k2), u) for bb in b_map] for aa in a_map])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.pcolormesh(100 * b_map, a_map, Z, cmap=ListedColormap([GREY, C2, C1]), vmin=-0.5, vmax=2.5,
                  shading="auto", rasterized=True)
    a_lo = np.linspace(0, bd["a_c"], 200)
    a_hi = np.linspace(bd["a_c"], a_map[-1], 200)
    ax.plot(100 * bd["full_partial"](a_lo), a_lo, color="black", lw=1)
    ax.plot(100 * bd["partial_none"](a_lo), a_lo, color="black", lw=1)
    ax.plot(100 * bd["full_none"](a_hi), a_hi, color="black", lw=1)
    ax.plot(100 * bd["b0"], bd["a_c"], "o", color="black", ms=5)
    ax.annotate("triple point", (100 * bd["b0"], bd["a_c"]), xytext=(8, 8), textcoords="offset points")
    ax.text(0.03, 150, "FULL", color="white", fontsize=12, weight="bold")
    ax.text(0.13, 300, "NONE", color="black", fontsize=12, weight="bold")
    ax.annotate("PARTIAL", (0.105, 30), xytext=(0.125, 120), arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel(r"Proportional cost $b$ (%)")
    ax.set_ylabel(r"Fixed fee $a$ (\$)")
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_dynamic(sims):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    fx = [r["cost"] / 1e3 for _, r in sims["full"]]
    fy = [r["std"] / 1e3 for _, r in sims["full"]]
    ax.plot(fx, fy, "o-", color=C2, ms=6, label="Full rebalancing, fixed schedule")
    for (m, _), x_, y_ in zip(sims["full"], fx, fy):
        ax.annotate(f"{m}", (x_, y_), xytext=(4, 4), textcoords="offset points", fontsize=8)
    offsets = {0.0: (-6, 8), 0.5: (-6, 8), 0.9: (-14, 8), 1.1: (-14, -14), 2.0: (2, 8)}
    for mult, r in sims["rule"]:
        ax.plot(r["cost"] / 1e3, r["std"] / 1e3, "s", color=C3 if mult < 1 else C1, ms=7)
        ax.annotate(rf"${mult:g}\lambda_*$", (r["cost"] / 1e3, r["std"] / 1e3),
                    xytext=offsets.get(mult, (4, 4)), textcoords="offset points", fontsize=8)
    ax.plot([], [], "s", color=C3, label=r"Rule, $\lambda<\lambda_*$ (partial trades)")
    ax.plot([], [], "s", color=C1, label=r"Rule, $\lambda>\lambda_*$ (full trades)")
    ax.set_xlabel(r"Mean transaction cost (\$ thousand)")
    ax.set_ylabel(r"Std. dev. of final P\&L (\$ thousand)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# results.tex
# -----------------------------------------------------------------------------
def write_results(op, sims, path):
    p = op["p"]
    b_lo, b_hi = pr.exp_interval()
    full_d, rule_d = dict(sims["full"]), dict(sims["rule"])
    M = dict(
        ExpBlo=f"{100 * b_lo:.1f}", ExpBhi=f"{100 * b_hi:.1f}", ExpWidth=f"{100 * (b_hi - b_lo):.1f}",
        ParS=fmt(p["S"]), ParSigma=f"{p['sigma']:.2f}", ParN=r"10^{5}", ParGap=f"{p['gap']:.2f}",
        ParU=fmt(op["u"]), ParA=fmt(p["a"]), ParLam=f"{p['lam']:.3f}", ParGamma=r"1.33\times10^{-7}",
        Kone=f"{op['k1']:.4f}", Ktwo=r"%.2f\times10^{-7}" % (op["k2"] * 1e7),
        BloPct=f"{100 * op['b0']:.3f}", BhiPct=f"{100 * op['b_hi']:.3f}",
        RelWidth=f"{op['k2'] * op['u'] / op['k1']:.2f}", Amax=fmt(op["a_c"]),
        Premium=fmt(sims["zero"][0][1]["prem"]),
        LamStar=f"{sims['lam_star']:.3f}", GamSim=r"10^{-5}",
        FullFiftyCost=fmt(full_d[50]["cost"] / 1e3, 1), FullFiftyStd=fmt(full_d[50]["std"] / 1e3, 1),
        FullHundCost=fmt(full_d[100]["cost"] / 1e3, 1), FullHundStd=fmt(full_d[100]["std"] / 1e3, 1),
        RuleZeroCost=fmt(rule_d[0.0]["cost"] / 1e3, 1), RuleZeroStd=fmt(rule_d[0.0]["std"] / 1e3, 1),
        RuleHalfCost=fmt(rule_d[0.5]["cost"] / 1e3, 1), RuleHalfStd=fmt(rule_d[0.5]["std"] / 1e3, 1),
        RuleTwoCost=fmt(rule_d[2.0]["cost"] / 1e3, 1), RuleTwoStd=fmt(rule_d[2.0]["std"] / 1e3, 1),
        RuleHalfGain=f"{100 * (1 - rule_d[0.5]['std'] / full_d[50]['std']):.0f}",
    )
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    lines.append(r"\newcommand{\ExampleRows}{%")
    for b, fs, name, gain in op["examples"]:
        lines.append(f"{100 * b:.3f}\\% & {fs:.3f} & {name} & {fmt(gain, 0)} \\\\")
    lines.append("}")
    lines.append(r"\newcommand{\ZeroRows}{%")
    for m, r in sims["zero"]:
        lines.append(f"{m} & {fmt(r['mean'])} & {r['std'] / r['prem']:.3f} \\\\")
    lines.append("}")
    lines.append(r"\newcommand{\DynRows}{%")
    for m, r in sims["full"]:
        lines.append(f"Full, {m} dates & -- & {r['trades']:.1f} & {fmt(r['cost'])} & {fmt(r['std'])} \\\\")
    lines.append(r"\midrule")
    for mult, r in sims["rule"]:
        lines.append(f"Rule, {pr.SIM['n_steps']} dates & ${mult:g}\\,\\lambda_*$ & {r['trades']:.1f} & "
                     f"{fmt(r['cost'])} & {fmt(r['std'])} \\\\")
    lines.append("}")
    with open(path, "w") as fh:
        fh.write("% Generated by make_figures.py -- do not edit by hand\n" + "\n".join(lines) + "\n")


def main():
    import matplotlib
    matplotlib.use("Agg")
    plt.rcParams.update(STYLE)
    os.makedirs(FIG, exist_ok=True)

    op = one_period()
    check_closed_form(op)
    sims = simulations()

    for name, fig in [("fig_cases", plot_cases(op)),
                      ("fig_first_guess", plot_first_guess()), ("fig_hills", plot_net(op)),
                      ("fig_fstar", plot_fstar(op)), ("fig_phase", plot_regions(op)),
                      ("fig_dynamic", plot_dynamic(sims))]:
        fig.savefig(os.path.join(FIG, name + ".pdf"))
        plt.close(fig)
    write_results(op, sims, os.path.join(HERE, "results.tex"))
    print("done: figures/*.pdf and results.tex written")


if __name__ == "__main__":
    main()
