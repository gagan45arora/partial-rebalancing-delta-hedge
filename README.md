# Partial rebalancing of a delta hedge under fixed and proportional transaction costs

Paper (`main.pdf`), source and code for a one-period analysis of the optimal size of a
delta-hedge correction when each trade costs a fixed amount $a$ plus a proportional cost $b$
of the traded value.

## Summary

* To leading order in the horizon $h$, the P&L of a hedge error of $g$ shares is linear in $g$,
  with standard deviation $|g|\,S\sigma\sqrt h$. Any dependence of the value of a correction on the
  size of the remaining error therefore comes from the risk functional $\rho(g)$ used to evaluate it.
* For convex $\rho$, a partial correction can be optimal only if the marginal cost per share $bS$
  lies in $(\rho'(0^+),\ \rho'(u))$, where $u$ is the current error (Lemma 2).
* For $\rho(g) = k_1|g| + \tfrac12 k_2 g^2$, with $k_1 = \lambda S\sigma\sqrt h$ (Lagrange multiplier of a
  constraint on the P&L standard deviation) and $k_2 = \gamma S^2\sigma^2 h$ (variance penalty), the optimal
  correction is obtained in closed form. The $(b, a)$ plane divides into three regions (full, partial,
  none) with explicit boundaries meeting at the triple point $(k_1/S,\ \tfrac12 k_2 u^2)$. The relative
  width of the partial region is $\gamma u S\sigma\sqrt h/\lambda$.
* An exponential risk functional that produced partial optima only for $b \approx 24\%$ is shown not to
  represent the risk of the hedge error.
* A Monte Carlo simulation of the 20-week call example in Hull, *Options, Futures, and Other
  Derivatives*, compares the rule with full rebalancing on a fixed schedule.

## Contents

| File | Description |
|---|---|
| `main.pdf` | The paper |
| `main.tex`, `results.tex`, `figures/` | LaTeX source; `results.tex` holds every number quoted in the text |
| `partial_rebalancing.py` | Core calculations: risk functionals, closed-form solution, region boundaries, hedge simulation |
| `make_figures.py` | Regenerates `figures/*.pdf` and `results.tex` |
| `partial_rebalancing.ipynb` | Companion notebook following the paper section by section, with additional numerical checks |

## Reproducing the paper

```bash
pip install -r requirements.txt
python make_figures.py        # about 2 minutes; writes figures/*.pdf and results.tex
pdflatex main.tex
pdflatex main.tex             # second pass resolves cross-references
```

The notebook runs in about 3 minutes (`jupyter notebook partial_rebalancing.ipynb`).

Tested with Python 3.8.10, NumPy 1.24.4, SciPy 1.10.1, Matplotlib 3.7.5, pandas 2.0.3.

## Author

Gagandeep Singh — M.Sc. Physics, TIFR (2024). Independent work. gagan45arora@gmail.com
