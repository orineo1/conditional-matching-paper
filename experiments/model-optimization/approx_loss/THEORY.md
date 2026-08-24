# Approximate / linear-time distributional objectives for CDM guidance — theory note (Agent 3)

## 0. What is being approximated, and what the honest baseline costs

Per diffusion step the guidance loss is the repository `MMDLoss` (`simulations/src/LossFunctions.py`)

    K(u,v)   = sum_{k=-2..2} exp( -||u-v||^2 / (bw 2^k) )          (5 bandwidths, sigma_k^2 = bw 2^k / 2)
    MMD^2_V  = mean_ij K(x_i,x_j) - 2 mean_ij K(x_i,y_j) + mean_ij K(y_i,y_j)

with `X` = n conditional-model draws (n in 1..32, requires grad), `Y` = S_G fixed (m = 250 synthetic, ~100-120
CLIP-768 for SD), bandwidth `bw` frozen by `_common.fixed_bandwidth` (mean off-diagonal squared distance of S_G).
SD (`SD_cond_SD_controlnet/src/metrics.py::compute_mmd`) uses the single-bandwidth generalised kernel
`exp(-(d^2/2bw^2)^alpha)`, unbiased U-statistic, `sqrt(|MMD^2|+1e-8)`, median-heuristic `bw`, and the adaptive
step `zeta = base_zeta / MMD` (`generation.py`). MNIST (`MNIST/run_mlgdf.py`) guides with a sliced
Wasserstein distance (50 *freshly drawn* projections and a *freshly drawn* target sample each step).

Two facts fix the cost accounting:

1. `Y` never changes along a trajectory, so the YY block is a constant: it contributes nothing to the gradient and
   can be computed once (Agent 2's exact simplification, `ReferenceCachedYY` here). The repository's stacked
   `(X;Y)` matrix recomputes it every call: `(n+m)^2 (d+5)` vs `n(n+m)(d+5)`; measured 9-11x wall at n=8, m=250 (d=1..16) and 2.6x at d=768, m=120.
   **Every approximation must be compared with the cached-YY exact loss, not with the repository call.**
2. The gradient-relevant work of the exact loss is `n^2 + n m` kernel evaluations, each `O(d)` plus 5 `exp`.
   At m=250, d=1 that is ~2000 kernel evaluations for n=8 — microseconds. The conditional model dominates the
   step (Agent 1's profile), so the *achievable* speed-up from any loss approximation in the synthetic setting is
   bounded by the loss's share of the step, which is small once YY is cached.

Hardware-independent cost model used in `diagnostics.csv` (`cost_ratio_vs_cachedYY`): flops-ish count of the
per-call forward, the backward being a constant multiple.

## 1. "Fixed features across a trajectory" (common random numbers)

All randomised candidates (RFF/ORF frequencies, Nystrom landmark subset, target subsample, sliced projections)
have an extra source of randomness besides the n conditional draws. If the features are redrawn at every diffusion
step, the guidance sees a *different* objective each step: the error `L_D(X) - L(X)` is then independent noise
across steps, its gradient noise adds to the Monte-Carlo noise of the n draws, and the trajectory performs a
random walk around the exact one. If instead the features `omega` are drawn ONCE (seeded) and frozen,
`L_D(.; omega)` is a *single, smooth, deterministic* surrogate objective for the whole trajectory: its error is a
fixed bias (of the same order) rather than fresh variance, gradients at consecutive steps are correlated the way
they are for the exact loss, and Adam's moment estimates (the `temporal=adam` rule) see a consistent landscape.
This is the common-random-numbers argument; it is also what makes runs reproducible and what lets a Stage-1
screen attribute differences to the candidate rather than to feature noise. All classes in `approx_mmd.py` draw
from a `torch.Generator(seed)` at construction and expose `resample(seed)` only as an explicit opt-in; the
target-side features (`mean phi(Y)`, sorted projections, landmark Gram inverse, `E k(y,y')`) are precomputed once.

## 2. Candidates

### (a) Random Fourier features, multi-bandwidth, D features

Bochner: `exp(-||u-v||^2/(2 sigma^2)) = E_{w~N(0,I/sigma^2)} cos(w.(u-v))`. With
`phi_k(x) = sqrt(2/D)[cos(W x/sigma_k), sin(W x/sigma_k)]`, `W in R^{D/2 x d}` Gaussian, shared across the 5
bandwidths (CRN across bandwidths too), `phi_k(x).phi_k(y)` is unbiased for the k-th kernel with variance `O(1/D)`;
`MMD^2 ~ sum_k ||mean phi_k(X) - mean phi_k(Y)||^2`. Because MMD^2 is quadratic in the features, the estimator
is biased upward by `E||noise||^2 = O(1/D)`; the bias of the *gradient* is zero at first order but its variance
scales like `d/(D)` relative to the exact gradient: d/dx of `cos(w.x)` carries a factor `w`, `E||w||^2 = d/sigma^2`.
Hence RFF needs `D >> d` for a useful gradient — measured: d=1, D=256: cos 0.96-0.999; d=8-16, D=256: cos
0.82-0.90; d=768, D=256: cos 0.23, D=16384: 0.88, D=65536: 0.97 (`diagnostics.log`, section 4 below).
Cost per call: `n D (d+2) 5/2 + 5D` vs exact `n(n+m)(d+5)`. Ratio at m=250: `D=256 -> 1.2-2.2x` (NOT cheaper),
`D=64 -> 0.3-0.5x` with cos <= 0.98 (d=1) / 0.63-0.75 (d=8,16). At m=120, d=768 the break-even D is ~70, where
the gradient is noise. Target mean feature: precomputed once (`m D d`).

### (b) Orthogonal random features (ORF)

Same estimator, `W = S Q` with `Q` orthogonal (QR of a Gaussian block) and `S` chi(d) row norms; variance
reduction of the kernel estimate by a factor `~(1 - 1/d)`-ish relative terms that only matter when `D <~ d`
(Yu et al. 2016). For d=1 a "block" has one row, ORF == RFF (so it is only implemented for d>1 and only exercised
at d=768). Measured in CLIP-like d=768: ORF has a lower loss error and lower gradient-norm error than RFF at every D
(e.g. D=1024: loss err 2.5% vs 3.2-3.9%, grad-norm err 0.93-0.96 vs 1.2-1.35) but NOT a better gradient cosine
(0.35-0.39 vs 0.43-0.46, within seed noise); in a separate sweep to D=65536 ORF reaches cos 0.977 vs RFF 0.966.
The improvement is the expected `O(1/d)`-relative one — never enough to make D<=1024 usable. Rejected with RFF.

### (c) Nystrom with target landmarks

Landmarks `Z` = L points of S_G; `phi(x) = K_ZZ^{-1/2} k(Z,x)` (eigen-floored pseudo-inverse), per bandwidth;
`MMD^2 ~ sum_k ||mean phi_k(X) - mean phi_k(Y)||^2` = exact MMD of the kernel projected on `span{k(z,.)}`.
Deterministic given Z, downward biased, error = RKHS norm of the witness function outside the span.
**Structural failure mode for guidance:** early in a trajectory X sits far from S_G (that is when guidance is
needed); `k(x,.)` for such x is nearly orthogonal to the landmark span, so `P phi(x) ~ 0`, the XX term and the
gradient collapse. Measured: d=1, L=16: cos 0.99996 (the 5-bandwidth family with sigma_max ~ 10 covers the line,
so 16 landmarks span everything: L=16 is 0.8-0.9x the exact cost, L=64 is 12x because of the `n L^2` projection);
d=8, L=64: cos 0.94; d=16, L=64: 0.93; d=768, L=64 of 120: cos 0.25 (X cluster outside the target span).
Including X in the landmark set fixes the bias but requires re-factorising `K_ZZ` (`O(L^3)`) every step. Cost per
call: `n L (d + 5(1+L))` — the `L^2` term makes it more expensive than exact for `L >~ sqrt(m)`. Rejected except
as a curiosity in d=1 where it is exact-ish but no cheaper.

### (d) Linear-time / block estimators

Gretton et al.'s linear-time MMD pairs `(x_{2i}, x_{2i+1}, y_{2i}, y_{2i+1})`; B-test uses blocks. For us n<=32
so the XX term costs `n^2 <= 1024` kernel evaluations — nothing to save; YY is constant; the only term that is
`O(m)` is the cross term `mean_i mu_S(x_i)`, `mu_S(x) = mean_j K(x,y_j)`. "Linear time in m" therefore means
subsampling the target: `B` of `m` targets (`SubsampledTargetMMD`). With a FIXED subset (CRN) it is exactly the
repository loss against a smaller target set: unbiased for `E_{y~G}` but its bias w.r.t. S_G is the `O(1/sqrt(B))`
sampling error of B points; gradient-norm error 4-13% at B=64, cos 0.96-0.98 in d=1..16 (0.74 at d=768 where
m=120 is already small). With a FRESH subset per step (B-test style) the estimator is unbiased for S_G at every
step but the error becomes variance added to the trajectory (section 1). Cost `n(n+B)(d+5)`: 0.28-0.34x at B=64.
Honest reading: this is "use fewer targets", and it quantifies how much of m=250 the gradient direction actually
uses. Viable only as a cost-accuracy knob; see REPORT.

### (e) Sliced / projected distances with fixed projections

`SW_2^2(X, S_G) = mean_p W_2^2(<X,theta_p>, <S_G,theta_p>)` with P FIXED unit projections, target sorted once,
quantile matching for n != m; cost `n P (d + log n)`. In d=1 every projection is +-1 so it is exactly the 1-D
`W_2^2` (P irrelevant). It is a DIFFERENT objective, not an MMD approximation: loss values are not comparable
(rel. "error" ~ 10-100 in the tables), gradient cosine to the MMD gradient 0.34-0.89 (d=1), 0.5-0.7 (d=8,16),
~0.1 (d=768 with P=32: sliced distances in 768-D need P ~ d projections to see a shift, the known curse of slicing).
Relation to MNIST: `run_mlgdf.py` guides with SWD (fresh projections and fresh target draws every step) and ALSO
evaluates with SWD (`final_loss`). If SW were adopted as the guidance surrogate in the synthetic/SD tasks, the
paper's final metric must stay what it is (`|x - x*|` / L2 to the target in the synthetic, held-out CLIP MMD /
gender-classifier in SD) and must not be the same sliced statistic, and MNIST's evaluation SWD should at least use
projections independent of the guidance ones. Not recommended as an MMD replacement; in d=1 the exact W_2 is a
legitimate *alternative* objective (Agent 4 / estimator domain) rather than an approximation.

### (f) Target-specific exact simplification: population GMM target

When the target is a known GMM `G = sum_c w_c N(mu_c, S_c)` (all synthetic settings: K=2 in the 2D/5D/10D params,
<=11 in `dimy_benchmark`), the cross term has a closed form (`tfg.gmm_mmd.kernel_mean_embedding_multibandwidth`):

    mu_G(x) = E_{y~G} K(x,y) = sum_k (2 pi sigma_k^2)^{d/2} sum_c w_c N(x; mu_c, S_c + sigma_k^2 I)

and `E_{y,y'~G} K(y,y')` is a constant (`gmm_mmd._inner`). `MMD^2(P_X, G) = mean_ij K(x_i,x_j) - 2 mean_i mu_G(x_i)
+ const` is the exact "semi-population" MMD^2 between the empirical X and the population target. Cost per call
`n^2 (d+5) + 5 n K d^2` (the `d^2` from the Cholesky of `S_c + sigma^2 I`; with diagonal covariances it is `n K d`):
at K=2, d=1 it is 0.02-0.12x the exact-empirical cost, and it removes the target-sampling noise entirely (the
gradient is the gradient of the objective whose minimiser is x*, not of a 250-sample proxy). This is NOT an
approximation of the repository objective but a change of target: `MMD^2(P_X, G) - MMD^2(P_X, S_G)` is the
`O(1/sqrt(m))` sampling error of S_G (measured rel. loss error 1-5%, gradient cosine 0.989-0.998 across d=1,8,16).
Tests: equals `population_mmd2_multibandwidth(X-as-zero-variance-mixture, G)` to 1e-10 and matches the empirical
loss against a 200k-sample target to <0.5%. It is exact, cheap, deterministic, differentiable; applicable only
where the target is given parametrically (synthetic; MNIST's angular GMM; NOT SD where targets are CLIP images).

### FFT: why it does not apply, and when it would

An FFT accelerates *convolution on a regular grid*. `mu_S(x) = mean_j K(x - y_j)` IS a convolution of the kernel
with the empirical measure `1/m sum_j delta_{y_j}`, but in CLIP-768 (and any d >~ 3) the points are not on a grid,
a grid with G points per axis has `G^d` cells, and the result is wanted at n arbitrary query points — there is
no grid structure to transform, and the non-uniform FFT alternatives (NUFFT / fast Gauss transform / FMM) are
`O((n+m) log) ` with constants that only beat `n m d` for `n m >~ 10^5-10^6`; here `n m <= 8000`. It also does
not exist for the `alpha=2` SD kernel `exp(-(r^2/2bw^2)^2) = exp(-r^4/..)`, which is not even positive definite
(Bochner fails for `exp(-|r|^beta)`, beta>2), so neither FFT nor RFF applies there (Nystrom would produce an
indefinite `K_ZZ`). In d_y = 1 the structure does exist: bin S_G into a histogram `h` on a G-point grid, then
`mu_S` on the grid is `h * K` = `ifft(fft(h) fft(K))` in `O(G log G)`, evaluated at x by interpolation in `O(n)`
(with O(h^2) value bias and a piecewise-constant/linear gradient). But the grid table can equally be filled by a
direct `G m` sum ONCE per run — `2048 x 250 = 5 x 10^5` kernel evaluations ~ 0.2 ms — because S_G is fixed; the FFT
would only save that one-off cost, which is already far below one call of the conditional model. The per-call
saving comes from the *table*, not the FFT: `TabulatedKME1D` (direct fill) achieves cos 0.999, rel. loss error
<2e-3 at 0.04-0.12x the exact cost. Decision: **no FFT candidate proceeds**; the d=1 tabulated KME is kept as
the concrete embodiment of the "grid structure" argument, and the population-GMM closed form dominates it
whenever the target is parametric (cheaper, no grid bias, exact).

## 3. Bias / variance summary

| candidate | randomness | bias vs S_G objective | variance (fixed features) | gradient quality driver |
|---|---|---|---|---|
| RFF/ORF | W (frozen) | `+O(1/D)` on MMD^2, 0 on E grad | across seeds `O(1/sqrt D)`; grad noise `~ d/D` | needs D >> d |
| Nystrom (target landmarks) | subset (frozen) | downward, = witness outside span; -> gradient collapse far from S_G | small | span coverage of X |
| subsample B (fixed) | subset (frozen) | = sampling error of B targets, `O(1/sqrt B)` | across subsets `O(1/sqrt B)` | B |
| subsample B (fresh) | per step | 0 per step | adds per-step noise | B |
| sliced W2 | projections (frozen) | different objective | `O(1/sqrt P)` for d>1 | P vs d |
| population GMM | none | = sampling error of S_G itself | 0 | exact |
| tabulated KME (d=1) | none | `O(h^2)` | 0 | grid |

## 4. Where the numbers come from

`diagnostics.py` (run from `simulations/`): `diagnostics.csv`, `DIAGNOSTICS.md`, `diagnostics_grad_cos_and_time.png`,
`diagnostics.log`. Settings: synthetic d=1 (paper 2D GMM target, m=250), d=8,16 (`tfg.dimy_benchmark`, m=250),
X = n conditional draws at a random x in [-7,7]; CLIP-like d=768, m=120 (two unit-sphere Gaussian clusters), X from
a shifted third cluster; n in {4,8,32} / {8,32}; 4 X draws x 6 feature seeds; wall = median of 7 fwd+bwd after
warm-up, float64 CPU.
