"""
analysis.py
-----------
Experiments and plots comparing smoothing functions for the
projectile contact problem.

Experiments:
  1. Smoothing functions and their derivatives (visual comparison)
  2. Loss landscape L_kappa(theta) for each smoothing
  3. Gradient field: FD vs FoG
  4. Convergence of gradient descent
  5. Smoothing bias ||theta*_kappa - theta*|| vs kappa
  6. ZoG vs FoG comparison at a single theta
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

from smoothing import (get_smoothing, SMOOTHING_NAMES, SMOOTHING_LABELS,
                       SMOOTHING_COLORS, SMOOTHING_LS)
from simulator import (Params, simulate, loss, grad_fd, grad_fog,
                       grad_zog, analytical_optimum)

OUT = '/home/claude/figures'
os.makedirs(OUT, exist_ok=True)

PARAMS = Params()
KAPPA  = 300.0   # default stiffness (as in Schwarke et al.)

# Default smoothing names to compare (exclude hard for most plots)
COMPARE = ['sigmoid', 'erf', 'smoothstep', 'sigmoid_mass']


# ============================================================
# Utilities
# ============================================================

def get_fn(name):
    return get_smoothing(name, mass=PARAMS.m)


def theta_grid(n=60, lo=-3.5, hi=3.5):
    t1 = np.linspace(lo, hi, n)
    t2 = np.linspace(lo, hi, n)
    return np.meshgrid(t1, t2)


# ============================================================
# Experiment 1: Smoothing functions and their derivatives
# ============================================================

def plot_smoothing_functions(kappa=KAPPA):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    d_vals = np.linspace(-0.05, 0.05, 500)

    for name in COMPARE + ['hard']:
        sigma_fn, sigma_prime_fn = get_fn(name)
        lbl   = SMOOTHING_LABELS[name]
        col   = SMOOTHING_COLORS[name]
        ls    = SMOOTHING_LS[name]

        axes[0].plot(d_vals, sigma_fn(d_vals, kappa),
                     label=lbl, color=col, ls=ls, lw=2)
        axes[1].plot(d_vals, sigma_prime_fn(d_vals, kappa),
                     label=lbl, color=col, ls=ls, lw=2)

    for ax in axes:
        ax.axvline(0, color='k', lw=0.7, ls=':')
        ax.set_xlabel(r'Penetration depth $d = -q_z^M$ [m]')
    axes[0].set_ylabel(r'$\sigma(d, \kappa)$')
    axes[1].set_ylabel(r"$\sigma'(d, \kappa)$")
    axes[0].set_title(r'Smoothing functions ($\kappa = {}$)'.format(int(kappa)))
    axes[1].set_title(r'Derivatives (gradient signal)')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = f'{OUT}/smoothing_functions.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ============================================================
# Experiment 2: Loss landscape
# ============================================================

def compute_loss_landscape(sigma_fn, kappa, n=40):
    T1, T2 = theta_grid(n=n, lo=-3.0, hi=3.0)
    L = np.zeros_like(T1)
    for i in range(n):
        for j in range(n):
            theta = np.array([T1[i, j], T2[i, j]])
            L[i, j] = loss(theta, sigma_fn, kappa, PARAMS)
    return T1, T2, L


def plot_loss_landscapes(kappa=KAPPA, n=35):
    theta_star = analytical_optimum(PARAMS)
    fig, axes = plt.subplots(1, len(COMPARE), figsize=(14, 3.5),
                             sharey=True)

    for ax, name in zip(axes, COMPARE):
        sigma_fn, _ = get_fn(name)
        T1, T2, L = compute_loss_landscape(sigma_fn, kappa, n=n)
        vmax = np.percentile(L, 95)
        cf = ax.contourf(T1, T2, np.clip(L, 0, vmax),
                         levels=20, cmap='viridis')
        ax.contour(T1, T2, L, levels=10, colors='white',
                   linewidths=0.5, alpha=0.4)
        ax.scatter(*theta_star, c='red', s=80, zorder=5,
                   label=r'$\theta^*$ analytical')
        # Find numerical minimum
        idx = np.unravel_index(np.argmin(L), L.shape)
        ax.scatter(T1[idx], T2[idx], c='yellow', marker='*',
                   s=120, zorder=5, label=r'$\theta^*_\kappa$ numerical')
        ax.set_title(SMOOTHING_LABELS[name], fontsize=9)
        ax.set_xlabel(r'$\theta_1$ [m/s]')
        plt.colorbar(cf, ax=ax, fraction=0.046)

    axes[0].set_ylabel(r'$\theta_2$ [m/s]')
    axes[0].legend(fontsize=7)
    fig.suptitle(r'Loss landscape $L_\kappa(\theta)$, $\kappa={}$'.format(
        int(kappa)), fontsize=11)
    fig.tight_layout()
    path = f'{OUT}/loss_landscapes.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ============================================================
# Experiment 3: Gradient quality -- FD vs FoG
# ============================================================

def plot_gradient_comparison(kappa=KAPPA, n=15):
    """
    Compare FD and FoG gradient fields for each smoothing function.
    Also show the norm of (FoG - FD) as a measure of gradient error.
    """
    theta_star = analytical_optimum(PARAMS)

    fig, axes = plt.subplots(2, len(COMPARE),
                             figsize=(14, 7))

    T1, T2 = theta_grid(n=n, lo=-2.5, hi=2.5)

    for col_idx, name in enumerate(COMPARE):
        sigma_fn, sigma_prime_fn = get_fn(name)

        G_fd  = np.zeros((*T1.shape, 2))
        G_fog = np.zeros((*T1.shape, 2))
        err   = np.zeros(T1.shape)

        for i in range(n):
            for j in range(n):
                theta = np.array([T1[i, j], T2[i, j]])
                gfd  = grad_fd(theta, sigma_fn, kappa, PARAMS)
                gfog = grad_fog(theta, sigma_fn, sigma_prime_fn,
                                kappa, PARAMS)
                G_fd[i, j]  = gfd
                G_fog[i, j] = gfog
                err[i, j]   = np.linalg.norm(gfog - gfd)

        # Row 0: FD gradient field
        ax0 = axes[0, col_idx]
        ax0.quiver(T1[::2, ::2], T2[::2, ::2],
                   G_fd[::2, ::2, 0], G_fd[::2, ::2, 1],
                   color=SMOOTHING_COLORS[name], alpha=0.8,
                   scale=None, width=0.005)
        ax0.scatter(*theta_star, c='red', s=60, zorder=5)
        ax0.set_title(f'FD grad\n{SMOOTHING_LABELS[name]}', fontsize=8)
        ax0.set_aspect('equal')

        # Row 1: FoG gradient error ||FoG - FD||
        ax1 = axes[1, col_idx]
        cf = ax1.contourf(T1, T2, err, levels=15, cmap='Reds')
        ax1.scatter(*theta_star, c='blue', s=60, zorder=5)
        ax1.set_title(r'$\|\nabla^{[1]}L - \nabla^{FD}L\|$', fontsize=8)
        plt.colorbar(cf, ax=ax1, fraction=0.046)
        ax1.set_aspect('equal')

        for ax in [ax0, ax1]:
            ax.set_xlabel(r'$\theta_1$')
    axes[0, 0].set_ylabel(r'$\theta_2$')
    axes[1, 0].set_ylabel(r'$\theta_2$')

    fig.suptitle(r'Gradient comparison (FD vs FoG), $\kappa={}$'.format(
        int(kappa)), fontsize=11)
    fig.tight_layout()
    path = f'{OUT}/gradient_comparison.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ============================================================
# Experiment 4: Gradient descent convergence
# ============================================================

def gradient_descent(theta0, sigma_fn, sigma_prime_fn, kappa, params,
                     lr=0.05, n_iter=300, use_fog=True):
    """Run gradient descent and return trajectory of theta and loss."""
    theta = theta0.copy()
    hist_theta = [theta.copy()]
    hist_loss  = [loss(theta, sigma_fn, kappa, params)]

    for _ in range(n_iter):
        if use_fog:
            g = grad_fog(theta, sigma_fn, sigma_prime_fn, kappa, params)
        else:
            g = grad_fd(theta, sigma_fn, kappa, params)
        theta = theta - lr * g
        hist_theta.append(theta.copy())
        hist_loss.append(loss(theta, sigma_fn, kappa, params))

    return np.array(hist_theta), np.array(hist_loss)


def plot_convergence(kappa=KAPPA, n_iter=200, lr=0.03):
    theta_star = analytical_optimum(PARAMS)
    theta0_list = [
        np.array([0.5,  0.3]),
        np.array([-1.0, 1.5]),
        np.array([2.5, -0.5]),
    ]
    colors_init = ['#333', '#666', '#999']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name in COMPARE:
        sigma_fn, sigma_prime_fn = get_fn(name)
        col = SMOOTHING_COLORS[name]
        lbl = SMOOTHING_LABELS[name]

        for k, theta0 in enumerate(theta0_list):
            hist_theta, hist_loss = gradient_descent(
                theta0, sigma_fn, sigma_prime_fn,
                kappa, PARAMS, lr=lr, n_iter=n_iter)

            # Loss curve
            axes[0].semilogy(hist_loss,
                             color=col, alpha=0.6,
                             lw=1.5 if k == 0 else 0.8,
                             label=lbl if k == 0 else None)

            # Distance to optimum
            dist = np.linalg.norm(hist_theta - theta_star[None, :], axis=1)
            axes[1].semilogy(dist,
                             color=col, alpha=0.6,
                             lw=1.5 if k == 0 else 0.8,
                             label=lbl if k == 0 else None)

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel(r'$L_\kappa(\theta^{(n)})$')
    axes[0].set_title('Loss convergence')
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel(r'$\|\theta^{(n)} - \theta^*\|$')
    axes[1].set_title('Distance to analytical optimum')

    fig.suptitle(r'Gradient descent (FoG), $\kappa={}$, $\eta={}$'.format(
        int(kappa), lr), fontsize=11)
    fig.tight_layout()
    path = f'{OUT}/convergence.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ============================================================
# Experiment 5: Smoothing bias vs kappa
# ============================================================

def find_numerical_minimum(sigma_fn, kappa, n=50):
    """Find the numerical minimum of L_kappa by grid search."""
    T1, T2 = theta_grid(n=n, lo=-3.5, hi=3.5)
    best_L = np.inf
    best_theta = np.zeros(2)
    for i in range(n):
        for j in range(n):
            theta = np.array([T1[i, j], T2[i, j]])
            L = loss(theta, sigma_fn, kappa, PARAMS)
            if L < best_L:
                best_L = L
                best_theta = theta.copy()
    return best_theta


def plot_smoothing_bias():
    """
    Plot ||theta*_kappa - theta*|| as a function of kappa
    for each smoothing function.
    """
    kappas = np.logspace(0, 3, 20)   # 1 to 1000
    theta_star = analytical_optimum(PARAMS)

    fig, ax = plt.subplots(figsize=(7, 4))

    for name in COMPARE:
        sigma_fn, _ = get_fn(name)
        biases = []
        for k in kappas:
            theta_k = find_numerical_minimum(sigma_fn, k, n=40)
            biases.append(np.linalg.norm(theta_k - theta_star))
        ax.loglog(kappas, biases,
                  color=SMOOTHING_COLORS[name],
                  ls=SMOOTHING_LS[name],
                  lw=2,
                  label=SMOOTHING_LABELS[name])

    ax.set_xlabel(r'Stiffness $\kappa$')
    ax.set_ylabel(r'Smoothing bias $\|\theta^*_\kappa - \theta^*\|$')
    ax.set_title('Smoothing bias as a function of stiffness')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    path = f'{OUT}/smoothing_bias.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ============================================================
# Experiment 6: ZoG vs FoG at a single point
# ============================================================

def plot_zog_vs_fog(kappa=KAPPA, n_samples_list=None):
    """
    Compare ZoG and FoG estimates at a fixed theta.
    Shows how ZoG variance decreases with N, and how FoG
    compares to the FD ground truth.
    """
    if n_samples_list is None:
        n_samples_list = [10, 50, 100, 500, 1000]

    # Evaluate at a point where contact happens
    theta_eval = np.array([1.0, 0.5])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for name in COMPARE:
        sigma_fn, sigma_prime_fn = get_fn(name)
        col = SMOOTHING_COLORS[name]

        # FD ground truth
        gfd = grad_fd(theta_eval, sigma_fn, kappa, PARAMS)

        # FoG
        gfog = grad_fog(theta_eval, sigma_fn, sigma_prime_fn,
                        kappa, PARAMS)

        # ZoG at varying N
        rng = np.random.default_rng(42)
        zog_means   = []
        zog_stds    = []
        sigma_noise = 0.1

        for N in n_samples_list:
            estimates = []
            for _ in range(30):   # 30 runs for variance estimate
                g = grad_zog(theta_eval, sigma_fn, kappa, PARAMS,
                             sigma_noise=sigma_noise,
                             N_samples=N, rng=rng)
                estimates.append(g)
            estimates = np.array(estimates)
            zog_means.append(np.mean(estimates, axis=0))
            zog_stds.append(np.std(np.linalg.norm(estimates, axis=1)))

        zog_norms = [np.linalg.norm(m) for m in zog_means]

        axes[0].errorbar(n_samples_list, zog_norms,
                         yerr=zog_stds,
                         color=col, lw=1.5, capsize=3,
                         label=SMOOTHING_LABELS[name])
        axes[0].axhline(np.linalg.norm(gfd), color=col,
                        ls=':', lw=1.0)

    axes[0].set_xscale('log')
    axes[0].set_xlabel('Number of ZoG samples $N$')
    axes[0].set_ylabel(r'$\|\nabla^{[0]} F(\theta)\|$')
    axes[0].set_title('ZoG gradient norm vs sample size\n(dotted = FD reference)')
    axes[0].legend(fontsize=7)

    # Bar chart: FD vs FoG for each smoothing
    names_short = COMPARE
    fd_norms  = []
    fog_norms = []
    fog_errs  = []

    for name in names_short:
        sigma_fn, sigma_prime_fn = get_fn(name)
        gfd  = grad_fd(theta_eval, sigma_fn, kappa, PARAMS)
        gfog = grad_fog(theta_eval, sigma_fn, sigma_prime_fn,
                        kappa, PARAMS)
        fd_norms.append(np.linalg.norm(gfd))
        fog_norms.append(np.linalg.norm(gfog))
        fog_errs.append(np.linalg.norm(gfog - gfd))

    x = np.arange(len(names_short))
    w = 0.3
    axes[1].bar(x - w/2, fd_norms,  width=w, label='FD (reference)',
                color='#555', alpha=0.8)
    axes[1].bar(x + w/2, fog_norms, width=w, label='FoG (analytical)',
                color=[SMOOTHING_COLORS[n] for n in names_short],
                alpha=0.8)
    axes[1].bar(x + w/2, fog_errs,  width=w, label=r'$\|$FoG$-$FD$\|$',
                color='red', alpha=0.4, bottom=0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [SMOOTHING_LABELS[n].split('$')[0].strip() for n in names_short],
        fontsize=8)
    axes[1].set_ylabel('Gradient norm')
    axes[1].set_title(r'FD vs FoG at $\theta = (1.0, 0.5)$')
    axes[1].legend(fontsize=8)

    fig.suptitle(r'ZoG vs FoG comparison, $\kappa={}$'.format(int(kappa)),
                 fontsize=11)
    fig.tight_layout()
    path = f'{OUT}/zog_vs_fog.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ============================================================
# Experiment 7: Gradient along trajectory to optimum
# ============================================================

def plot_gradient_along_path(kappa=KAPPA, lr=0.03, n_iter=150):
    """
    Show gradient norm and FoG/FD agreement along the GD trajectory.
    """
    theta0     = np.array([0.5, 0.3])
    theta_star = analytical_optimum(PARAMS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for name in COMPARE:
        sigma_fn, sigma_prime_fn = get_fn(name)
        col = SMOOTHING_COLORS[name]
        lbl = SMOOTHING_LABELS[name]

        theta = theta0.copy()
        fog_norms = []
        fd_norms  = []
        errors    = []

        for _ in range(n_iter):
            gfd  = grad_fd(theta, sigma_fn, kappa, PARAMS)
            gfog = grad_fog(theta, sigma_fn, sigma_prime_fn,
                            kappa, PARAMS)
            fog_norms.append(np.linalg.norm(gfog))
            fd_norms.append(np.linalg.norm(gfd))
            errors.append(np.linalg.norm(gfog - gfd))
            theta = theta - lr * gfog

        iters = np.arange(n_iter)
        axes[0].semilogy(iters, fog_norms, color=col, lw=1.5, label=lbl)
        axes[0].semilogy(iters, fd_norms,  color=col, lw=1.0, ls='--',
                         alpha=0.5)
        axes[1].semilogy(iters, errors,    color=col, lw=1.5, label=lbl)

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Gradient norm')
    axes[0].set_title('FoG norm (solid) vs FD norm (dashed)')
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel(r'$\|\nabla^{[1]}L - \nabla^{FD}L\|$')
    axes[1].set_title('FoG error along GD trajectory')

    fig.suptitle(r'Gradient quality during optimisation, $\kappa={}$'.format(
        int(kappa)), fontsize=11)
    fig.tight_layout()
    path = f'{OUT}/gradient_along_path.pdf'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ============================================================
# Main: run all experiments
# ============================================================

if __name__ == '__main__':
    print('='*60)
    print('Analytical optimum:')
    theta_star = analytical_optimum(PARAMS)
    t_c = PARAMS.t_c
    mug_tc = PARAMS.mu * PARAMS.g * t_c
    print(f'  t_c = {t_c:.4f} s')
    print(f'  mu*g*t_c = {mug_tc:.4f} m/s  (sticking threshold)')
    print(f'  theta* = {theta_star}')
    print(f'  ||theta*|| = {np.linalg.norm(theta_star):.4f} m/s')
    regime = 'STICKING' if np.linalg.norm(theta_star) <= mug_tc else 'SLIDING'
    print(f'  Regime: {regime}')
    print()

    print('Running Experiment 1: Smoothing functions ...')
    plot_smoothing_functions()

    print('Running Experiment 2: Loss landscapes ...')
    plot_loss_landscapes(n=30)

    print('Running Experiment 3: Gradient comparison ...')
    plot_gradient_comparison(n=12)

    print('Running Experiment 4: Convergence ...')
    plot_convergence(n_iter=150, lr=0.03)

    print('Running Experiment 5: Smoothing bias ...')
    plot_smoothing_bias()

    print('Running Experiment 6: ZoG vs FoG ...')
    plot_zog_vs_fog()

    print('Running Experiment 7: Gradient along path ...')
    plot_gradient_along_path()

    print()
    print('All experiments complete. Figures saved to:', OUT)
