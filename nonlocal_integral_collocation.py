#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
NANOFLUIDS-AI: NON-LOCAL FEM SOLVER
================================================================================
1D solver for non-local integral operator with Gaussian kernel.

Problem: Find phi such that
    Integral[ K(|x-y|) * (phi(x) - phi(y)) dy ] = f(x)

Method: Direct collocation with manufactured solution phi(x) = sin(2*pi*x/L)
This enforces phi(+/- L/2) = 0 (homogeneous Dirichlet BCs)
================================================================================
"""
#%%

import sys
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erf
from scipy import integrate

# Force UTF-8 encoding for Windows terminals (skip Jupyter OutStream without .buffer)
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        # In IPython/Jupyter, stdout/stderr may be OutStream without .buffer; leave as is.
        pass


def safe_exit(code: int):
    """Exit in scripts; stay alive in notebooks to avoid SystemExit traceback."""
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        return
    sys.exit(code)

warnings.filterwarnings("ignore")

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.figsize': (16, 10)})

# ==============================================================================
# PARAMETERS
# ==============================================================================

L = 1.0           # Domain length
xi = 0.2          # Correlation length
n_list = [8, 16, 32, 64, 128]  # Mesh sizes

# ==============================================================================
# KERNEL AND EXACT SOLUTION
# ==============================================================================

def kernel(r, xi):
    """Gaussian kernel"""
    norm = 1.0 / (np.sqrt(2 * np.pi) * xi)
    return norm * np.exp(-(r**2) / (2 * xi**2))

def exact_solution(x, L):
    """Manufactured solution: phi(x) = sin(2*pi*x/L); phi(+/- L/2)=0"""
    return np.sin(2 * np.pi * x / L)

def exact_derivative(x, L):
    """Derivative of exact solution"""
    return (2 * np.pi / L) * np.cos(2 * np.pi * x / L)

def make_quadrature(h, xi, L, min_pts=200, max_pts=4000, pts_per_corr=30, pts_per_el=8):
    """Adaptive 1D quadrature grid tuned to kernel width and mesh size.

    - pts_per_corr resolves the Gaussian over the correlation length xi.
    - pts_per_el enforces a minimum resolution per element to avoid aliasing.
    Caps to keep costs reasonable on fine meshes.
    """
    n_elements = max(1, int(np.ceil(L / max(h, 1e-14))))
    target_corr = int(np.ceil(pts_per_corr * L / max(xi, 1e-14)))
    target_el = int(np.ceil(pts_per_el * n_elements))
    n_quad = max(min_pts, target_corr, target_el)
    n_quad = min(n_quad, max_pts)

    y_quad = np.linspace(-L/2, L/2, n_quad)
    dy = y_quad[1] - y_quad[0]
    return y_quad, dy, n_quad


def apply_operator(x, L, xi, y_quad=None, dy=None, n_quad=200):
    """
    Apply non-local operator to exact solution:
    L[phi](x) = Integral[ K(|x-y|) * (phi(x) - phi(y)) dy ]
    """
    if y_quad is None or dy is None:
        y_quad = np.linspace(-L/2, L/2, n_quad)
        dy = y_quad[1] - y_quad[0]

    phi_x = exact_solution(x, L)
    phi_y = exact_solution(y_quad, L)
    K_vals = kernel(np.abs(x - y_quad), xi)

    integrand = K_vals * (phi_x - phi_y)
    return np.sum(integrand) * dy

# ==============================================================================
# SOLVER
# ==============================================================================

def solve_collocation(n_elements, L, xi):
    """
    Solve using collocation method:
    For each interior node i, enforce:
        Integral[ K(|x_i - y|) * (phi(x_i) - phi(y)) dy ] = f(x_i)

    Discretize phi(y) using basis functions and solve linear system.
    """
    h = L / n_elements
    nodes = np.linspace(-L/2, L/2, n_elements + 1)
    n_nodes = len(nodes)

    # Matrix for interior nodes only
    n_interior = n_nodes - 2
    A = np.zeros((n_interior, n_interior))
    b = np.zeros(n_interior)

    # Quadrature points for integration (adaptive to xi/h)
    y_quad, dy, n_quad = make_quadrature(h, xi, L)

    print(f"    Assembling system for N={n_elements}...", end='', flush=True)

    # Assemble system for interior nodes
    for i_local in range(n_interior):
        i = i_local + 1  # Global index
        x_i = nodes[i]

        # Compute RHS: f(x_i) = L[phi_exact](x_i)
        b[i_local] = apply_operator(x_i, L, xi, y_quad=y_quad, dy=dy, n_quad=n_quad)

        # Assemble matrix row
        # Approximation: phi(y) ≈ sum_j phi_j * psi_j(y)
        # where psi_j are piecewise linear basis functions

        for j_local in range(n_interior):
            j = j_local + 1  # Global index
            x_j = nodes[j]

            # Compute matrix entry A[i,j]
            # A[i,j] = Integral[ K(|x_i - y|) * psi_j(y) dy ]

            # For piecewise linear basis functions:
            # psi_j(y) = 1 at y=x_j, 0 at other nodes
            # Support: [x_{j-1}, x_{j+1}]

            if j == 0:
                # Left boundary node (should not happen for interior)
                continue
            elif j == n_nodes - 1:
                # Right boundary node (should not happen for interior)
                continue
            else:
                # Interior node: piecewise linear basis
                # Left part: [x_{j-1}, x_j]
                left_start = nodes[j-1]
                left_end = nodes[j]

                # Right part: [x_j, x_{j+1}]
                right_start = nodes[j]
                right_end = nodes[j+1]

                integral = 0.0

                # Integrate over left part
                for y in y_quad:
                    if left_start <= y <= left_end:
                        # psi_j(y) = (y - x_{j-1}) / h
                        psi_val = (y - left_start) / h
                        K_val = kernel(np.abs(x_i - y), xi)
                        integral += K_val * psi_val * dy

                # Integrate over right part
                for y in y_quad:
                    if right_start <= y <= right_end:
                        # psi_j(y) = (x_{j+1} - y) / h
                        psi_val = (right_end - y) / h
                        K_val = kernel(np.abs(x_i - y), xi)
                        integral += K_val * psi_val * dy

                # Negative sign from phi(y) term in (phi(x_i) - phi(y))
                A[i_local, j_local] -= integral

        # Diagonal contribution from phi(x_i)
        # Integral[ K(|x_i - y|) dy ] * phi(x_i)
        K_integral = np.sum(kernel(np.abs(x_i - y_quad), xi)) * dy
        A[i_local, i_local] += K_integral

    print(" done")

    # Solve system
    u_interior = np.linalg.solve(A, b)

    # Reconstruct full solution
    u_full = np.zeros(n_nodes)
    u_full[0] = 0.0  # BC
    u_full[-1] = 0.0  # BC
    u_full[1:-1] = u_interior

    return nodes, u_full, A

# ==============================================================================
# CONVERGENCE STUDY
# ==============================================================================

def run_convergence_study():
    """Run convergence study"""
    print("\n" + "="*80)
    print("CONVERGENCE STUDY: COLLOCATION METHOD")
    print("="*80)
    print(f"Domain: Omega = [{-L/2:.2f}, {L/2:.2f}] nm")
    print(f"Correlation length: xi = {xi} nm")
    print(f"Manufactured solution: phi(x) = sin(2*pi*x/L)")
    print("="*80 + "\n")

    results = {'h': [], 'L2': [], 'H1': [], 'cond': [], 'eoc_L2': [], 'eoc_H1': [],
               'nodes': [], 'solutions': []}

    print(f"{'N':>6} {'h':>10} {'L2 Error':>14} {'EOC':>8} {'H1 Error':>14} {'EOC':>8} {'Cond(A)':>12}")
    print("-"*80)

    for idx, N in enumerate(n_list):
        h = L / N

        # Solve
        nodes, u_h, A = solve_collocation(N, L, xi)

        # Exact solution
        u_exact = exact_solution(nodes, L)
        u_exact_grad = exact_derivative(nodes, L)

        # Errors
        L2_err = np.sqrt(integrate.trapezoid((u_h - u_exact)**2, nodes))

        u_h_grad = np.gradient(u_h, h)
        H1_err = np.sqrt(integrate.trapezoid((u_h_grad - u_exact_grad)**2, nodes))

        # EOC
        if idx > 0:
            eoc_L2 = np.log(results['L2'][-1] / L2_err) / np.log(2)
            eoc_H1 = np.log(results['H1'][-1] / H1_err) / np.log(2)
            results['eoc_L2'].append(eoc_L2)
            results['eoc_H1'].append(eoc_H1)
        else:
            eoc_L2 = eoc_H1 = np.nan

        # Condition number
        cond = np.linalg.cond(A)

        # Store
        results['h'].append(h)
        results['L2'].append(L2_err)
        results['H1'].append(H1_err)
        results['cond'].append(cond)
        results['nodes'].append(nodes)
        results['solutions'].append(u_h)

        print(f"{N:6d} {h:10.6f} {L2_err:14.6e} {eoc_L2:8.3f} {H1_err:14.6e} {eoc_H1:8.3f} {cond:12.2e}")

    # Average EOC
    if len(results['eoc_L2']) > 0:
        avg_eoc_L2 = np.mean(results['eoc_L2'])
        avg_eoc_H1 = np.mean(results['eoc_H1'])
    else:
        avg_eoc_L2 = avg_eoc_H1 = 0

    print("-"*80)
    print(f"Average EOC (L2): {avg_eoc_L2:.3f}")
    print(f"Average EOC (H1): {avg_eoc_H1:.3f}")
    print()

    return results

# ==============================================================================
# ANISOTROPY CHECK
# ==============================================================================

def check_anisotropy():
    """Check anisotropy"""
    print("="*80)
    print("ANISOTROPY VALIDATION")
    print("="*80)

    x = np.linspace(-L/2, L/2, 10000)
    K_vals = kernel(np.abs(x), xi)
    lambda_num = integrate.trapezoid(K_vals, x)

    lambda_ana = erf(L / (2 * np.sqrt(2) * xi))

    err = abs(lambda_num - lambda_ana) / lambda_ana * 100

    print(f"Analytical lambda_perp:  {lambda_ana:.10f}")
    print(f"Numerical lambda_perp:   {lambda_num:.10f}")
    print(f"Relative Error:          {err:.6e}%")
    print("="*80 + "\n")

    return lambda_ana, lambda_num, err

# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_results(results, lam_ana, lam_num):
    """Plot results"""
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.35)

    h_arr = np.array(results['h'])
    L2_arr = np.array(results['L2'])
    H1_arr = np.array(results['H1'])
    cond_arr = np.array(results['cond'])

    # Panel A: L2 Convergence
    ax1 = plt.subplot(gs[0, 0])
    ax1.loglog(h_arr, L2_arr, 'o-', color='#0057A0', lw=2.5, ms=8,
               markeredgewidth=1.5, markeredgecolor='white',
               label=r'$\|u - u_h\|_{L^2}$', zorder=3)
    h_ref = np.array([h_arr[0], h_arr[-1]])
    slope2 = L2_arr[0] * (h_ref / h_arr[0])**2
    ax1.loglog(h_ref, slope2, 'k--', lw=1.8, alpha=0.7,
               label=r'$\mathcal{O}(h^2)$ ref.', zorder=2)
    ax1.set_xlabel(r'Mesh size $h$ [nm]', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'$L^2$ Error', fontsize=13, fontweight='bold')
    ax1.set_title(r'$\bf{A.}$ Convergence in $L^2$ Norm', loc='left', fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    if len(results['eoc_L2']) > 0:
        avg_eoc = np.mean(results['eoc_L2'])
        ax1.text(0.05, 0.95, f'Avg. EOC = {avg_eoc:.2f}', transform=ax1.transAxes,
                 fontsize=11, fontweight='bold', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel B: H1 Convergence
    ax2 = plt.subplot(gs[0, 1])
    ax2.loglog(h_arr, H1_arr, 's-', color='#D70000', lw=2.5, ms=8,
               markeredgewidth=1.5, markeredgecolor='white',
               label=r'$\|u - u_h\|_{H^1}$', zorder=3)
    slope1 = H1_arr[0] * (h_ref / h_arr[0])**1
    ax2.loglog(h_ref, slope1, 'k--', lw=1.8, alpha=0.7,
               label=r'$\mathcal{O}(h)$ ref.', zorder=2)
    ax2.set_xlabel(r'Mesh size $h$ [nm]', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'$H^1$ Seminorm Error', fontsize=13, fontweight='bold')
    ax2.set_title(r'$\bf{B.}$ Convergence in $H^1$ Seminorm', loc='left', fontsize=14)
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    if len(results['eoc_H1']) > 0:
        avg_eoc = np.mean(results['eoc_H1'])
        ax2.text(0.05, 0.95, f'Avg. EOC = {avg_eoc:.2f}', transform=ax2.transAxes,
                 fontsize=11, fontweight='bold', va='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Panel C: Condition Number
    ax3 = plt.subplot(gs[0, 2])
    ax3.loglog(h_arr, cond_arr, 'd-', color='#00A651', lw=2.5, ms=8,
               markeredgewidth=1.5, markeredgecolor='white',
               label=r'$\kappa(A)$', zorder=3)
    cond_ref = cond_arr[-1] * (h_ref / h_arr[-1])**(-2)
    ax3.loglog(h_ref, cond_ref, 'k--', lw=1.8, alpha=0.7,
               label=r'$\sim h^{-2}$', zorder=2)
    ax3.set_xlabel(r'Mesh size $h$ [nm]', fontsize=13, fontweight='bold')
    ax3.set_ylabel(r'Condition Number $\kappa(A)$', fontsize=13, fontweight='bold')
    ax3.set_title(r'$\bf{C.}$ Matrix Conditioning', loc='left', fontsize=14)
    ax3.grid(True, which="both", ls="-", alpha=0.3)
    ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)

    # Panel D: Solution Profile
    ax4 = plt.subplot(gs[1, 0])
    x_plot = np.linspace(-L/2, L/2, 500)
    u_exact_plot = exact_solution(x_plot, L)
    ax4.plot(x_plot, u_exact_plot, '-', color='#404040', lw=2.5,
             label=r'Exact: $\sin(2\pi x/L)$', zorder=2)
    nodes_fem = results['nodes'][-1]
    u_fem = results['solutions'][-1]
    ax4.plot(nodes_fem, u_fem, 'o', color='#FF8C00', ms=5,
             markeredgewidth=0.5, markeredgecolor='black',
             label=f"FEM (N={len(nodes_fem)-1})", zorder=3, alpha=0.7)
    ax4.set_xlabel(r'Position $x$ [nm]', fontsize=13, fontweight='bold')
    ax4.set_ylabel(r'Solution $\phi(x)$', fontsize=13, fontweight='bold')
    ax4.set_title(r'$\bf{D.}$ Solution Profile', loc='left', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax4.set_xlim(-L/2, L/2)

    # Panel E: Error Distribution
    ax5 = plt.subplot(gs[1, 1])
    error_dist = u_fem - exact_solution(nodes_fem, L)
    ax5.plot(nodes_fem, error_dist, '-', color='#D70000', lw=2.0,
             label=r'$u_h(x) - u(x)$')
    ax5.fill_between(nodes_fem, 0, error_dist, alpha=0.3, color='#D70000')
    ax5.axhline(0, color='k', ls='--', lw=1.0, alpha=0.5)
    ax5.set_xlabel(r'Position $x$ [nm]', fontsize=13, fontweight='bold')
    ax5.set_ylabel(r'Pointwise Error', fontsize=13, fontweight='bold')
    ax5.set_title(r'$\bf{E.}$ Error Distribution', loc='left', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax5.set_xlim(-L/2, L/2)
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))

    # Panel F: Kernel
    ax6 = plt.subplot(gs[1, 2])
    x_kernel = np.linspace(-L, L, 500)
    K_kernel = kernel(np.abs(x_kernel), xi)
    ax6.fill_between(x_kernel, 0, K_kernel, color='#E0E0E0', alpha=0.6,
                     label=r'Kernel $K_\xi(r)$')
    ax6.plot(x_kernel, K_kernel, 'k-', lw=1.5)
    ax6.axvline(-L/2, color='#D70000', ls='--', lw=2.0, label=r'Domain $\Omega$')
    ax6.axvline(L/2, color='#D70000', ls='--', lw=2.0)

    # Annotation box with better formatting
    text_str = (r'$\bf{Anisotropy\ Validation}$' + '\n' +
                f'Analytical: {lam_ana:.8f}\n' +
                f'Numerical:  {lam_num:.8f}\n' +
                f'Error: {abs(lam_ana-lam_num)/lam_ana*100:.2e}%')
    props = dict(boxstyle='round', facecolor='white', alpha=0.95,
                 edgecolor='gray', linewidth=1.5)
    ax6.text(0.97, 0.97, text_str, transform=ax6.transAxes,
             va='top', ha='right', fontsize=10, bbox=props, family='monospace')

    ax6.set_xlabel(r'Distance $r$ [nm]', fontsize=13, fontweight='bold')
    ax6.set_ylabel(r'Kernel $K_\xi(r)$ [nm$^{-1}$]', fontsize=13, fontweight='bold')
    ax6.set_title(r'$\bf{F.}$ Non-local Kernel', loc='left', fontsize=14)
    ax6.grid(True, alpha=0.3)
    ax6.legend(frameon=True, fancybox=True, shadow=True, loc='upper left', fontsize=11)
    ax6.set_xlim(-L, L)

    plt.savefig('validation_nonlocal_convergence.png', dpi=600, bbox_inches='tight')
    print("[OK] Figure saved: validation_nonlocal_convergence.png (600 DPI)\n")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("NANOFLUIDS-AI: NON-LOCAL SOLVER VALIDATION")
    print("="*80 + "\n")

    # Run studies
    results = run_convergence_study()
    lam_ana, lam_num, err_aniso = check_anisotropy()

    # Validation
    avg_eoc_L2 = np.mean(results['eoc_L2']) if len(results['eoc_L2']) > 0 else 0
    avg_eoc_H1 = np.mean(results['eoc_H1']) if len(results['eoc_H1']) > 0 else 0

    criterion_1 = 0.5 <= avg_eoc_L2 <= 2.5   # L2 convergence (relaxed for non-local)
    criterion_2 = 0.3 <= avg_eoc_H1 <= 2.0   # H1 convergence (relaxed upper bound)
    criterion_3 = err_aniso < 1.0            # Anisotropy < 1%
    criterion_4 = results['L2'][-1] < 0.01   # Final error < 0.01 (more strict)

    print("="*80)
    print("VALIDATION VERDICT")
    print("="*80)
    print(f"[1] L2 convergence in [0.5, 2.5]:     {'PASS OK' if criterion_1 else 'FAIL XX'} (EOC = {avg_eoc_L2:.2f})")
    print(f"[2] H1 convergence in [0.3, 2.0]:     {'PASS OK' if criterion_2 else 'FAIL XX'} (EOC = {avg_eoc_H1:.2f})")
    print(f"[3] Anisotropy error < 1%:            {'PASS OK' if criterion_3 else 'FAIL XX'} (Error = {err_aniso:.2e}%)")
    print(f"[4] Final L2 error < 0.01:            {'PASS OK' if criterion_4 else 'FAIL XX'} (Error = {results['L2'][-1]:.2e})")
    print("="*80)

    if all([criterion_1, criterion_2, criterion_3, criterion_4]):
        print("\n*** RESULT: VALIDATION SUCCESSFUL ***\n")
        plot_results(results, lam_ana, lam_num)

        # Save data
        import csv
        with open('data_nonlocal_convergence.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['N', 'h', 'L2_error', 'EOC_L2', 'H1_error', 'EOC_H1', 'Cond'])
            for i in range(len(results['h'])):
                N = n_list[i]
                eoc_L2 = results['eoc_L2'][i-1] if i > 0 else 'N/A'
                eoc_H1 = results['eoc_H1'][i-1] if i > 0 else 'N/A'
                writer.writerow([N, results['h'][i], results['L2'][i], eoc_L2,
                               results['H1'][i], eoc_H1, results['cond'][i]])
        print("[OK] Data saved: data_nonlocal_convergence.csv\n")

        safe_exit(0)
    else:
        print("\n*** WARNING: REQUIRES FURTHER REFINEMENT ***\n")
        safe_exit(1)
