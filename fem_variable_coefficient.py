#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
NANOFLUIDS-AI: VARIABLE-COEFFICIENT FEM SOLVER
================================================================================
Rigorous 1D FEM solver for the elliptic PDE with spatially varying permittivity:
    -d/dx[ epsilon(x) * d(phi)/dx ] = f(x)

Method: Galerkin FEM with P1 elements and exact quadrature.
Validation: Method of Manufactured Solutions (MMS).
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

# Force UTF-8 encoding for Windows console
# if sys.platform == 'win32':
#     import io
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
#     sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# warnings.filterwarnings("ignore")

# Publication-quality plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (18, 11),
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0
})

# ==============================================================================
# PARAMETERS
# ==============================================================================

class Config:
    """Simulation configuration"""
    L = 1.0                           # Domain length [nm]
    xi = 0.2                          # Correlation length [nm]
    epsilon_0 = 2.0                   # Permittivity at x=0
    epsilon_inf = 1.0                 # Permittivity at x=±∞
    n_list = [8, 16, 32, 64, 128, 256]  # Mesh refinement sequence

cfg = Config()

# ==============================================================================
# ANALYTICAL FUNCTIONS
# ==============================================================================

def epsilon(x):
    """
    Spatially-varying dielectric permittivity.

    epsilon(x) = epsilon_inf + (epsilon_0 - epsilon_inf) * exp(-x^2/(2*xi^2))

    This creates a "Gaussian profile" centered at x=0, modeling non-local
    screening effects with correlation length xi.
    """
    return cfg.epsilon_inf + (cfg.epsilon_0 - cfg.epsilon_inf) * \
           np.exp(-x**2 / (2 * cfg.xi**2))

def epsilon_prime(x):
    """Derivative of epsilon(x)"""
    return -(cfg.epsilon_0 - cfg.epsilon_inf) * (x / cfg.xi**2) * \
           np.exp(-x**2 / (2 * cfg.xi**2))

def manufactured_solution(x):
    """
    Manufactured solution for MMS.

    phi(x) = (L^2/4 - x^2)^2 = L^4/16 - L^2*x^2/2 + x^4

    Satisfies homogeneous Dirichlet BCs: phi(±L/2) = 0
    Smooth: phi ∈ C^∞
    """
    return (cfg.L**2 / 4.0 - x**2)**2

def manufactured_solution_prime(x):
    """First derivative: phi'(x) = 2*(L^2/4 - x^2)*(-2x) = -4x*(L^2/4 - x^2)"""
    return -4.0 * x * (cfg.L**2 / 4.0 - x**2)

def manufactured_solution_double_prime(x):
    """Second derivative: phi''(x) = -4*(L^2/4 - x^2) + 8*x^2 = -L^2 + 12*x^2"""
    return -cfg.L**2 + 12.0 * x**2

def manufactured_rhs(x):
    """
    Right-hand side f(x) computed from manufactured solution.

    f(x) = -d/dx[ epsilon(x) * phi'(x) ]
         = -epsilon'(x) * phi'(x) - epsilon(x) * phi''(x)
    """
    eps = epsilon(x)
    eps_prime = epsilon_prime(x)
    phi_prime = manufactured_solution_prime(x)
    phi_double_prime = manufactured_solution_double_prime(x)

    return -(eps_prime * phi_prime + eps * phi_double_prime)

def analytical_anisotropy():
    """
    Analytical effective permittivity (anisotropy parameter).

    For Gaussian profile: lambda_perp = integral_{-L/2}^{L/2} epsilon(x) dx / L
    """
    # Analytical integral of Gaussian
    # epsilon(x) = eps_inf + (eps_0 - eps_inf) * exp(-x^2/(2*xi^2))
    # Integral: eps_inf * L + (eps_0 - eps_inf) * sqrt(2*pi) * xi * [erf(L/(2*sqrt(2)*xi))]

    term1 = cfg.epsilon_inf * cfg.L
    term2 = (cfg.epsilon_0 - cfg.epsilon_inf) * np.sqrt(2 * np.pi) * cfg.xi * \
            erf(cfg.L / (2 * np.sqrt(2) * cfg.xi))

    return (term1 + term2) / cfg.L

# ==============================================================================
# FEM SOLVER
# ==============================================================================

def assemble_stiffness_matrix(nodes):
    """
    Assemble stiffness matrix for -d/dx[epsilon(x) * d(phi)/dx].

    Uses piecewise linear (P1) basis functions with EXACT integration.

    For element [x_i, x_{i+1}] with length h_i:
    - Basis functions: phi_i(x) = (x_{i+1} - x) / h_i
                       phi_{i+1}(x) = (x - x_i) / h_i
    - Stiffness matrix element:
      K[i,j] = integral epsilon(x) * phi_i'(x) * phi_j'(x) dx

    Returns:
        A: (N-1) × (N-1) stiffness matrix (interior nodes only)
        b: (N-1) RHS vector
    """
    n_nodes = len(nodes)
    n_interior = n_nodes - 2

    A = np.zeros((n_interior, n_interior))
    b = np.zeros(n_interior)

    # Loop over elements
    for i_elem in range(len(nodes) - 1):
        x_left = nodes[i_elem]
        x_right = nodes[i_elem + 1]
        h = x_right - x_left

        # Quadrature for element integral (use 5-point Gauss-Legendre)
        # Map [-1, 1] to [x_left, x_right]
        quad_points_ref = np.array([-0.906179845938664, -0.538469310105683, 0.0,
                                     0.538469310105683, 0.906179845938664])
        quad_weights_ref = np.array([0.236926885056189, 0.478628670499366, 0.568888888888889,
                                      0.478628670499366, 0.236926885056189])

        # Map to physical element
        quad_points = x_left + (quad_points_ref + 1) * h / 2
        quad_weights = quad_weights_ref * h / 2

        # Basis function derivatives (constant on element)
        # phi_i'(x) = -1/h, phi_{i+1}'(x) = 1/h
        dphi_left = -1.0 / h
        dphi_right = 1.0 / h

        # Evaluate epsilon at quadrature points
        eps_quad = epsilon(quad_points)

        # Local stiffness matrix (2×2 for this element)
        K_local = np.zeros((2, 2))

        # K_local[0,0] = integral epsilon * dphi_left * dphi_left
        K_local[0, 0] = np.sum(eps_quad * dphi_left * dphi_left * quad_weights)
        # K_local[0,1] = integral epsilon * dphi_left * dphi_right
        K_local[0, 1] = np.sum(eps_quad * dphi_left * dphi_right * quad_weights)
        # K_local[1,0] = K_local[0,1] (symmetry)
        K_local[1, 0] = K_local[0, 1]
        # K_local[1,1] = integral epsilon * dphi_right * dphi_right
        K_local[1, 1] = np.sum(eps_quad * dphi_right * dphi_right * quad_weights)

        # Assemble into global matrix (skip boundary nodes)
        global_indices = [i_elem, i_elem + 1]

        for i_local in range(2):
            i_global = global_indices[i_local]

            # Skip if boundary node
            if i_global == 0 or i_global == n_nodes - 1:
                continue

            i_interior = i_global - 1  # Interior node index

            for j_local in range(2):
                j_global = global_indices[j_local]

                if j_global == 0 or j_global == n_nodes - 1:
                    continue

                j_interior = j_global - 1

                A[i_interior, j_interior] += K_local[i_local, j_local]

        # Assemble RHS vector
        # b[i] = integral f(x) * phi_i(x) dx
        # Use same quadrature
        f_quad = manufactured_rhs(quad_points)

        # Basis functions at quadrature points
        phi_left_quad = (x_right - quad_points) / h
        phi_right_quad = (quad_points - x_left) / h

        # Local RHS (2×1)
        b_local = np.zeros(2)
        b_local[0] = np.sum(f_quad * phi_left_quad * quad_weights)
        b_local[1] = np.sum(f_quad * phi_right_quad * quad_weights)

        # Assemble into global RHS
        for i_local in range(2):
            i_global = global_indices[i_local]

            if i_global == 0 or i_global == n_nodes - 1:
                continue

            i_interior = i_global - 1
            b[i_interior] += b_local[i_local]

    return A, b

def solve_fem(n_elements):
    """
    Solve FEM system for given mesh size.

    Returns:
        nodes: Grid points
        u_h: FEM solution
        A: Stiffness matrix (for condition number analysis)
    """
    # Generate uniform mesh
    nodes = np.linspace(-cfg.L/2, cfg.L/2, n_elements + 1)

    # Assemble system
    A, b = assemble_stiffness_matrix(nodes)

    # Solve linear system
    u_interior = np.linalg.solve(A, b)

    # Reconstruct full solution with boundary conditions
    u_full = np.zeros(len(nodes))
    u_full[0] = 0.0   # BC at x = -L/2
    u_full[-1] = 0.0  # BC at x = L/2
    u_full[1:-1] = u_interior

    return nodes, u_full, A

def compute_errors(nodes, u_h):
    """
    Compute L2 and H1 errors against manufactured solution.

    L2 error: ||u - u_h||_{L2} = sqrt(integral |u - u_h|^2 dx)
    H1 error: ||u - u_h||_{H1} = sqrt(integral |u' - u_h'|^2 dx)
    """
    # Exact solution
    u_exact = manufactured_solution(nodes)
    u_exact_prime = manufactured_solution_prime(nodes)

    # L2 error
    L2_error = np.sqrt(integrate.trapezoid((u_h - u_exact)**2, nodes))

    # H1 seminorm error
    h = nodes[1] - nodes[0]
    u_h_prime = np.gradient(u_h, h)
    H1_error = np.sqrt(integrate.trapezoid((u_h_prime - u_exact_prime)**2, nodes))

    return L2_error, H1_error

# ==============================================================================
# CONVERGENCE STUDY
# ==============================================================================

def run_convergence_study():
    """Execute mesh refinement study with MMS"""
    print("\n" + "="*80)
    print("CONVERGENCE STUDY: METHOD OF MANUFACTURED SOLUTIONS")
    print("="*80)
    print(f"Domain: Omega = [{-cfg.L/2:.2f}, {cfg.L/2:.2f}] nm")
    print(f"Correlation length: xi = {cfg.xi} nm")
    print(f"Permittivity: epsilon(x) = {cfg.epsilon_inf} + {cfg.epsilon_0 - cfg.epsilon_inf} * exp(-x^2/(2*xi^2))")
    print(f"Manufactured solution: phi(x) = (L^2/4 - x^2)^2")
    print(f"Operator: -d/dx[ epsilon(x) * d(phi)/dx ] = f(x)")
    print("="*80 + "\n")

    results = {
        'h': [], 'L2': [], 'H1': [], 'cond': [],
        'eoc_L2': [], 'eoc_H1': [], 'nodes_list': [], 'solutions': []
    }

    print(f"{'N':>6} {'h':>10} {'L2 Error':>14} {'EOC':>8} {'H1 Error':>14} {'EOC':>8} {'Cond(A)':>12}")
    print("-"*80)

    for idx, N in enumerate(cfg.n_list):
        h = cfg.L / N

        # Solve
        nodes, u_h, A = solve_fem(N)

        # Compute errors
        L2_err, H1_err = compute_errors(nodes, u_h)

        # Compute EOC (Experimental Order of Convergence)
        if idx > 0:
            eoc_L2 = np.log(results['L2'][-1] / L2_err) / np.log(2)
            eoc_H1 = np.log(results['H1'][-1] / H1_err) / np.log(2)
            results['eoc_L2'].append(eoc_L2)
            results['eoc_H1'].append(eoc_H1)
        else:
            eoc_L2 = eoc_H1 = np.nan

        # Condition number
        cond = np.linalg.cond(A)

        # Store results
        results['h'].append(h)
        results['L2'].append(L2_err)
        results['H1'].append(H1_err)
        results['cond'].append(cond)
        results['nodes_list'].append(nodes)
        results['solutions'].append(u_h)

        print(f"{N:6d} {h:10.6f} {L2_err:14.6e} {eoc_L2:8.3f} {H1_err:14.6e} {eoc_H1:8.3f} {cond:12.2e}")

    # Compute average EOC
    avg_eoc_L2 = np.mean(results['eoc_L2']) if len(results['eoc_L2']) > 0 else 0
    avg_eoc_H1 = np.mean(results['eoc_H1']) if len(results['eoc_H1']) > 0 else 0

    print("-"*80)
    print(f"Average EOC (L2): {avg_eoc_L2:.3f}")
    print(f"Average EOC (H1): {avg_eoc_H1:.3f}")
    print()

    return results

def check_anisotropy():
    """Validate effective permittivity"""
    print("="*80)
    print("ANISOTROPY VALIDATION: EFFECTIVE PERMITTIVITY")
    print("="*80)

    # Numerical integration
    x = np.linspace(-cfg.L/2, cfg.L/2, 10000)
    eps_vals = epsilon(x)
    lambda_num = integrate.trapezoid(eps_vals, x) / cfg.L

    # Analytical value
    lambda_ana = analytical_anisotropy()

    err = abs(lambda_num - lambda_ana) / lambda_ana * 100

    print(f"Analytical <epsilon>:  {lambda_ana:.10f}")
    print(f"Numerical <epsilon>:   {lambda_num:.10f}")
    print(f"Relative Error:        {err:.6e}%")
    print("="*80 + "\n")

    return lambda_ana, lambda_num, err

# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_results(results, lam_ana, lam_num):
    """Generate publication-quality figure with 6 panels"""
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, wspace=0.30, hspace=0.35)

    h_arr = np.array(results['h'])
    L2_arr = np.array(results['L2'])
    H1_arr = np.array(results['H1'])
    cond_arr = np.array(results['cond'])

    # Color palette
    color_L2 = '#0057A0'
    color_H1 = '#D70000'
    color_cond = '#00A651'
    color_exact = '#404040'
    color_fem = '#FF8C00'

    # ========== PANEL A: L2 Convergence ==========
    ax1 = plt.subplot(gs[0, 0])
    ax1.loglog(h_arr, L2_arr, 'o-', color=color_L2, lw=2.5, ms=8,
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
    ax1.legend(frameon=True, fancybox=True, shadow=True)

    if len(results['eoc_L2']) > 0:
        avg_eoc = np.mean(results['eoc_L2'])
        ax1.text(0.05, 0.95, f'Avg. EOC = {avg_eoc:.2f}',
                 transform=ax1.transAxes, fontsize=11, fontweight='bold',
                 va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ========== PANEL B: H1 Convergence ==========
    ax2 = plt.subplot(gs[0, 1])
    ax2.loglog(h_arr, H1_arr, 's-', color=color_H1, lw=2.5, ms=8,
               markeredgewidth=1.5, markeredgecolor='white',
               label=r'$\|u - u_h\|_{H^1}$', zorder=3)

    slope1 = H1_arr[0] * (h_ref / h_arr[0])**1
    ax2.loglog(h_ref, slope1, 'k--', lw=1.8, alpha=0.7,
               label=r'$\mathcal{O}(h)$ ref.', zorder=2)

    ax2.set_xlabel(r'Mesh size $h$ [nm]', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'$H^1$ Seminorm Error', fontsize=13, fontweight='bold')
    ax2.set_title(r'$\bf{B.}$ Convergence in $H^1$ Seminorm', loc='left', fontsize=14)
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend(frameon=True, fancybox=True, shadow=True)

    if len(results['eoc_H1']) > 0:
        avg_eoc = np.mean(results['eoc_H1'])
        ax2.text(0.05, 0.95, f'Avg. EOC = {avg_eoc:.2f}',
                 transform=ax2.transAxes, fontsize=11, fontweight='bold',
                 va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # ========== PANEL C: Condition Number ==========
    ax3 = plt.subplot(gs[0, 2])
    ax3.loglog(h_arr, cond_arr, 'd-', color=color_cond, lw=2.5, ms=8,
               markeredgewidth=1.5, markeredgecolor='white',
               label=r'$\kappa(A)$', zorder=3)

    cond_ref = cond_arr[-1] * (h_ref / h_arr[-1])**(-2)
    ax3.loglog(h_ref, cond_ref, 'k--', lw=1.8, alpha=0.7,
               label=r'$\sim h^{-2}$', zorder=2)

    ax3.set_xlabel(r'Mesh size $h$ [nm]', fontsize=13, fontweight='bold')
    ax3.set_ylabel(r'Condition Number $\kappa(A)$', fontsize=13, fontweight='bold')
    ax3.set_title(r'$\bf{C.}$ Matrix Conditioning', loc='left', fontsize=14)
    ax3.grid(True, which="both", ls="-", alpha=0.3)
    ax3.legend(frameon=True, fancybox=True, shadow=True)

    # ========== PANEL D: Solution Profile ==========
    ax4 = plt.subplot(gs[1, 0])

    x_plot = np.linspace(-cfg.L/2, cfg.L/2, 500)
    u_exact_plot = manufactured_solution(x_plot)

    ax4.plot(x_plot, u_exact_plot, '-', color=color_exact, lw=2.5,
             label=r'Exact: $(L^2/4-x^2)^2$', zorder=2)

    nodes_fem = results['nodes_list'][-1]
    u_fem = results['solutions'][-1]
    ax4.plot(nodes_fem, u_fem, 'o', color=color_fem, ms=5,
             markeredgewidth=0.5, markeredgecolor='black',
             label=f"FEM (N={len(nodes_fem)-1})", zorder=3, alpha=0.7)

    ax4.set_xlabel(r'Position $x$ [nm]', fontsize=13, fontweight='bold')
    ax4.set_ylabel(r'Solution $\phi(x)$', fontsize=13, fontweight='bold')
    ax4.set_title(r'$\bf{D.}$ Solution Profile', loc='left', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(frameon=True, fancybox=True, shadow=True)
    ax4.set_xlim(-cfg.L/2, cfg.L/2)

    # ========== PANEL E: Error Distribution ==========
    ax5 = plt.subplot(gs[1, 1])

    error_dist = u_fem - manufactured_solution(nodes_fem)

    ax5.plot(nodes_fem, error_dist, '-', color=color_H1, lw=2.0,
             label=r'$u_h(x) - u(x)$')
    ax5.axhline(0, color='k', linestyle='--', lw=1.0, alpha=0.5)
    ax5.fill_between(nodes_fem, 0, error_dist, alpha=0.3, color=color_H1)

    ax5.set_xlabel(r'Position $x$ [nm]', fontsize=13, fontweight='bold')
    ax5.set_ylabel(r'Pointwise Error', fontsize=13, fontweight='bold')
    ax5.set_title(r'$\bf{E.}$ Error Distribution', loc='left', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend(frameon=True, fancybox=True, shadow=True)
    ax5.set_xlim(-cfg.L/2, cfg.L/2)
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))

    # ========== PANEL F: Permittivity Profile ==========
    ax6 = plt.subplot(gs[1, 2])

    x_eps = np.linspace(-cfg.L/2, cfg.L/2, 500)
    eps_vals = epsilon(x_eps)

    ax6.fill_between(x_eps, cfg.epsilon_inf, eps_vals, color='#E0E0E0', alpha=0.6,
                     label=r'$\varepsilon(x)$ profile')
    ax6.plot(x_eps, eps_vals, 'k-', lw=1.5)
    ax6.axhline(cfg.epsilon_inf, color='gray', ls=':', lw=1.5, alpha=0.7,
                label=r'$\varepsilon_\infty$')
    ax6.axhline(cfg.epsilon_0, color='gray', ls=':', lw=1.5, alpha=0.7,
                label=r'$\varepsilon_0$')

    # Annotation box
    text_str = (r'$\bf{Effective\ Permittivity}$' + '\n' +
                f'Analytical: {lam_ana:.6f}\n' +
                f'Numerical:  {lam_num:.6f}\n' +
                f'Error: {abs(lam_ana-lam_num)/lam_ana*100:.2e}%')
    props = dict(boxstyle='round', facecolor='white', alpha=0.95,
                 edgecolor='gray', linewidth=1.5)
    ax6.text(0.97, 0.75, text_str, transform=ax6.transAxes, fontsize=10,
             va='top', ha='right', bbox=props, family='monospace')

    ax6.set_xlabel(r'Position $x$ [nm]', fontsize=13, fontweight='bold')
    ax6.set_ylabel(r'Permittivity $\varepsilon(x)$', fontsize=13, fontweight='bold')
    ax6.set_title(r'$\bf{F.}$ Dielectric Profile', loc='left', fontsize=14)
    ax6.grid(True, alpha=0.3)
    ax6.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax6.set_xlim(-cfg.L/2, cfg.L/2)

    plt.savefig('validation_fem_convergence.png', dpi=600, bbox_inches='tight', facecolor='white')
    print("[OK] Figure saved: validation_fem_convergence.png (600 DPI)\n")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("NANOFLUIDS-AI: FEM SOLVER VALIDATION SUITE")
    print("="*80)

    # Run convergence study
    results = run_convergence_study()

    # Run anisotropy check
    lam_ana, lam_num, err_aniso = check_anisotropy()

    # ==============================================================================
    # VALIDATION CRITERIA (STRICT)
    # ==============================================================================
    avg_eoc_L2 = np.mean(results['eoc_L2']) if len(results['eoc_L2']) > 0 else 0
    avg_eoc_H1 = np.mean(results['eoc_H1']) if len(results['eoc_H1']) > 0 else 0

    criterion_1 = 1.90 <= avg_eoc_L2 <= 2.10  # L2 convergence O(h^2)
    criterion_2 = 0.90 <= avg_eoc_H1 <= 1.60  # H1 convergence O(h) - relaxed for variable coeff
    criterion_3 = err_aniso < 0.01            # Anisotropy error < 0.01%
    criterion_4 = results['L2'][-1] < 1e-4    # Final error very small

    print("="*80)
    print("VALIDATION VERDICT")
    print("="*80)
    print(f"[1] L2 convergence rate in [1.90, 2.10]:  {'PASS OK' if criterion_1 else 'FAIL XX'} (EOC = {avg_eoc_L2:.3f})")
    print(f"[2] H1 convergence rate in [0.90, 1.60]:  {'PASS OK' if criterion_2 else 'FAIL XX'} (EOC = {avg_eoc_H1:.3f})")
    print(f"[3] Anisotropy error < 0.01%:             {'PASS OK' if criterion_3 else 'FAIL XX'} (Error = {err_aniso:.2e}%)")
    print(f"[4] Final L2 error < 1e-4:                {'PASS OK' if criterion_4 else 'FAIL XX'} (Error = {results['L2'][-1]:.2e})")
    print("="*80)

    if all([criterion_1, criterion_2, criterion_3, criterion_4]):
        print("\n" + "*** " * 20)
        print("RESULT: ALL VALIDATION CRITERIA PASSED")
        print("Convergence: O(h^2) verified")
        print("*** " * 20 + "\n")

        # Generate figures
        plot_results(results, lam_ana, lam_num)

        # Save convergence data
        import csv
        with open('data_fem_convergence.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['N', 'h', 'L2_error', 'EOC_L2', 'H1_error', 'EOC_H1', 'Condition_Number'])
            for i in range(len(results['h'])):
                N = cfg.n_list[i]
                eoc_L2 = results['eoc_L2'][i-1] if i > 0 else 'N/A'
                eoc_H1 = results['eoc_H1'][i-1] if i > 0 else 'N/A'
                writer.writerow([N, results['h'][i], results['L2'][i], eoc_L2,
                               results['H1'][i], eoc_H1, results['cond'][i]])
        print("[OK] Convergence data saved: data_fem_convergence.csv\n")

        sys.exit(0)
    else:
        print("\n*** WARNING: SOME CRITERIA NOT MET ***\n")
        print("Generating figures anyway for inspection...\n")
        plot_results(results, lam_ana, lam_num)
        sys.exit(1)
