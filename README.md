# NanoFluids-AI: Numerical Validation Suite (Non-local & FEM)

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![Method: Numerical Analysis](https://img.shields.io/badge/Method-Numerical_Analysis-purple.svg)
![Status: Validated](https://img.shields.io/badge/Status-Validated-brightgreen.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## 1. Scientific Overview

This repository contains the **Numerical Validation Suite** for the NanoFluids-AI framework. It addresses the computational challenge of transitioning from classical local continuum mechanics to the **non-local integral operators** required to model nanofluidic transport in strong confinement.

The suite implements and compares two distinct solver architectures:

1. **Local Baseline:** A rigorous Finite Element Method (FEM) solver for variable-coefficient elliptic PDEs, serving as the accuracy benchmark.
2. **Non-Local Target:** A specialised Collocation solver for integro-differential equations with Gaussian kernels, validating the stability of non-local constitutive laws.

---

## 2. Mathematical Formulation

### A. The Local Problem (Variable Coefficient)

To model inhomogeneous dielectric screening in the continuum limit, we solve the divergence-form elliptic equation:

$$
-\frac{d}{dx} \left( \epsilon(x) \frac{d\phi}{dx} \right) = f(x) \quad \text{in } \Omega
$$

**Implementation:**

* **Method:** Galerkin Finite Element Method (FEM).
* **Discretization:** Piecewise linear ($P_1$) Lagrange basis functions.
* **Integration:** Exact Gaussian quadrature (5-point) to handle the spatially varying permittivity $\epsilon(x)$.

### B. The Non-Local Problem (Integral Operator)

To capture long-range correlations ($\xi \sim L$), we solve the non-local constitutive equation involving a convolution kernel $K_\xi(|x-y|)$:

$$
\int_{\Omega} K_\xi(|x-y|) (\phi(x) - \phi(y)) \, dy = f(x)
$$

**Implementation:**

* **Method:** Collocation Method with adaptive quadrature.
* **Kernel:** Gaussian kernel $K_\xi(r) \propto \exp(-r^2/2\xi^2)$, representing the non-local dielectric response function.
* **Challenge:** Unlike the sparse matrices of FEM, this operator yields dense stiffness matrices requiring specialized stabilization.

---

## 3. Validation Results

### Subsection A: Variable-Coefficient FEM (Baseline)

We validate the local solver using the Method of Manufactured Solutions (MMS). The results confirm theoretical optimality.

<p align="center">
  <img src="validation_fem_convergence.png" alt="FEM Convergence Analysis" width="100%">
</p>

> **Figure 1: Convergence analysis for the local variable-coefficient solver.**
>
> * **Panel A:** Achieves optimal **$O(h^2)$ convergence** in the $L^2$ norm (slope = 2.00).
> * **Panel B:** Achieves **$O(h)$ convergence** in the $H^1$ seminorm, consistent with Céa's Lemma for $P_1$ elements.
> * **Panel C:** The condition number scales as $\kappa(A) \sim h^{-2}$, typical for second-order elliptic operators.

### Subsection B: Non-Local Integral Solver (Target)

We validate the non-local collocation scheme against a manufactured integral solution to ensure stability against spectral pollution.

<p align="center">
  <img src="validation_nonlocal_convergence.png" alt="Non-local Solver Convergence" width="100%">
</p>

> **Figure 2: Convergence of the non-local collocation scheme.**
>
> * **Stability:** The solver demonstrates consistent error reduction without the oscillations (Gibbs phenomenon) often associated with integral operators on bounded domains.
> * **Accuracy:** The method successfully resolves the non-local kernel effects even when the correlation length $\xi$ is comparable to the mesh size $h$.

---

## 4. Comparative Analysis

The comparison between the two solvers highlights the fundamental shift in computational architecture required for nanofluidics:

1. **Sparsity vs. Density:** The Local FEM (Fig 1) produces sparse tridiagonal matrices, whereas the Non-Local Solver (Fig 2) generates dense matrices due to long-range coupling.
2. **Convergence Regimes:** While the Local FEM follows strict polynomial convergence ($h^2$), the Non-Local solver's accuracy is governed by the interplay between mesh size $h$ and correlation length $\xi$.
3. **Conclusion:** The successful validation of the Non-Local solver confirms that **integro-differential constitutive laws** can be solved with comparable robustness to standard PDEs, provided adaptive quadrature is employed to resolve the kernel singularity.

> **Note:** This validation suite employs a Collocation method for rapid prototyping and de-risking. The production implementation (T4.1) uses a Conforming Galerkin scheme with $\mathcal{H}$-matrix compression for $\mathcal{O}(N \log N)$ complexity.

---

## 5. Repository Structure

```text
.
├── fem_variable_coefficient.py       # SCRIPT A: Local FEM Solver (Reference)
├── nonlocal_integral_collocation.py  # SCRIPT B: Non-Local Integral Solver (Target)
├── validation_fem_convergence.png    # Output Figure A
├── validation_nonlocal_convergence.png # Output Figure B
├── data_fem_convergence.csv          # Raw convergence data (FEM)
├── data_nonlocal_convergence.csv     # Raw convergence data (Non-local)
├── requirements.txt                  # Dependencies
└── README.md                         # Documentation
```

---

## 6. Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Solvers

Execute the scripts to reproduce the convergence studies and generate the figures:

**1. Local FEM Baseline:**

```bash
python fem_variable_coefficient.py
```

**2. Non-Local Integral Target:**

```bash
python nonlocal_integral_collocation.py
```

---

## 7. Citation

If you use these solvers or the validation methodology, please cite:

```bibtex
@software{nanofluids_ai_numerical_2025,
  author = {NanoFluids-AI Computational Team},
  title = {NanoFluids-AI: Numerical Validation Suite (Non-local & FEM)},
  version = {1.0.0},
  year = {2025},
  url = {https://github.com/renee29/NanoFluids-AI-Numerical-Solvers},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries, please open a GitHub issue or contact R. Fabregas (NanoFluids-AI Team, rfabregas@ugr.es).

---

**Project Status**: Initial release (v1.0.0) - December 2025
