![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/sebwolf-de/b9c4e4cac4b1c91e4645e5a319e18c9a/raw/19983ec92c5951ee8eee983fb7bc51e316bc3017/seissol-matrices.json)

# SeisSol Matrices #

## Basis functions ##

We use the basis functions based on Jacobi polynomials as explained in appendix A of J. de la Puente, ‘Seismic Wave Simulation for Complex Rheologies on Unstructured Meshes’, PhD-Thesis, Ludwig-Maximilians-Universität München, Munich, 2008.


On triangles, we denote the polynomials as $\Phi_{k(p,q)}$, where $k$ is a multiindex, based on $p$ and $q$.
If we use basis functions for order $\mathcal{O}$, we have $\frac{1}{2} \times \mathcal{O} \times (\mathcal{O} + 1)$ basis functions.

On tetrahedrons, we denote the polynomials as $\Psi_{k(p,q,r)}$, where $l$ is a multiindex, based on $p$, $q$ and $r$.
If we use basis functions for order $\mathcal{O}$, we have $\frac{1}{6} \times \mathcal{O} \times (\mathcal{O} + 1) \times (\mathcal{O} + 2)$ basis functions.

## Matrices ##

### Discountinuos Galerkin matrices ###


| Notation           | Formula                                       | SeisSol            |
| ------------------ | --------------------------------------------- | ------------------ |
| $M_{kl}$           | $\int_T \Psi_k \Psi_l dx$                     | `M3`               |
|                    | $\int_T \Phi_k \Phi_l dx$                     | `M2`               |
| $F_{kl}^{-,j}$     | $\int_T \Psi_k \Psi_l dx$                     | `rT`               |
