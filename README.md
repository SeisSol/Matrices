# SeisSol Matrices #

## Basis functions ##

We use the basis functions based on Jacobi polynomials as explained in appendix A of J. de la Puente, ‘Seismic Wave Simulation for Complex Rheologies on Unstructured Meshes’, PhD-Thesis, Ludwig-Maximilians-Universität München, Munich, 2008.


On triangles, we denote the polynomials as $\Phi_{k(p,q)}$, where $k$ is a multiindex, based on $p$ and $q$.
If we use basis functions for order $\mathcal{O}$, we have $\frac{1}{2} \times \mathcal{O} \times (\mathcal{O} + 1)$ basis functions.

On tetrahedrons, we denote the polynomials as $\Psi_{k(p,q,r)}$, where $l$ is a multiindex, based on $p$, $q$ and $r$.
If we use basis functions for order $\mathcal{O}$, we have $\frac{1}{6} \times \mathcal{O} \times (\mathcal{O} + 1) \times (\mathcal{O} + 2)$ basis functions.

## Matrices ##

We use $T^3$ to denote the unit tetrahedron and $T^2$ to denote the unit triangle.
With $T^2_j$, we denote the $j$ th face of the unit tetrahedron.

### Discountinuos Galerkin matrices ###


| Notation           | Formula                                       | SeisSol            |
| ------------------ | --------------------------------------------- | ------------------ |
| $M_{kl}$           | $\int_{T^3} \Psi_k \Psi_l dx$                 | `M3`               |
|                    | $\int_{T^2} \Phi_k \Phi_l dx$                 | `M2`               |
| $F_{kl}^{-,j}$     | $\int_{T^2_j} \Psi_k \Psi_l dx$               | `rT`               |
