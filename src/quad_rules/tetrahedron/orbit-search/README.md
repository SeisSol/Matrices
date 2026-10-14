# Symmetric rules for the tetrahedron found by a search over orbit structures

Fully symmetric rules with positive weights and all barycentric coordinates above 1e-3,
with fewer points than the other rules of this kind and degree in `quad_rules`:

| Degree | Points | Orbits | Smallest barycentric coordinate | Other fully symmetric rules with positive weights |
| ------ | ------ | ------ | ------------------------------- | ------------------------------------------------- |
| 10 | 79 | S1x1 S4x3 S6x1 S12x5 | 1.06e-2 | Witherden–Vincent: 81 points (smallest barycentric coordinate 6.1e-3) |

They were computed for SeisSol with `tools/search_symmetric_tet_rule.py`: for a given number of orbits of
each kind, it solves the moment equations from many random starts, keeping the weights positive and the
points inside. The structure of degree 10 has one unknown more than there are equations, so its solutions
form a one-parameter family; the rule here is the member whose smallest barycentric coordinate is largest.
The result is refined to 40 digits; the moments of all monomials up to the degree are exact to a relative
error below 1e-40. Each file lists its orbits in the header.

A rule of degree 10 with 79 points has also been published by G. Chuluunbaatar, O. Chuluunbaatar,
A. A. Gusev, S. I. Vinitsky, PI-type fully symmetric quadrature rules on the 3-, ..., 6-simplexes,
Computers & Mathematics with Applications 124, 89–97, 2022, and one with the same orbits, another member of
the family, by Z. A. Worku, J. E. Hicken, D. W. Zingg, Very high-order symmetric positive-interior
quadrature rules on triangles and tetrahedra, arXiv:2409.02027, 2024, whose smallest barycentric
coordinate is 3.6e-3.

To reproduce (about ten minutes):

```sh
python tools/search_symmetric_tet_rule.py --degree 10 --structure S1:1,S4:3,S6:1,S12:5 \
    orbit-search-n79-d10.txt
```
