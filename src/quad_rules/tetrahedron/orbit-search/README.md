# Symmetric rules for the tetrahedron found by a search over orbit structures

Fully symmetric rules with positive weights and all barycentric coordinates above 1e-3, with no more points
than the other rules of this kind and degree in `quad_rules` and further inside than those with as many:

| Degree | Points | Orbits | Smallest barycentric coordinate | Other fully symmetric rules with positive weights |
| ------ | ------ | ------ | ------------------------------- | ------------------------------------------------- |
| 9 | 59 | S1x1 S4x4 S6x1 S12x3 | 2.6e-3 | Witherden–Vincent, Jaśkowiec–Sukumar: 59 points each, with points on the boundary (smallest barycentric coordinate 6.2e-10, 3.4e-9); Worku–Hicken–Zingg: 61 points (3.8e-4) |
| 10 | 79 | S1x1 S4x3 S6x1 S12x5 | 1.06e-2 | Witherden–Vincent: 81 points (smallest barycentric coordinate 6.1e-3); Worku–Hicken–Zingg: 79 points (3.6e-3) |

They were computed for SeisSol with `tools/search_symmetric_tet_rule.py`: for a given number of orbits of
each kind, it solves the moment equations from many random starts, keeping the weights positive and the
points inside. The structure of degree 10 has one unknown more than there are equations, so its solutions
form a one-parameter family; the rule here is the member whose smallest barycentric coordinate is largest.
The structure of degree 9 has two unknowns more than its 18 equations, and the rule here is where the tool
climbs to within that two-parameter family. The rules of Witherden–Vincent and of Jaśkowiec–Sukumar with
59 points have the same orbits and solve the same equations, with an orbit at the vertices and one on the
edges respectively. A search over the 15 structures with 51 to 58 points whose ranks can reach the 18
equations of degree 9, 6000 random starts each, found no rule with all barycentric coordinates above 1e-3.
The results are refined to 40 digits; the moments of all monomials up to the degree are exact to a relative
error below 1e-40. Each file lists its orbits in the header.

A rule of degree 10 with 79 points has also been published by G. Chuluunbaatar, O. Chuluunbaatar,
A. A. Gusev, S. I. Vinitsky, PI-type fully symmetric quadrature rules on the 3-, ..., 6-simplexes,
Computers & Mathematics with Applications 124, 89–97, 2022, and one with the same orbits, another member of
the family, by Z. A. Worku, J. E. Hicken, D. W. Zingg, Very high-order symmetric positive-interior
quadrature rules on triangles and tetrahedra, arXiv:2409.02027, 2024, whose smallest barycentric
coordinate is 3.6e-3. For degree 9, Chuluunbaatar et al. list a rule with 59 points as well, and Worku,
Hicken and Zingg one with 61 points, whose smallest barycentric coordinate is 3.8e-4.

To reproduce (about ten and about four minutes):

```sh
python tools/search_symmetric_tet_rule.py --degree 10 --structure S1:1,S4:3,S6:1,S12:5 \
    orbit-search-n79-d10.txt
python tools/search_symmetric_tet_rule.py --degree 9 --structure S1:1,S4:4,S6:1,S12:3 \
    orbit-search-n59-d9.txt
```
