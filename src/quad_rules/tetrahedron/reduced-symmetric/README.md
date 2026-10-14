# Reduced symmetric rules for the tetrahedron

Fully symmetric rules with positive weights and all barycentric coordinates above 1e-3,
with fewer points than the other rules of this kind and degree in `quad_rules`:

| Degree | Points | Start rule | Other fully symmetric rules with positive weights |
| ------ | ------ | ---------- | ------------------------------------------------- |
| 12 | 128 | `jaskowiec-sukumar-n172-d13.txt` | Jaśkowiec–Sukumar: 168 points (smallest barycentric coordinate 5.8e-4); Witherden: 123 points, four of them at the face centroids |
| 14 | 188 | `jaskowiec-sukumar-n204-d14.txt` | Jaśkowiec–Sukumar: 204 points |

They were computed for SeisSol with `tools/reduce_symmetric_tet_rule.py`, starting from the symmetric
Jaśkowiec–Sukumar rules in `../jaskowiec-sukumar-symmetric`: the tool removes orbits of points or replaces
them by orbits with fewer points and solves the moment equations again, keeping the weights positive and the
points inside, until no such move succeeds. The result is refined to 40 digits; the moments of all monomials
up to the degree are exact to a relative error below 1e-40. Each file lists its orbits in the header.

To reproduce, e.g. the rule of degree 12 (about ten minutes):

```sh
python tools/reduce_symmetric_tet_rule.py --degree 12 \
    src/quad_rules/tetrahedron/jaskowiec-sukumar-symmetric/jaskowiec-sukumar-n172-d13.txt \
    reduced-symmetric-n128-d12.txt
```

## License

The rules are adapted from the symmetric rules of
J. Jaśkowiec, N. Sukumar, High-order symmetric cubature rules for tetrahedra and pyramids,
International Journal for Numerical Methods in Engineering 122(1), 148–171, 2021,
https://doi.org/10.1002/nme.6528,
licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), Copyright (C) 2020 Jan Jaśkowiec, N. Sukumar:
orbits were removed or merged and all weights and positions computed anew.
