# Tabulated quadrature rules for the tetrahedron

Each subdirectory holds one family of rules, one file per rule, named `<name>-n<points>-d<degree>.txt`,
where `<points>` is the number of points and `<degree>` the polynomial degree up to which the rule is exact.
The README in each subdirectory names the source of the data and its license.

## File format

Lines starting with `#` are comments.
Every other line holds the barycentric coordinates `l0 l1 l2 l3` of one point and its weight.
The point is `l0 * v0 + l1 * v1 + l2 * v2 + l3 * v3` for the vertices `v0, ..., v3` of the tetrahedron,
and the weights sum to 1, i.e. the integral over a tetrahedron `T` is approximated by `|T| * sum_i w_i f(x_i)`.

`quad_rules.TabulatedTet.TabulatedTet(<directory>)` reads a family and returns the rules
on the reference tetrahedron (-1, -1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1) used throughout `quad_rules`.

## Families

| Directory | Degrees | Fully symmetric | Source | License |
| --------- | ------- | --------------- | ------ | ------- |
| `witherden-vincent` | 1–10 | yes | Witherden, Vincent (2015), via PyFR | BSD 3-Clause (PyFR) |
| `witherden` | 10–20 (without 14) | yes | Witherden (2023), via PyFR | BSD 3-Clause (PyFR) |
