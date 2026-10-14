# Tabulated quadrature rules for the triangle

Each subdirectory holds one family of rules, one file per rule, named `<name>-n<points>-d<degree>.txt`,
where `<points>` is the number of points and `<degree>` the polynomial degree up to which the rule is exact.
The README in each subdirectory names the source of the data and its license.

## File format

Lines starting with `#` are comments.
Every other line holds the barycentric coordinates `l0 l1 l2` of one point and its weight.
The point is `l0 * v0 + l1 * v1 + l2 * v2` for the vertices `v0, v1, v2` of the triangle,
and the weights sum to 1, i.e. the integral over a triangle `T` is approximated by `|T| * sum_i w_i f(x_i)`.

`quad_rules.TabulatedTri.TabulatedTri(<directory>)` reads a family and returns the rules
on the reference triangle (-1, -1), (1, -1), (-1, 1) used throughout `quad_rules`.
The values are converted in decimal arithmetic and rounded to double once.

## Families

| Directory | Degrees | Fully symmetric | Source | License |
| --------- | ------- | --------------- | ------ | ------- |
| `witherden-vincent` | 1–20 (without 3) | yes | Witherden, Vincent (2015), via PyFR | BSD 3-Clause (PyFR) |
| `xiao-gimbutas` | 1–30 | yes | Xiao, Gimbutas (2010), via quadraturerules.org | CC BY 4.0 |

All rules have positive weights.
All points lie inside the triangle, except for three points of the Witherden–Vincent rule of degree 16,
which lie on its edges.
