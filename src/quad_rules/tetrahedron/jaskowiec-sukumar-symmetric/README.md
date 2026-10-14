# Symmetric Jaśkowiec–Sukumar rules for the tetrahedron

Fully symmetric rules with positive weights and points inside the tetrahedron, exact up to degree 1 to 20, from

J. Jaśkowiec, N. Sukumar, High-order symmetric cubature rules for tetrahedra and pyramids,
International Journal for Numerical Methods in Engineering 122(1), 148–171, 2021,
https://doi.org/10.1002/nme.6528

These are not the rules of `quad_rules.JaskowiecSukumar`, which come from the earlier, non-symmetric set of
J. Jaśkowiec, N. Sukumar, High-order cubature rules for tetrahedra,
International Journal for Numerical Methods in Engineering 121(11), 2418–2436, 2020,
https://doi.org/10.1002/nme.6313

The data are taken from modepy 2026.1 (https://github.com/inducer/modepy),
`modepy/quadrature/jaskowiec_sukumar_tet_data.py`, which extracted them from the supplemental material of the paper.
modepy stores the points on the tetrahedron (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1) with weights summing to 1;
the files here hold the same values in barycentric coordinates, converted exactly in decimal arithmetic:
(l1, l2, l3) = (x, y, z), l0 = 1 - x - y - z.
The values have 16 significant digits, so the rules are exact up to relative errors of about 1e-14.

The rule of degree 9 has twelve points within 3.5e-9 (in barycentric coordinates) of the faces,
and the rule of degree 13 has six points within 2e-7 of the edge midpoints.

## License

The data are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/.
Copyright (C) 2020 Jan Jaśkowiec, N. Sukumar.
