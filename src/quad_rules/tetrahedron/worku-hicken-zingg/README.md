# Worku–Hicken–Zingg rules for the tetrahedron

Fully symmetric rules with positive weights and points inside the tetrahedron, from

Z. A. Worku, J. E. Hicken, D. W. Zingg, Very high-order symmetric positive-interior quadrature rules on
triangles and tetrahedra, arXiv:2409.02027, 2024, https://arxiv.org/abs/2409.02027

The paper gives rules up to degree 40; this directory holds those of degrees 1 to 21. `quad_rules.SymmetricTet`
selects the ones of degrees 11, 12, 14, 19, 20 and 21, which have fewer points than every other fully symmetric
rule here with positive weights and all barycentric coordinates above 1e-3 (100, 124, 181, 408, 469 and 540
points). The rules of degrees 9 and 16 have points closer to the boundary than that (smallest barycentric
coordinates 3.8e-4 and 5.8e-5); the others have as many points as the rules selected for their degree, or more.

The data are taken from SummationByParts.jl (https://github.com/OptimalDesignLab/SummationByParts.jl),
`pi_quadrature_data/high_precision/tet/expanded/tet_q<degree>_n<points>_ext.dat` at commit
258976a71a95fb33fd859ac3d582de278b2a86ac, which stores the points on the tetrahedron
(-1, -1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1) to 128 digits with weights summing to its volume 4/3.
The files here hold the same values in barycentric coordinates, converted in decimal arithmetic and rounded
to 40 significant digits: (l1, l2, l3) = ((x + 1) / 2, (y + 1) / 2, (z + 1) / 2), l0 = 1 - l1 - l2 - l3, and
the weights multiplied by 3/4. Each file names the orbits of its rule in the header.

## License

The SummationByParts.jl package is licensed under the MIT "Expat" License:

> Copyright (c) 2015: Jason Hicken.
>
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
> CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
> TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
> SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
