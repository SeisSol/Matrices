# Witherden–Vincent rules for the triangle

Fully symmetric rules with positive weights, exact up to degree 1 to 20 (there is no rule of degree 3), from

F. D. Witherden, P. E. Vincent, On the identification of symmetric quadrature rules for finite element methods,
Computers & Mathematics with Applications 69(10), 1232–1241, 2015,
https://doi.org/10.1016/j.camwa.2015.03.017

The data are taken from PyFR (https://github.com/PyFR/PyFR), files `pyfr/quadrules/tri/witherden-vincent-n*-d*-sp.txt`
as added in commit 7124cb7.
PyFR stores the points on the triangle (-1, -1), (1, -1), (-1, 1) with weights summing to 2.
The files here hold the same values in barycentric coordinates, converted in decimal arithmetic with 80 digits:
l1 = (x + 1) / 2, l2 = (y + 1) / 2, l0 = 1 - l1 - l2, and the weight w / 2.
Read back with `quad_rules.TabulatedTri`, they give the same doubles as the PyFR values.

The rule of degree 16 has three points on the edges of the triangle, at their midpoints.
PyFR gives the coordinate 0 of these points as -5.4e-109; it reads as 0 here.

## License

The data are distributed under the license of PyFR:

```
Copyright (c) 2012–2026 Imperial College London

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of PyFR nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
```
