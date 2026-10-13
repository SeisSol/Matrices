# Witherden's rules for the tetrahedron

Fully symmetric rules with positive weights, exact up to degree 10 to 20, computed by F. D. Witherden
with the method of

F. D. Witherden, P. E. Vincent, On the identification of symmetric quadrature rules for finite element methods,
Computers & Mathematics with Applications 69(10), 1232–1241, 2015,
https://doi.org/10.1016/j.camwa.2015.03.017

The data are taken from PyFR (https://github.com/PyFR/PyFR), files `pyfr/quadrules/tet/witherden-n*-d*-sp.txt`
as added in commit dc8d836 (2023).
PyFR stores the points on the tetrahedron (-1, -1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
with weights summing to 4/3.
The files here hold the same values in barycentric coordinates, converted exactly in decimal arithmetic:
l1 = (x + 1) / 2, l2 = (y + 1) / 2, l3 = (z + 1) / 2, l0 = 1 - l1 - l2 - l3, and the weight 3/4 w.

Some of the rules have points on the boundary or very close to it:

* degree 10: six points within 1.5e-9 (in barycentric coordinates) of the edge midpoints,
* degree 11: six points within 1.3e-6 of the edge midpoints and twelve points on the faces,
* degree 12: four points at the face centroids,
* degree 19: twelve points on the edges,
* degree 20: twelve points on the faces.

The rule of degree 14 in PyFR (`witherden-n172-d14-sp.txt`, 172 points) is not included:
eight of its weights are negative (down to -2.8 for weights summing to 1, with a sum of absolute values of 23.8).

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
