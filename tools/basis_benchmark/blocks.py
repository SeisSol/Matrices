"""
The blocks of a time step of SeisSol's ADER-DG scheme as yateto kernels, for one
basis of seissol_matrices (the Dubiner or the Bernstein basis) and one way to
keep the operators of the basis:

- dense: each operator on the bounding box of its non-zeros, aligned, for a
  GEMM generator;
- csc: the non-zeros in compressed columns, read from memory;
- immediate: the numbers written into the generated code
  (yateto.type.AddressingMode.IMMEDIATE), so that a one is no multiplication,
  a minus one a sign and a zero nothing at all.

The matrices that evaluate at points or integrate over them are dense in every
basis and stay dense.

With one simulation the basis functions lead the data and an operator is
applied from the left; fused simulations lead with the simulation, and the
operator, stored transposed, is applied from the right.

The kernels, for the degree N = order - 1, the degrees of freedom Q and the
time-integrated degrees of freedom I:

ck             the time derivatives dQ(k), k = 1 ... N, of degree N - k: the sum
               over the directions of a derivative times dQ(k - 1) times the
               star matrix of the direction (dQ(0) = Q);
taylor         I, and the value at the end of the step, which replaces Q (the
               strong form needs no volume kernel);
trace(f, o)    the trace of I on the face f in the canonical frame of the face,
               for the orientation o of the cell's face against it
               (dg_matrices.FACE_ORIENTATIONS);
lift(f, o)     the jump of the traces of both cells of a face, the flux of the
               face, and its lift into Q;
nodal(f, o)    the same with the flux at the points of a face quadrature
               (face_quadrature) and its moments;
drTaylor       dynamic rupture as SeisSol reads the fault: the Taylor
drEval(a, b)   expansion of the cell at a time point, evaluated at the points
drBack(a, b)   of the fault seen from side a with orientation b (V3mTo2n); the
               flux of the fault goes back through V3mTo2nTWDivM;
drTrace(f, o)  dynamic rupture through the trace: the traces of the time
drCombine      derivatives in the canonical frame, evaluated at the fault's
drFaceBack     points once per face and combined per time point; the flux goes
(f, o)         back through its moments on the face and the lift.

In the Dubiner basis every operator is a matrix. The Bernstein basis applies
the factors of bernstein_matrices: the derivative as a difference stencil
(its factor, the degree, moves into the Taylor coefficients, see
Operators.derivative_scale), sums of time derivatives of different degree by
Horner's scheme with the elevation, the trace as a gather and the lift
factorised.
"""

import math

import numpy as np
from yateto import Scalar, Tensor
from yateto.memory import CSCMemoryLayout, DenseMemoryLayout
from yateto.type import AddressingMode

from seissol_matrices import (
    bernstein_matrices,
    dg_matrices,
    dr_matrices,
    face_quadrature,
    quad_points,
)

QUANTITIES = 9
FAMILIES = ("dubiner", "bernstein")
STORAGES = ("dense", "csc", "immediate")
ORIENTATIONS = ("sorted", "all")


def dim3(degree):
    return bernstein_matrices.count(degree, 3)


def dim2(degree):
    return bernstein_matrices.count(degree, 2)


class WholeLayout:
    """
    The layout of an operator written into the code: dense over its whole
    shape, whatever its non-zeros. A GEMM widens its rows to the vector
    boundary as soon as the layout of an operand has an aligned stride, which a
    bounding box that starts at an unaligned row can have as well; over the
    whole shape, the widened rows lie inside the layout.
    """

    @staticmethod
    def fromSpp(spp, alignStride=False, alignOffset=0, alignmentArch=None):
        return DenseMemoryLayout(
            tuple(spp.shape), alignStride=alignStride, alignmentArch=alignmentArch
        )


def embed(block, shape):
    """The block in the leading rows and columns of a zero matrix."""
    matrix = np.zeros(shape)
    matrix[: block.shape[0], : block.shape[1]] = block
    return matrix


class Operators:
    """
    The constant operators of the blocks as numpy arrays, at the shape of the
    scheme's degree: an operator of a lower degree sits in the leading rows and
    columns, which is where the degrees of freedom of that degree are kept.

    `orientations` selects the orientations of the faces against their
    canonical frame that the kernels cover: "sorted", the one each face has
    when a cell orders its vertices by their global IDs, or "all" six.
    """

    def __init__(self, order, family, orientations="sorted"):
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family}")
        if orientations not in ORIENTATIONS:
            raise ValueError(f"unknown orientations {orientations}")
        self.order = order
        self.family = family
        self.degree = order - 1
        self.n3 = dim3(self.degree)
        self.n2 = dim2(self.degree)
        self.dg = dg_matrices.dg_generator(order, 3, family)
        if orientations == "sorted":
            self.combos = [
                (
                    f,
                    dg_matrices.FACE_ORIENTATIONS.index(
                        dg_matrices.sorted_vertex_ranks(f)
                    ),
                )
                for f in range(4)
            ]
        else:
            self.combos = [
                (f, o)
                for f in range(4)
                for o in range(len(dg_matrices.FACE_ORIENTATIONS))
            ]
        self.dr_combos = [(a, b) for a in range(4) for b in range(4)]
        self.face_rule = face_quadrature.FaceQuadratureGenerator(order, family)
        self.dr = dr_matrices.dr_generator(order, quad_points.stroud(order + 1), family)
        self._derivatives = {}
        self._canonical = {}

    @property
    def bernstein(self):
        return self.family == "bernstein"

    def derivative(self, k, direction):
        """The map from dQ(k - 1), of degree N - k + 1, to the part of dQ(k) of
        the direction."""
        m = self.degree - k + 1
        if self.bernstein:
            block = bernstein_matrices.derivative(m, 3, direction) / m
        else:
            if direction not in self._derivatives:
                self._derivatives[direction] = self.dg.kDivMT(direction)
            block = self._derivatives[direction][: dim3(m - 1), : dim3(m)]
        return embed(block, (self.n3, self.n3))

    def derivative_scale(self, k):
        """dQ(k) over what ck computes for it: the product of the degrees that the
        difference stencils of the Bernstein basis leave out."""
        if not self.bernstein:
            return 1.0
        return float(math.perm(self.degree, k))

    def elevation(self, m):
        """The elevation from degree m - 1 to m in the cell (Bernstein basis)."""
        return embed(bernstein_matrices.elevation(m, 3), (self.n3, self.n3))

    def canonical_rT(self, face, orientation):
        key = (face, orientation)
        if key not in self._canonical:
            ranks = dg_matrices.FACE_ORIENTATIONS[orientation]
            self._canonical[key] = self.dg.canonical_rT(face, ranks)
        return self._canonical[key]

    def trace(self, face, orientation, degree=None):
        """The trace of a polynomial of the degree (by default N) in the canonical
        frame of the face."""
        degree = self.degree if degree is None else degree
        if self.bernstein:
            ranks = dg_matrices.FACE_ORIENTATIONS[orientation]
            block = bernstein_matrices.trace(degree, face, ranks)
        else:
            block = self.canonical_rT(face, orientation)[: dim2(degree), : dim3(degree)]
        return embed(block, (self.n2, self.n3))

    def face_mass(self):
        return self.dg.face_generator.mass_matrix()

    def lift(self, face, orientation, moments=False):
        """M^-1 rT^T M_face in the canonical frame (Dubiner basis): face
        coefficients into the cell; with `moments`, M^-1 rT^T for the integrals
        of a flux against the face basis instead."""
        rDivM = np.linalg.solve(
            self.dg.mass_matrix(), self.canonical_rT(face, orientation).T
        )
        return rDivM if moments else rDivM @ self.face_mass()

    def lift_face(self):
        """L_0 of the factorised lift (Bernstein basis)."""
        return bernstein_matrices.lift_face(self.degree)

    def lift_elevation(self, level):
        """(E^(N - level + 1)_(N - level))^T: one level of the lift further."""
        m = self.degree - level + 1
        return embed(bernstein_matrices.elevation(m, 2).T, (self.n2, self.n2))

    def lift_scatter(self, face, orientation, level):
        """l_level S_level: a level of the lift into the cell."""
        ranks = dg_matrices.FACE_ORIENTATIONS[orientation]
        scale = float(bernstein_matrices.lift_coefficients(self.degree)[level])
        indices = bernstein_matrices.lift_scatter_indices(
            self.degree, face, level, ranks
        )
        matrix = np.zeros((self.n3, self.n2))
        matrix[indices, np.arange(len(indices))] = scale
        return matrix

    def from_moments(self, moments):
        """The input of the factorised lift, L_0 M_face^-1, ahead of a matrix of
        moments (Bernstein basis)."""
        return self.lift_face() @ np.linalg.solve(self.face_mass(), moments)

    def nodal_values(self):
        """The face basis at the points of the face quadrature (points x n2)."""
        return self.face_rule.generate_V2mToFaceQuad()

    def nodal_moments(self):
        """The integrals against the face basis from the points (n2 x points)."""
        return self.face_rule.generate_VFaceQuadTo2m()

    def dr_points(self):
        return self.dr.quadpoints().shape[0]

    def dr_volume(self, side, orientation):
        return self.dr.V3mTo2n(side, orientation)

    def dr_volume_back(self, side, orientation):
        return self.dr.V3mTo2nTWDivM(side, orientation)

    def dr_face_values(self, degree):
        """The face basis of a degree at the fault's points, which are given in the
        canonical frame of the face."""
        return embed(self.dr.V2mTo2Quad(degree), (self.dr_points(), self.n2))

    def dr_moments(self):
        weights = self.dr.quadweights()
        return self.dr.V2mTo2Quad().T * weights[None, :]

    def time_nodes(self):
        """The time points of the fault in a step of length one: Gauss-Legendre,
        as many as the order."""
        nodes, _ = np.polynomial.legendre.leggauss(self.order)
        return 0.5 * (nodes + 1.0)

    def taylor_coefficients(self, dt, scaled):
        """
        The coefficients of dQ(k), k = 0 ... N, in the time integral over a step
        dt ("integral"), the value at its end ("end") and the values at the time
        points of the fault ("points", one row per point). `scaled` gives them
        for the derivatives as ck computes them, see derivative_scale.
        """
        N = self.degree
        scale = np.array(
            [self.derivative_scale(k) if scaled else 1.0 for k in range(N + 1)]
        )
        k = np.arange(N + 1)
        factorial = np.array([math.factorial(i) for i in k], dtype=float)
        times = dt * self.time_nodes()
        return {
            "integral": scale * dt ** (k + 1) / (factorial * (k + 1)),
            "end": scale * dt**k / factorial,
            "points": scale[None, :] * times[:, None] ** k[None, :] / factorial,
        }


class Kernels:
    """The yateto kernels of the blocks for the operators of a basis, the storage
    of the operators and the number of fused simulations."""

    def __init__(self, operators, storage, simulations=1):
        if storage not in STORAGES:
            raise ValueError(f"unknown storage {storage}")
        self.ops = operators
        self.storage = storage
        self.simulations = simulations
        self.runtime = {}
        self.operators = {}
        self.costs = {}
        self.temporaries = {}
        N = operators.degree
        n3, n2 = operators.n3, operators.n2
        P = operators.dr_points()

        self.Q = self._data("Q", n3)
        self.I = self._data("I", n3)
        self.dQ = {k: self._data(f"dQ({k})", n3, dim3(N - k)) for k in range(1, N + 1)}
        self.star = {
            d: self._runtime(Tensor(f"star({d})", (QUANTITIES, QUANTITIES)))
            for d in range(3)
        }
        self.AminusT = self._runtime(Tensor("AminusT", (QUANTITIES, QUANTITIES)))
        self.T = self._data("T", n2)
        self.Tn = self._data("Tn", n2)
        self.q = self._data("q", n3)
        self.qp = self._data("qp", P)
        self.F = self._data("F", P)
        self.Pk = {k: self._data(f"Pk({k})", P) for k in range(N + 1)}
        self.cI = {k: Scalar(f"cI({k})") for k in range(N + 1)}
        self.cQ = {k: Scalar(f"cQ({k})") for k in range(N + 1)}
        self.cT = {k: Scalar(f"cT({k})") for k in range(N + 1)}

    # tensors

    def _runtime(self, tensor):
        self.runtime[tensor.name()] = tensor
        return tensor

    def _data(self, name, rows, valid=None, temporary=False):
        """Data of `rows` basis functions (or points) by the quantities, of which
        the leading `valid` rows are used."""
        if self.simulations == 1:
            shape = (rows, QUANTITIES)
        else:
            shape = (self.simulations, rows, QUANTITIES)
        if temporary and name in self.temporaries:
            return self.temporaries[name]
        spp = None
        if valid is not None and valid < rows:
            spp = np.zeros(shape, dtype=bool)
            spp[..., :valid, :] = True
        tensor = Tensor(name, shape, spp=spp, alignStride=True, temporary=temporary)
        if temporary:
            self.temporaries[name] = tensor
            return tensor
        return self._runtime(tensor)

    def _operator(self, name, matrix, storage=None):
        """A constant operator, created once per name, kept as `storage` (by
        default the storage under test) says; transposed for fused simulations.
        What is left of a zero after computing the matrix in floating point is
        dropped."""
        if name in self.operators:
            return self.operators[name]
        storage = self.storage if storage is None else storage
        stored = matrix if self.simulations == 1 else matrix.T
        largest = np.abs(stored).max()
        stored = np.where(np.abs(stored) > 1e-13 * largest, stored, 0.0)
        spp = {
            tuple(int(i) for i in entry): repr(float(stored[tuple(entry)]))
            for entry in np.argwhere(stored != 0.0)
        }
        if storage == "dense":
            tensor = Tensor(name, stored.shape, spp=spp, alignStride=True)
        elif storage == "csc":
            tensor = Tensor(
                name, stored.shape, spp=spp, memoryLayoutClass=CSCMemoryLayout
            )
        else:
            tensor = Tensor(
                name,
                stored.shape,
                spp=spp,
                memoryLayoutClass=WholeLayout,
                addressing=AddressingMode.IMMEDIATE,
            )
        # entries a product with the operator goes through
        self.costs[name] = len(spp) if storage != "dense" else self._box(stored)
        self.operators[name] = tensor
        return tensor

    @staticmethod
    def _box(matrix):
        rows, columns = np.nonzero(matrix)
        return int((rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1))

    def _ix(self, row, column="p"):
        return row + column if self.simulations == 1 else "s" + row + column

    def _apply(self, operator, data, out, inner, column="p"):
        """The operator times the data; `out` indexes the rows of the operator,
        `inner` its columns and the rows of the data."""
        if self.simulations == 1:
            return operator[out + inner] * data[inner + column]
        return data["s" + inner + column] * operator[inner + out]

    # building blocks of the kernels

    def _sum(self, coefficients, first, k_range):
        """first + the sum over k of coefficients[k] dQ(k), for a hierarchical
        basis, in which dQ(k) is the leading part of degree N."""
        ix = self._ix("a")
        expression = first
        for k in k_range:
            expression = expression + coefficients[k] * self.dQ[k][ix]
        return expression

    def _horner(self, name, coefficients):
        """J(1) = sum over k >= 1 of coefficients[k] (dQ(k) elevated to degree
        N - 1), from J(N) = coefficients[N] dQ(N) by
        J(k) = coefficients[k] dQ(k) + E(N - k) J(k + 1). Returns the statements
        and J(1)."""
        N = self.ops.degree
        ix = self._ix("a")
        statements = []
        previous = None
        for k in range(N, 0, -1):
            J = self._data(f"{name}{k}", self.ops.n3, dim3(N - k), temporary=True)
            expression = coefficients[k] * self.dQ[k][ix]
            if previous is not None:
                expression = expression + self._apply(
                    self._elevation(N - k), previous, "a", "b"
                )
            statements.append(J[ix] <= expression)
            previous = J
        return statements, previous

    def _elevation(self, m):
        return self._operator(f"E({m})", self.ops.elevation(m))

    def _taylor_into(self, target, coefficients, name, accumulate):
        """target = coefficients[0] Q + sum over k >= 1 of coefficients[k] dQ(k),
        or target += the sum over k >= 1 (`accumulate`)."""
        N = self.ops.degree
        ix = self._ix("a")
        first = self.Q[ix] if accumulate else coefficients[0] * self.Q[ix]
        if not self.ops.bernstein:
            return [target[ix] <= self._sum(coefficients, first, range(1, N + 1))]
        statements, J1 = self._horner(name, coefficients)
        statements.append(
            target[ix] <= first + self._apply(self._elevation(N), J1, "a", "b")
        )
        return statements

    def _chain(self, face, orientation, X0, tag):
        """Q += sum over the levels j of l_j S_j (E^N_(N-j))^T X0 (Bernstein)."""
        N = self.ops.degree
        ix, ik = self._ix("a"), self._ix("k")
        statements = []
        current = X0
        for level in range(N + 1):
            if level > 0:
                X = self._data(
                    f"{tag}X{level}", self.ops.n2, dim2(N - level), temporary=True
                )
                elevation = self._operator(
                    f"LE({level})", self.ops.lift_elevation(level)
                )
                statements.append(X[ix] <= self._apply(elevation, current, "a", "b"))
                current = X
            scatter = self._operator(
                f"S({face},{orientation},{level})",
                self.ops.lift_scatter(face, orientation, level),
            )
            statements.append(
                self.Q[ik] <= self.Q[ik] + self._apply(scatter, current, "k", "a")
            )
        return statements

    def _lift(self, face, orientation, F, tag):
        """Q += the lift of F, given by its face coefficients."""
        ix, ik = self._ix("a"), self._ix("k")
        if not self.ops.bernstein:
            lift = self._operator(
                f"L({face},{orientation})", self.ops.lift(face, orientation)
            )
            return [self.Q[ik] <= self.Q[ik] + self._apply(lift, F, "k", "a")]
        X0 = self._data(f"{tag}X0", self.ops.n2, temporary=True)
        L0 = self._operator("L0", self.ops.lift_face())
        return [X0[ix] <= self._apply(L0, F, "a", "b")] + self._chain(
            face, orientation, X0, tag
        )

    def _lift_points(self, face, orientation, Fp, moments, tag):
        """Q += the lift of a flux given at points, from its moments
        (moments: face basis by points). The Dubiner basis takes the moments and
        the lift as one matrix if that is cheaper."""
        ix, ik = self._ix("a"), self._ix("k")
        if self.ops.bernstein:
            X0 = self._data(f"{tag}X0", self.ops.n2, temporary=True)
            W = self._operator(f"{tag}W", self.ops.from_moments(moments), "dense")
            return [X0[ix] <= self._apply(W, Fp, "a", "i")] + self._chain(
                face, orientation, X0, tag
            )
        lift = self.ops.lift(face, orientation, moments=True)
        W = self._operator(f"{tag}W", moments, "dense")
        separate = self._operator(f"{tag}L({face},{orientation})", lift)
        folded = self._operator(
            f"{tag}LW({face},{orientation})", lift @ moments, "dense"
        )
        cost = lambda tensor: self.costs[tensor.name()]
        if cost(folded) <= cost(W) + cost(separate):
            return [self.Q[ik] <= self.Q[ik] + self._apply(folded, Fp, "k", "i")]
        X = self._data(f"{tag}X", self.ops.n2, temporary=True)
        return [
            X[ix] <= self._apply(W, Fp, "a", "i"),
            self.Q[ik] <= self.Q[ik] + self._apply(separate, X, "k", "a"),
        ]

    # kernels

    def add_to(self, generator):
        self._add_predictor(generator)
        self._add_faces(generator)
        self._add_dr_volume(generator)
        self._add_dr_face(generator)

    def _add_predictor(self, generator):
        N = self.ops.degree
        ix = self._ix("a")
        statements = []
        previous = self.Q
        for k in range(1, N + 1):
            terms = [
                self._apply(
                    self._operator(f"kD({k},{d})", self.ops.derivative(k, d)),
                    previous,
                    "a",
                    "b",
                    "q",
                )
                * self.star[d]["qp"]
                for d in range(3)
            ]
            statements.append(self.dQ[k][ix] <= terms[0] + terms[1] + terms[2])
            previous = self.dQ[k]
        generator.add("ck", statements)
        generator.add(
            "taylor",
            self._taylor_into(self.I, self.cI, "tI", accumulate=False)
            + self._taylor_into(self.Q, self.cQ, "tQ", accumulate=True),
        )

    def _add_faces(self, generator):
        ops = self.ops
        ix, ii = self._ix("a"), self._ix("i")
        traces = {
            (f, o): self._operator(f"tr({f},{o},0)", ops.trace(f, o))
            for f, o in ops.combos
        }
        generator.addFamily(
            "trace",
            ops.combos,
            lambda f, o: self.T[ix] <= self._apply(traces[(f, o)], self.I, "a", "l"),
        )

        J = self._data("fJ", ops.n2, temporary=True)
        F = self._data("fF", ops.n2, temporary=True)
        jump = [
            J[ix] <= self.Tn[ix] - self.T[ix],
            F[ix] <= J[self._ix("a", "q")] * self.AminusT["qp"],
        ]
        generator.addFamily(
            "lift", ops.combos, lambda f, o: jump + self._lift(f, o, F, "f")
        )

        points = ops.nodal_values().shape[0]
        V = self._operator("nV", ops.nodal_values(), "dense")
        Jp = self._data("nJ", points, temporary=True)
        Fp = self._data("nF", points, temporary=True)
        nodal_jump = [
            J[ix] <= self.Tn[ix] - self.T[ix],
            Jp[ii] <= self._apply(V, J, "i", "a"),
            Fp[ii] <= Jp[self._ix("i", "q")] * self.AminusT["qp"],
        ]
        moments = ops.nodal_moments()
        generator.addFamily(
            "nodal",
            ops.combos,
            lambda f, o: nodal_jump + self._lift_points(f, o, Fp, moments, "n"),
        )

    def _add_dr_volume(self, generator):
        ops = self.ops
        ii, ik = self._ix("i"), self._ix("k")
        generator.add(
            "drTaylor", self._taylor_into(self.q, self.cT, "tT", accumulate=False)
        )
        volume = {
            (a, b): self._operator(f"V3({a},{b})", ops.dr_volume(a, b), "dense")
            for a, b in ops.dr_combos
        }
        generator.addFamily(
            "drEval",
            ops.dr_combos,
            lambda a, b: self.qp[ii] <= self._apply(volume[(a, b)], self.q, "i", "a"),
        )
        back = {
            (a, b): self._operator(f"V3TW({a},{b})", ops.dr_volume_back(a, b), "dense")
            for a, b in ops.dr_combos
        }
        generator.addFamily(
            "drBack",
            ops.dr_combos,
            lambda a, b: self.Q[ik]
            <= self.Q[ik] + self._apply(back[(a, b)], self.F, "k", "i"),
        )

    def _add_dr_face(self, generator):
        ops = self.ops
        N = ops.degree
        ix, ii = self._ix("a"), self._ix("i")
        values = {
            k: self._operator(f"V2({k})", ops.dr_face_values(N - k), "dense")
            for k in range(N + 1)
        }

        def traces(f, o):
            statements = []
            for k in range(N + 1):
                source = self.Q if k == 0 else self.dQ[k]
                trace = self._operator(f"tr({f},{o},{k})", ops.trace(f, o, N - k))
                Tk = self._data(f"dT{k}", ops.n2, dim2(N - k), temporary=True)
                statements.append(Tk[ix] <= self._apply(trace, source, "a", "l"))
                statements.append(
                    self.Pk[k][ii] <= self._apply(values[k], Tk, "i", "a")
                )
            return statements

        generator.addFamily("drTrace", ops.combos, traces)
        combination = self.cT[0] * self.Pk[0][ii]
        for k in range(1, N + 1):
            combination = combination + self.cT[k] * self.Pk[k][ii]
        generator.add("drCombine", self.qp[ii] <= combination)
        moments = ops.dr_moments()
        generator.addFamily(
            "drFaceBack",
            ops.combos,
            lambda f, o: self._lift_points(f, o, self.F, moments, "d"),
        )
