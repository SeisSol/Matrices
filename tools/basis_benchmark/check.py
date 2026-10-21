"""
What the kernels of blocks.py have to compute on the input of `driver --check`,
from the dense operators of seissol_matrices, and the comparison with what they
wrote.
"""

import numpy as np

from seissol_matrices import basis_functions, bernstein_matrices, dg_matrices

from blocks import QUANTITIES, dim3


def value(h, shape):
    """The input the driver writes for the number h: one sine per entry
    (simulation, row, column)."""
    s, i, j = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return np.sin(0.1 + 0.37 * h + 0.417 * s + 0.731 * i + 1.913 * j)


def read(path):
    """The records of a --check file: name -> array (simulations, rows, columns)."""
    records = {}
    with open(path, "rb") as f:
        while True:
            header = f.readline()
            if not header:
                break
            name, *shape = header.decode().split()
            shape = tuple(int(n) for n in shape)
            data = np.frombuffer(f.read(8 * int(np.prod(shape))), dtype=np.float64)
            records[name] = data.reshape(shape)
    return records


class Reference:
    """The expected results for the operators of a basis, `simulations` fused,
    and the Taylor coefficients the driver used (unscaled)."""

    def __init__(self, ops, simulations, coefficients):
        self.ops = ops
        self.S = simulations
        self.c = coefficients
        self.dg = ops.dg
        N = ops.degree
        self.Q1 = value(1, (self.S, ops.n3, QUANTITIES))
        self.Q2 = value(2, (self.S, ops.n3, QUANTITIES))
        self.star = [value(10 + d, (1, QUANTITIES, QUANTITIES))[0] for d in range(3)]
        self.A = value(20, (1, QUANTITIES, QUANTITIES))[0]
        self.T = value(30, (self.S, ops.n2, QUANTITIES))
        self.Tn = value(31, (self.S, ops.n2, QUANTITIES))
        self.F = value(40, (self.S, ops.dr_points(), QUANTITIES))

        # the time derivatives, of degree N - k in the leading rows
        self.dQ = [self.Q1]
        for k in range(1, N + 1):
            m = N - k + 1
            previous = self.dQ[-1][:, : dim3(m)]
            current = np.zeros_like(self.Q1)
            for d in range(3):
                if ops.bernstein:
                    D = bernstein_matrices.derivative(m, 3, d)
                else:
                    D = self.dg.kDivMT(d)[: dim3(m - 1), : dim3(m)]
                current[:, : dim3(m - 1)] += np.einsum(
                    "ab,sbq,qp->sap", D, previous, self.star[d]
                )
            self.dQ.append(current)

    def elevated(self, k):
        """dQ(k) as a polynomial of degree N."""
        N = self.ops.degree
        if not self.ops.bernstein:
            return self.dQ[k]
        E = np.eye(dim3(N - k))
        for m in range(N - k + 1, N + 1):
            E = bernstein_matrices.elevation(m, 3) @ E
        return np.einsum("ab,sbp->sap", E, self.dQ[k][:, : dim3(N - k)])

    def taylor(self, coefficients):
        return sum(c * self.elevated(k) for k, c in enumerate(coefficients))

    def ranks(self, orientation):
        return dg_matrices.FACE_ORIENTATIONS[orientation]

    def lift(self, face, orientation):
        """M^-1 rT^T for the moments of a flux on a face in its canonical frame."""
        return np.linalg.solve(
            self.dg.mass_matrix(),
            self.dg.canonical_rT(face, self.ranks(orientation)).T,
        )

    def expected(self, records):
        """name -> expected array, for every record of a --check file."""
        ops = self.ops
        N = ops.degree
        Mf = ops.face_mass()
        result = {}
        for k in range(1, N + 1):
            result[f"dQ({k})"] = self.dQ[k] / ops.derivative_scale(k)
        result["I"] = self.taylor(self.c["integral"])
        result["Qend"] = self.taylor(self.c["end"])
        V = ops.nodal_values()
        W = ops.nodal_moments()
        jump = self.Tn - self.T
        for f, o in ops.combos:
            rT = self.dg.canonical_rT(f, self.ranks(o))
            result[f"trace({f},{o})"] = np.einsum("al,slp->sap", rT, records["I"])
            lift = self.lift(f, o)
            flux = np.einsum("saq,qp->sap", jump, self.A)
            result[f"lift({f},{o})"] = self.Q2 + np.einsum(
                "ka,ab,sbp->skp", lift, Mf, flux
            )
            points = np.einsum("ia,saq,qp->sip", V, jump, self.A)
            result[f"nodal({f},{o})"] = self.Q2 + np.einsum(
                "ka,ai,sip->skp", lift, W, points
            )

        result["q"] = self.taylor(self.c["points"][0])
        for a, b in ops.dr_combos:
            result[f"drEval({a},{b})"] = np.einsum(
                "ia,sap->sip", ops.dr_volume(a, b), records["q"]
            )
            result[f"drBack({a},{b})"] = self.Q2 + np.einsum(
                "ki,sip->skp", ops.dr_volume_back(a, b), self.F
            )
        rule = ops.dr.quadrule
        weights = rule.weights()
        values = ops.dr.V2mTo2Quad()
        for f, o in ops.combos:
            local = dg_matrices.canonical_to_face_parametrisation(
                rule.points().T, self.ranks(o)
            )
            points = dg_matrices.volume_to_face_parametrisation(local, f)
            fault = 0
            for k, c in enumerate(self.c["points"][0]):
                basis = basis_functions.generator(ops.family, 3, N - k + 1)
                fault = fault + c * np.einsum(
                    "ia,sap->sip",
                    basis.eval_all(points),
                    self.dQ[k][:, : dim3(N - k)],
                )
            result[f"drFace({f},{o})"] = fault
            result[f"drFaceBack({f},{o})"] = self.Q2 + np.einsum(
                "ka,ia,i,sip->skp", self.lift(f, o), values, weights, self.F
            )
        return result


def compare(records, expected, tolerance=1e-9):
    """The records that differ from what is expected by more than the tolerance,
    relative to the largest expected entry where that exceeds one, and the
    largest such difference."""
    failures = []
    worst = 0.0
    missing = sorted(set(expected) - set(records))
    for name, data in records.items():
        if name not in expected:
            failures.append((name, "unexpected"))
            continue
        target = expected[name]
        if target.shape != data.shape:
            failures.append((name, f"shape {data.shape} instead of {target.shape}"))
            continue
        error = np.abs(data - target).max() / max(1.0, np.abs(target).max())
        worst = max(worst, error)
        if not error <= tolerance:
            failures.append((name, f"relative error {error:.2e}"))
    failures += [(name, "missing") for name in missing]
    return failures, worst
