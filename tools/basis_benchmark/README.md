# Basis benchmark

Times the blocks of a time step of SeisSol's ADER-DG scheme as yateto generates
them, in the Dubiner and in the Bernstein basis of `seissol_matrices`, outside
of SeisSol.

## Blocks

The kernels are defined in `blocks.py`; per cell unless noted:

| block | kernels | what it does |
|---|---|---|
| `ck` | `ck` | the time derivatives dQ(k) of the Cauchy-Kovalevskaya procedure, each of degree N - k |
| `taylor` | `taylor` | the time integral I and the value at the end of the step (strong form, no volume kernel) |
| `trace` | `trace(f, o)` | the traces of I on the four faces, in the canonical frame of each face |
| `lift` | `lift(f, o)` | per face: jump of the traces of both cells, flux, lift into Q |
| `nodal` | `nodal(f, o)` | the same with the flux at the points of a face quadrature |
| `dr-volume` | `drTaylor`, `drEval(a, b)`, `drBack(a, b)` | per fault face: dynamic rupture as SeisSol reads the fault today, per time point the Taylor expansion of both cells evaluated at the fault points (`V3mTo2n`), the flux back through `V3mTo2nTWDivM` |
| `dr-face` | `drTrace(f, o)`, `drCombine`, `drFaceBack(f, o)` | per fault face: dynamic rupture through the trace, the traces of the time derivatives evaluated at the fault points once, combined per time point, the flux back through its moments on the face and the lift |

In the Bernstein basis the derivative is a difference stencil with entries ±1
(the factor of the degree moves into the Taylor coefficients), sums of time
derivatives of different degrees go through the elevation (Horner's scheme),
the trace is a gather and the lift is factorised (`bernstein_matrices.lift`).

## Variants

- `--families`: `dubiner`, `bernstein`
- `--storages`, how the constant operators are kept: `dense` (bounding box of
  the non-zeros, aligned, multiplied by PSpaMM), `csc` (non-zeros read from
  memory) or `immediate` (the numbers in the generated code, so that a one is no
  multiplication and a zero no instruction)
- `--simulations`: 1, or fused simulations (8), which lead the data
- `--archs`: `dhsw` (AVX2), `dskx` (AVX-512)
- `--orientations`: `sorted`, the orientation each face has when the cells order
  their vertices by global ID (one per face), or `all` six
- `--orders`

## Running

    python3 tools/basis_benchmark/benchmark.py --yateto /path/to/yateto --work /tmp/bench

yateto has to come with its `include` directory, `pspamm-generator` has to be
on the `PATH` (or given with `--pspamm`), and the kernels are compiled with
`g++` (`--cxx`, `--cxxflags`). Builds run in parallel (`--jobs`), timings one
after another, on one core.

Every build is first run once on known input and compared with what numpy
computes from the dense matrices of `seissol_matrices` (`check.py`); a build
that does not match is not timed.

Each build is timed with a working set that fits the caches (`--cache-bytes`)
and with one larger than the last level cache (`--memory-bytes`); a block's
time is the best of its repetitions.

## Output

`results.csv` in the work directory has a row per build, working set and block
with the time per item (cell, or fault face for dynamic rupture) and the
non-zero and hardware flops per item that yateto counts. `summary.md` gives the
time per cell and simulation in ns, with the hardware flops in thousands in
parentheses.
