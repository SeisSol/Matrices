#!/usr/bin/env python3
"""
Generates the kernels of blocks.py with yateto for the bases, storages, orders,
numbers of fused simulations and architectures asked for, checks every build
against numpy (check.py), times it with driver.cpp and writes the times per
cell (per fault face for dynamic rupture) as CSV, with a summary in Markdown.

    python3 benchmark.py [--orders 4 6 8] [--families dubiner bernstein]
        [--storages dense csc immediate] [--simulations 1 8]
        [--archs dhsw dskx] [--orientations sorted] [--work DIR] [--jobs 2]

yateto must be importable (or given with --yateto) and have its headers in
include/ next to the package; the GEMMs go to pspamm-generator (--pspamm).
The working set of a run is either small enough for the caches ("cache",
--cache-bytes) or larger than the last level cache ("memory",
--memory-bytes); the time of a block is the best of its repetitions.
"""

import argparse
import concurrent.futures
import csv
import importlib
import itertools
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

ARCH_FLAGS = {
    "dhsw": ["-march=haswell"],
    "dskx": ["-march=skylake-avx512", "-mprefer-vector-width=512"],
}
SOURCES = ["kernel.cpp", "init.cpp", "pool.cpp", "tensor.cpp", "subroutine.cpp"]
DT = 0.25
BLOCKS = ("ck", "taylor", "trace", "lift", "nodal", "dr-volume", "dr-face")


def name(family, storage, simulations, order, arch, orientations):
    return f"{family}-{storage}-s{simulations}-o{order}-{arch}-{orientations}"


def import_yateto(path):
    if path:
        sys.path.insert(0, path)
    return importlib.import_module("yateto")


def include_directory(yateto):
    return os.path.join(os.path.dirname(os.path.dirname(yateto.__file__)), "include")


def layout(tensor):
    """The memory layout of a tensor as the driver's setup::Layout."""
    memory = tensor.memoryLayout()
    shape = list(tensor.shape())
    start = [r.start for r in memory.bbox()]
    stop = [r.stop for r in memory.bbox()]
    stride = list(memory.stride())
    pad = lambda v: ", ".join(str(int(x)) for x in v + [0] * (3 - len(v)))
    return f"{{{len(shape)}, {{{pad(shape)}}}, {{{pad(start)}}}, {{{pad(stop)}}}, {{{pad(stride)}}}}}"


def setup_header(kernels, orientations):
    """setup.h for the driver: configuration, coefficients and layouts."""
    from seissol_matrices import dg_matrices

    ops = kernels.ops
    N = ops.degree
    c = ops.taylor_coefficients(DT, scaled=True)
    numbers = lambda values: ", ".join(repr(float(v)) for v in values)
    runtime = kernels.runtime
    sorted_orientations = [
        dg_matrices.FACE_ORIENTATIONS.index(dg_matrices.sorted_vertex_ranks(f))
        for f in range(4)
    ]
    lines = [
        "// written by benchmark.py",
        "#pragma once",
        "namespace setup {",
        "struct Layout {",
        "  int rank;",
        "  unsigned shape[3];",
        "  unsigned start[3];",
        "  unsigned stop[3];",
        "  long stride[3];",
        "};",
        f'constexpr const char* Family = "{ops.family}";',
        f'constexpr const char* Storage = "{kernels.storage}";',
        f"constexpr int Order = {ops.order};",
        f"constexpr int Degree = {N};",
        f"constexpr int Simulations = {kernels.simulations};",
        f"constexpr int TimePoints = {len(c['points'])};",
        f"constexpr bool AllOrientations = {'true' if orientations == 'all' else 'false'};",
        "constexpr unsigned SortedOrientation[4] = {"
        + ", ".join(str(sorted_orientations[f]) for f in range(4))
        + "};",
        f"constexpr int NumberOfCombinations = {len(ops.combos)};",
        "constexpr unsigned Combinations[][2] = {"
        + ", ".join(f"{{{f}, {o}}}" for f, o in ops.combos)
        + "};",
        f"constexpr double IntegralCoefficients[] = {{{numbers(c['integral'])}}};",
        f"constexpr double EndCoefficients[] = {{{numbers(c['end'])}}};",
        "constexpr double PointCoefficients[][Degree + 1] = {"
        + ", ".join(f"{{{numbers(row)}}}" for row in c["points"])
        + "};",
    ]
    for tensor in ("Q", "I", "AminusT", "T", "Tn", "q", "qp", "F"):
        lines.append(f"constexpr Layout layout_{tensor} = {layout(runtime[tensor])};")
    lines.append(f"constexpr Layout layout_star = {layout(runtime['star(0)'])};")
    dQ = [layout(runtime["Q"])] + [layout(runtime[f"dQ({k})"]) for k in range(1, N + 1)]
    lines.append(f"constexpr Layout layout_dQ[] = {{{', '.join(dQ)}}};")
    lines.append("} // namespace setup")
    return "\n".join(lines) + "\n"


def build(args, config):
    """Generates and compiles one configuration in its directory under the work
    directory; returns the directory."""
    family, storage, simulations, order, arch, orientations = config
    yateto = import_yateto(args.yateto)
    from yateto import Generator, useArchitectureIdentifiedBy
    from yateto.gemm_configuration import GeneratorCollection, PSpaMM
    import blocks

    directory = os.path.join(args.work, name(*config))
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    architecture = useArchitectureIdentifiedBy(arch)
    generator = Generator(architecture)
    ops = blocks.Operators(order, family, orientations)
    kernels = blocks.Kernels(ops, storage, simulations)
    kernels.add_to(generator)
    with open(os.path.join(directory, "build.log"), "w") as log:
        stdout = sys.stdout
        sys.stdout = log
        try:
            generator.generate(
                directory,
                gemm_cfg=GeneratorCollection([PSpaMM(architecture, cmd=args.pspamm)]),
            )
        finally:
            sys.stdout = stdout
        with open(os.path.join(directory, "setup.h"), "w") as f:
            f.write(setup_header(kernels, orientations))
        command = (
            [args.cxx, "-O3", "-std=c++17", "-fopenmp-simd", "-DNDEBUG"]
            + ARCH_FLAGS[arch]
            + args.cxxflags.split()
            + ["-I", directory, "-I", include_directory(yateto)]
            + [os.path.join(HERE, "driver.cpp")]
            + [os.path.join(directory, source) for source in SOURCES]
            + ["-o", os.path.join(directory, "driver")]
        )
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    return directory


def check(config, directory, operators):
    """Runs the driver's check and compares with numpy; returns the failures and
    the largest relative error."""
    family, storage, simulations, order, arch, orientations = config
    import blocks
    from check import Reference, compare, read

    key = (order, family, orientations)
    if key not in operators:
        operators[key] = blocks.Operators(order, family, orientations)
    ops = operators[key]
    path = os.path.join(directory, "check.bin")
    subprocess.run([os.path.join(directory, "driver"), "--check", path], check=True)
    records = read(path)
    reference = Reference(ops, simulations, ops.taylor_coefficients(DT, scaled=False))
    return compare(records, reference.expected(records))


def run(directory, regime, size, repetitions):
    """The driver's results for a working set of `size` bytes."""
    result = subprocess.run(
        [
            os.path.join(directory, "driver"),
            "--bytes",
            str(size),
            "--repetitions",
            str(repetitions),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with open(os.path.join(directory, f"run-{regime}.txt"), "w") as f:
        f.write(result.stdout)
    lines = [line for line in result.stdout.splitlines() if not line.startswith("#")]
    return list(csv.DictReader(lines)), result.stdout.splitlines()[0]


def summary(rows, args):
    """Markdown tables of ns per cell and simulation (per fault face and
    simulation for dynamic rupture), one table per architecture, number of
    simulations and working set, a column per basis and storage."""
    variants = [f"{f} {s}" for f, s in itertools.product(args.families, args.storages)]
    out = []
    for arch, simulations, regime in itertools.product(
        args.archs, args.simulations, ("cache", "memory")
    ):
        selected = [
            r
            for r in rows
            if r["arch"] == arch
            and int(r["simulations"]) == simulations
            and r["regime"] == regime
        ]
        if not selected:
            continue
        out.append(
            f"### {arch}, {simulations} simulation{'s' if simulations > 1 else ''}, "
            f"working set: {regime}\n"
        )
        out.append("| order | block | " + " | ".join(variants) + " |")
        out.append("|---|---|" + "---:|" * len(variants))
        for order in args.orders:
            for block in BLOCKS:
                cells = []
                for variant in variants:
                    family, storage = variant.split()
                    match = [
                        r
                        for r in selected
                        if r["family"] == family
                        and r["storage"] == storage
                        and int(r["order"]) == order
                        and r["block"] == block
                    ]
                    if match:
                        r = match[0]
                        ns = float(r["ns_per_item"]) / simulations
                        hardware = float(r["hardware_flops"]) / simulations
                        cells.append(f"{ns:.0f} ({hardware / 1e3:.1f}k)")
                    else:
                        cells.append("")
                out.append(f"| {order} | {block} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--families", nargs="+", default=["dubiner", "bernstein"])
    parser.add_argument("--storages", nargs="+", default=["dense", "csc", "immediate"])
    parser.add_argument("--simulations", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--archs", nargs="+", default=["dhsw", "dskx"])
    parser.add_argument("--orientations", default="sorted", choices=["sorted", "all"])
    parser.add_argument("--work", default="basis_benchmark_work")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--yateto", default=None)
    parser.add_argument("--pspamm", default="pspamm-generator")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--cxxflags", default="")
    parser.add_argument("--cache-bytes", type=int, default=1 << 20)
    parser.add_argument("--memory-bytes", type=int, default=512 << 20)
    parser.add_argument("--cache-repetitions", type=int, default=200)
    parser.add_argument("--memory-repetitions", type=int, default=5)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--build-one", nargs=6, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.work = os.path.abspath(args.work)
    import_yateto(args.yateto)

    if args.build_one:
        family, storage, simulations, order, arch, orientations = args.build_one
        build(args, (family, storage, int(simulations), int(order), arch, orientations))
        return

    configs = list(
        itertools.product(
            args.families,
            args.storages,
            args.simulations,
            args.orders,
            args.archs,
            [args.orientations],
        )
    )
    os.makedirs(args.work, exist_ok=True)

    if not args.skip_build:

        def build_in_process(config):
            command = [sys.executable, os.path.abspath(__file__), "--work", args.work]
            command += ["--pspamm", args.pspamm, "--cxx", args.cxx]
            command += ["--cxxflags=" + args.cxxflags]
            if args.yateto:
                command += ["--yateto", args.yateto]
            command += ["--build-one"] + [str(c) for c in config]
            start = time.time()
            result = subprocess.run(command, capture_output=True, text=True)
            return config, result, time.time() - start

        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            for config, result, seconds in pool.map(build_in_process, configs):
                status = "built" if result.returncode == 0 else "FAILED"
                print(f"{status} {name(*config)} in {seconds:.0f} s", flush=True)
                if result.returncode != 0:
                    print(result.stdout[-2000:], result.stderr[-4000:])

    operators = {}
    rows = []
    for config in configs:
        directory = os.path.join(args.work, name(*config))
        if not os.path.exists(os.path.join(directory, "driver")):
            continue
        failures, worst = check(config, directory, operators)
        state = "ok" if not failures else f"{len(failures)} FAILED"
        print(f"check {name(*config)}: {state} (largest error {worst:.1e})", flush=True)
        for failure in failures[:10]:
            print("   ", *failure)
        if failures or args.skip_run:
            continue
        family, storage, simulations, order, arch, orientations = config
        for regime, size, repetitions in (
            ("cache", args.cache_bytes, args.cache_repetitions),
            ("memory", args.memory_bytes, args.memory_repetitions),
        ):
            results, header = run(directory, regime, size, repetitions)
            print("   ", regime, header[2:], flush=True)
            for r in results:
                r.update(
                    family=family,
                    storage=storage,
                    simulations=simulations,
                    order=order,
                    arch=arch,
                    orientations=orientations,
                    regime=regime,
                )
                rows.append(r)

    if rows:
        fields = [
            "family",
            "storage",
            "simulations",
            "order",
            "arch",
            "orientations",
            "regime",
            "block",
            "items",
            "ns_per_item",
            "nonzero_flops",
            "hardware_flops",
            "hardware_gflops",
        ]
        with open(os.path.join(args.work, "results.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with open(os.path.join(args.work, "summary.md"), "w") as f:
            f.write(summary(rows, args))
        print(f"results in {args.work}/results.csv and summary.md")


if __name__ == "__main__":
    main()
