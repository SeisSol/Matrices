// Times the blocks of blocks.py as yateto generated them into the build
// directory, per cell (per fault face for dynamic rupture), or with --check
// runs each kernel once on known input and writes what it computed, for
// benchmark.py to compare with numpy.
//
// setup.h, written by benchmark.py next to the generated code, holds the
// configuration, the Taylor coefficients and the memory layouts of the data.

#include "init.h"
#include "kernel.h"
#include "setup.h"
#include "tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using real = double;
namespace yk = yateto::kernel;
namespace yt = yateto::tensor;

constexpr int N = setup::Degree;
constexpr int Faces = 4;

real* allocate(std::size_t count) {
  std::size_t bytes = ((count * sizeof(real) + 63) / 64) * 64;
  auto* data = static_cast<real*>(std::aligned_alloc(64, bytes == 0 ? 64 : bytes));
  if (data == nullptr) {
    std::fprintf(stderr, "out of memory\n");
    std::exit(1);
  }
  std::memset(data, 0, bytes == 0 ? 64 : bytes);
  return data;
}

// `count` items of `size` reals each, every item on a cache line of its own
struct Store {
  real* data = nullptr;
  std::size_t stride = 0;
  std::size_t count = 0;
  void init(std::size_t size, std::size_t items) {
    stride = ((size + 7) / 8) * 8;
    count = items;
    data = allocate(stride * items);
  }
  real* operator[](std::size_t i) const { return data + stride * i; }
  std::size_t bytes() const { return stride * count * sizeof(real); }
};

// the logical entry (s, row, column) of a tensor in its memory layout, or -1
// where the layout keeps nothing (which reads as zero)
long address(const setup::Layout& layout, unsigned s, unsigned row, unsigned column) {
  unsigned entry[3];
  if (layout.rank == 2) {
    entry[0] = row;
    entry[1] = column;
  } else {
    entry[0] = s;
    entry[1] = row;
    entry[2] = column;
  }
  long result = 0;
  for (int d = 0; d < layout.rank; ++d) {
    if (entry[d] < layout.start[d] || entry[d] >= layout.stop[d]) {
      return -1;
    }
    result += static_cast<long>(entry[d] - layout.start[d]) * layout.stride[d];
  }
  return result;
}

unsigned sims(const setup::Layout& layout) { return layout.rank == 2 ? 1 : layout.shape[0]; }
unsigned rows(const setup::Layout& layout) { return layout.shape[layout.rank - 2]; }
unsigned columns(const setup::Layout& layout) { return layout.shape[layout.rank - 1]; }

// the input benchmark.py expects: sin(0.1 + 0.37 h + 0.417 s + 0.731 i + 1.913 j)
real value(int h, unsigned s, unsigned i, unsigned j) {
  return std::sin(0.1 + 0.37 * h + 0.417 * s + 0.731 * i + 1.913 * j);
}

void fill(const setup::Layout& layout, real* data, int h, unsigned validRows = ~0u) {
  for (unsigned s = 0; s < sims(layout); ++s) {
    for (unsigned i = 0; i < rows(layout) && i < validRows; ++i) {
      for (unsigned j = 0; j < columns(layout); ++j) {
        long a = address(layout, s, i, j);
        if (a >= 0) {
          data[a] = value(h, s, i, j);
        }
      }
    }
  }
}

void dump(std::FILE* file, const std::string& name, const setup::Layout& layout, const real* data) {
  std::fprintf(file, "%s %u %u %u\n", name.c_str(), sims(layout), rows(layout), columns(layout));
  std::vector<real> values;
  for (unsigned s = 0; s < sims(layout); ++s) {
    for (unsigned i = 0; i < rows(layout); ++i) {
      for (unsigned j = 0; j < columns(layout); ++j) {
        long a = address(layout, s, i, j);
        values.push_back(a >= 0 ? data[a] : 0.0);
      }
    }
  }
  std::fwrite(values.data(), sizeof(real), values.size(), file);
}

template <typename Body>
double bestTime(Body&& body, int repetitions) {
  body();
  double best = 1e300;
  for (int r = 0; r < repetitions; ++r) {
    auto start = std::chrono::steady_clock::now();
    body();
    auto stop = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(stop - start).count());
  }
  return best;
}

struct Fault {
  std::size_t cell[2];
  unsigned face[2];
  unsigned orientation[2];  // against the canonical frame of the face
  unsigned side[2];         // b of V3mTo2n(a, b): 0 on the first side
};

struct Run {
  std::size_t cells = 0;
  std::size_t faults = 0;
  Store Q, I, dQ[N + 1], star[3], AminusT, traces;
  Store qp, F, q, Pk[N + 1];
  std::vector<std::size_t> neighbour;
  std::vector<unsigned> neighbourFace, orientation;
  std::vector<Fault> fault;

  yk::ck ck;
  yk::taylor taylor;
  yk::trace trace;
  yk::lift lift;
  yk::nodal nodal;
  yk::drTaylor drTaylor;
  yk::drEval drEval;
  yk::drBack drBack;
  yk::drTrace drTrace;
  yk::drCombine drCombine;
  yk::drFaceBack drFaceBack;

  void allocate(std::size_t numberOfCells, std::size_t numberOfFaults, unsigned seed) {
    cells = numberOfCells;
    faults = numberOfFaults;
    Q.init(yt::Q::size(), cells);
    I.init(yt::I::size(), cells);
    for (int k = 1; k <= N; ++k) {
      dQ[k].init(yt::dQ::size(k), cells);
    }
    for (int d = 0; d < 3; ++d) {
      star[d].init(yt::star::size(d), cells);
    }
    AminusT.init(yt::AminusT::size(), Faces * cells);
    static_assert(yt::T::Size == yt::Tn::Size, "both traces share a buffer");
    traces.init(yt::T::size(), Faces * cells);
    qp.init(yt::qp::size(), 2 * setup::TimePoints * faults);
    F.init(yt::F::size(), 2 * faults);
    q.init(yt::q::size(), 1);
    for (int k = 0; k <= N; ++k) {
      Pk[k].init(yt::Pk::size(k), 1);
    }

    std::mt19937 random(seed);
    // neighbours mostly close by, as in a mesh ordered along a curve
    const std::size_t offsets[Faces] = {1, 3, cells / 7 + 1, cells - 1};
    neighbour.resize(Faces * cells);
    neighbourFace.resize(Faces * cells);
    orientation.resize(Faces * cells);
    for (std::size_t c = 0; c < cells; ++c) {
      for (unsigned f = 0; f < Faces; ++f) {
        neighbour[Faces * c + f] = (c + offsets[f]) % cells;
        neighbourFace[Faces * c + f] = random() % Faces;
        orientation[Faces * c + f] = pickOrientation(f, random);
      }
    }
    fault.resize(faults);
    for (std::size_t i = 0; i < faults; ++i) {
      for (int s = 0; s < 2; ++s) {
        fault[i].cell[s] = random() % cells;
        fault[i].face[s] = random() % Faces;
        fault[i].orientation[s] = pickOrientation(fault[i].face[s], random);
        fault[i].side[s] = s == 0 ? 0 : 1 + random() % 3;
      }
    }

    // values of order one everywhere, so that no block meets denormals
    for (std::size_t c = 0; c < cells; ++c) {
      fill(setup::layout_Q, Q[c], 1 + c % 5);
      fill(setup::layout_I, I[c], 2 + c % 5);
      for (int k = 1; k <= N; ++k) {
        fill(setup::layout_dQ[k], dQ[k][c], 3 + k + c % 5);
      }
      for (int d = 0; d < 3; ++d) {
        fill(setup::layout_star, star[d][c], 20 + d + c % 3);
      }
      for (unsigned f = 0; f < Faces; ++f) {
        fill(setup::layout_AminusT, AminusT[Faces * c + f], 30 + f + c % 3);
        fill(setup::layout_T, traces[Faces * c + f], 40 + f + c % 3);
      }
    }
    for (std::size_t i = 0; i < 2 * faults; ++i) {
      fill(setup::layout_F, F[i], 50 + i % 7);
    }

    const auto& pool = yateto::Pool::host();
    ck.bindGlobals(pool);
    taylor.bindGlobals(pool);
    trace.bindGlobals(pool);
    lift.bindGlobals(pool);
    nodal.bindGlobals(pool);
    drTaylor.bindGlobals(pool);
    drEval.bindGlobals(pool);
    drBack.bindGlobals(pool);
    drTrace.bindGlobals(pool);
    drCombine.bindGlobals(pool);
    drFaceBack.bindGlobals(pool);
    for (int k = 0; k <= N; ++k) {
      taylor.cI(k) = setup::IntegralCoefficients[k];
      taylor.cQ(k) = setup::EndCoefficients[k];
    }
  }

  static unsigned pickOrientation(unsigned face, std::mt19937& random) {
    if (setup::AllOrientations) {
      return random() % 6;
    }
    return setup::SortedOrientation[face];
  }

  std::size_t bytesPerCell() const {
    std::size_t bytes = Q.stride + I.stride + 3 * star[0].stride +
                        Faces * (AminusT.stride + traces.stride);
    for (int k = 1; k <= N; ++k) {
      bytes += dQ[k].stride;
    }
    return bytes * sizeof(real);
  }

  // the blocks, for one cell or fault face

  void runCk(std::size_t c) {
    ck.Q = Q[c];
    for (int k = 1; k <= N; ++k) {
      ck.dQ(k) = dQ[k][c];
    }
    for (int d = 0; d < 3; ++d) {
      ck.star(d) = star[d][c];
    }
    ck.execute();
  }

  void runTaylor(std::size_t c) {
    taylor.Q = Q[c];
    taylor.I = I[c];
    for (int k = 1; k <= N; ++k) {
      taylor.dQ(k) = dQ[k][c];
    }
    taylor.execute();
  }

  void runTrace(std::size_t c) {
    trace.I = I[c];
    for (unsigned f = 0; f < Faces; ++f) {
      trace.T = traces[Faces * c + f];
      trace.execute(f, orientation[Faces * c + f]);
    }
  }

  template <typename Kernel>
  void runFlux(Kernel& kernel, std::size_t c) {
    kernel.Q = Q[c];
    for (unsigned f = 0; f < Faces; ++f) {
      std::size_t i = Faces * c + f;
      kernel.T = traces[i];
      kernel.Tn = traces[Faces * neighbour[i] + neighbourFace[i]];
      kernel.AminusT = AminusT[i];
      kernel.execute(f, orientation[i]);
    }
  }

  void setDerivatives(std::size_t c) {
    drTaylor.Q = Q[c];
    drTrace.Q = Q[c];
    for (int k = 1; k <= N; ++k) {
      drTaylor.dQ(k) = dQ[k][c];
      drTrace.dQ(k) = dQ[k][c];
    }
  }

  void runDrVolume(std::size_t i) {
    for (int s = 0; s < 2; ++s) {
      const Fault& face = fault[i];
      setDerivatives(face.cell[s]);
      drTaylor.q = q[0];
      drEval.q = q[0];
      for (int t = 0; t < setup::TimePoints; ++t) {
        for (int k = 0; k <= N; ++k) {
          drTaylor.cT(k) = setup::PointCoefficients[t][k];
        }
        drTaylor.execute();
        drEval.qp = qp[(2 * i + s) * setup::TimePoints + t];
        drEval.execute(face.face[s], face.side[s]);
      }
      drBack.Q = Q[face.cell[s]];
      drBack.F = F[2 * i + s];
      drBack.execute(face.face[s], face.side[s]);
    }
  }

  void runDrFace(std::size_t i) {
    for (int s = 0; s < 2; ++s) {
      const Fault& face = fault[i];
      setDerivatives(face.cell[s]);
      for (int k = 0; k <= N; ++k) {
        drTrace.Pk(k) = Pk[k][0];
        drCombine.Pk(k) = Pk[k][0];
      }
      drTrace.execute(face.face[s], face.orientation[s]);
      for (int t = 0; t < setup::TimePoints; ++t) {
        for (int k = 0; k <= N; ++k) {
          drCombine.cT(k) = setup::PointCoefficients[t][k];
        }
        drCombine.qp = qp[(2 * i + s) * setup::TimePoints + t];
        drCombine.execute();
      }
      drFaceBack.Q = Q[face.cell[s]];
      drFaceBack.F = F[2 * i + s];
      drFaceBack.execute(face.face[s], face.orientation[s]);
    }
  }
};

struct Result {
  const char* block;
  std::size_t items;
  double seconds;
  double nonZeroFlops;
  double hardwareFlops;
};

void benchmark(std::size_t bytes, int repetitions, unsigned seed) {
  Run run;
  // a first allocation of a few cells tells the size of a cell
  {
    Run probe;
    probe.allocate(16, 1, seed);
    std::size_t cells = std::max<std::size_t>(16, bytes / probe.bytesPerCell());
    run.allocate(cells, std::max<std::size_t>(4, cells / 4), seed);
  }

  double traceFlops[2] = {0, 0}, liftFlops[2] = {0, 0}, nodalFlops[2] = {0, 0};
  for (std::size_t i = 0; i < Faces * run.cells; ++i) {
    unsigned f = i % Faces, o = run.orientation[i];
    traceFlops[0] += yk::trace::nonZeroFlops(f, o);
    traceFlops[1] += yk::trace::hardwareFlops(f, o);
    liftFlops[0] += yk::lift::nonZeroFlops(f, o);
    liftFlops[1] += yk::lift::hardwareFlops(f, o);
    nodalFlops[0] += yk::nodal::nonZeroFlops(f, o);
    nodalFlops[1] += yk::nodal::hardwareFlops(f, o);
  }
  double volumeFlops[2] = {0, 0}, faceFlops[2] = {0, 0};
  for (const Fault& face : run.fault) {
    for (int s = 0; s < 2; ++s) {
      volumeFlops[0] += setup::TimePoints * (yk::drTaylor::NonZeroFlops +
                                             yk::drEval::nonZeroFlops(face.face[s], face.side[s])) +
                        yk::drBack::nonZeroFlops(face.face[s], face.side[s]);
      volumeFlops[1] += setup::TimePoints * (yk::drTaylor::HardwareFlops +
                                             yk::drEval::hardwareFlops(face.face[s], face.side[s])) +
                        yk::drBack::hardwareFlops(face.face[s], face.side[s]);
      faceFlops[0] += yk::drTrace::nonZeroFlops(face.face[s], face.orientation[s]) +
                      setup::TimePoints * yk::drCombine::NonZeroFlops +
                      yk::drFaceBack::nonZeroFlops(face.face[s], face.orientation[s]);
      faceFlops[1] += yk::drTrace::hardwareFlops(face.face[s], face.orientation[s]) +
                      setup::TimePoints * yk::drCombine::HardwareFlops +
                      yk::drFaceBack::hardwareFlops(face.face[s], face.orientation[s]);
    }
  }
  const double cells = run.cells, faults = run.faults;

  std::vector<Result> results;
  auto overCells = [&](auto&& body) {
    return bestTime(
        [&] {
          for (std::size_t c = 0; c < run.cells; ++c) {
            body(c);
          }
        },
        repetitions);
  };
  results.push_back({"ck", run.cells, overCells([&](std::size_t c) { run.runCk(c); }),
                     double(yk::ck::NonZeroFlops), double(yk::ck::HardwareFlops)});
  results.push_back({"taylor", run.cells, overCells([&](std::size_t c) { run.runTaylor(c); }),
                     double(yk::taylor::NonZeroFlops), double(yk::taylor::HardwareFlops)});
  results.push_back({"trace", run.cells, overCells([&](std::size_t c) { run.runTrace(c); }),
                     traceFlops[0] / cells, traceFlops[1] / cells});
  results.push_back({"lift", run.cells,
                     overCells([&](std::size_t c) { run.runFlux(run.lift, c); }),
                     liftFlops[0] / cells, liftFlops[1] / cells});
  results.push_back({"nodal", run.cells,
                     overCells([&](std::size_t c) { run.runFlux(run.nodal, c); }),
                     nodalFlops[0] / cells, nodalFlops[1] / cells});
  auto overFaults = [&](auto&& body) {
    return bestTime(
        [&] {
          for (std::size_t i = 0; i < run.faults; ++i) {
            body(i);
          }
        },
        repetitions);
  };
  results.push_back({"dr-volume", run.faults,
                     overFaults([&](std::size_t i) { run.runDrVolume(i); }),
                     volumeFlops[0] / faults, volumeFlops[1] / faults});
  results.push_back({"dr-face", run.faults, overFaults([&](std::size_t i) { run.runDrFace(i); }),
                     faceFlops[0] / faults, faceFlops[1] / faults});

  std::printf("# %s %s order %d simulations %d %s orientations, %zu cells (%.1f MB), "
              "%zu fault faces\n",
              setup::Family, setup::Storage, setup::Order, setup::Simulations,
              setup::AllOrientations ? "all" : "sorted", run.cells,
              run.cells * run.bytesPerCell() / 1e6, run.faults);
  std::printf("block,items,ns_per_item,nonzero_flops,hardware_flops,hardware_gflops\n");
  for (const Result& r : results) {
    double ns = 1e9 * r.seconds / r.items;
    std::printf("%s,%zu,%.2f,%.0f,%.0f,%.2f\n", r.block, r.items, ns, r.nonZeroFlops,
                r.hardwareFlops, r.hardwareFlops / ns);
  }
}

void check(const char* path) {
  Run run;
  run.allocate(2, 1, 1);
  std::FILE* file = std::fopen(path, "wb");
  if (file == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", path);
    std::exit(1);
  }
  auto name = [](const char* base, int a, int b = -1) {
    std::string result = std::string(base) + "(" + std::to_string(a);
    if (b >= 0) {
      result += "," + std::to_string(b);
    }
    return result + ")";
  };

  fill(setup::layout_Q, run.Q[0], 1);
  for (int d = 0; d < 3; ++d) {
    fill(setup::layout_star, run.star[d][0], 10 + d);
  }
  run.runCk(0);
  for (int k = 1; k <= N; ++k) {
    dump(file, name("dQ", k), setup::layout_dQ[k], run.dQ[k][0]);
  }
  run.runTaylor(0);
  dump(file, "I", setup::layout_I, run.I[0]);
  dump(file, "Qend", setup::layout_Q, run.Q[0]);

  for (int c = 0; c < setup::NumberOfCombinations; ++c) {
    unsigned f = setup::Combinations[c][0], o = setup::Combinations[c][1];
    run.trace.I = run.I[0];
    run.trace.T = run.traces[0];
    run.trace.execute(f, o);
    dump(file, name("trace", f, o), setup::layout_T, run.traces[0]);
  }

  fill(setup::layout_AminusT, run.AminusT[0], 20);
  auto flux = [&](auto& kernel, const char* base, unsigned f, unsigned o) {
    fill(setup::layout_T, run.traces[0], 30);
    fill(setup::layout_Tn, run.traces[1], 31);
    fill(setup::layout_Q, run.Q[0], 2);
    kernel.Q = run.Q[0];
    kernel.T = run.traces[0];
    kernel.Tn = run.traces[1];
    kernel.AminusT = run.AminusT[0];
    kernel.execute(f, o);
    dump(file, name(base, f, o), setup::layout_Q, run.Q[0]);
  };
  for (int c = 0; c < setup::NumberOfCombinations; ++c) {
    unsigned f = setup::Combinations[c][0], o = setup::Combinations[c][1];
    flux(run.lift, "lift", f, o);
    flux(run.nodal, "nodal", f, o);
  }

  // dynamic rupture at the first time point, with the derivatives of the cell
  // as ck computed them from Q(1)
  fill(setup::layout_Q, run.Q[0], 1);
  run.setDerivatives(0);
  for (int k = 0; k <= N; ++k) {
    run.drTaylor.cT(k) = setup::PointCoefficients[0][k];
    run.drCombine.cT(k) = setup::PointCoefficients[0][k];
    run.drTrace.Pk(k) = run.Pk[k][0];
    run.drCombine.Pk(k) = run.Pk[k][0];
  }
  run.drTaylor.q = run.q[0];
  run.drTaylor.execute();
  dump(file, "q", setup::layout_q, run.q[0]);
  fill(setup::layout_F, run.F[0], 40);
  for (unsigned a = 0; a < 4; ++a) {
    for (unsigned b = 0; b < 4; ++b) {
      run.drEval.q = run.q[0];
      run.drEval.qp = run.qp[0];
      run.drEval.execute(a, b);
      dump(file, name("drEval", a, b), setup::layout_qp, run.qp[0]);
      fill(setup::layout_Q, run.Q[1], 2);
      run.drBack.Q = run.Q[1];
      run.drBack.F = run.F[0];
      run.drBack.execute(a, b);
      dump(file, name("drBack", a, b), setup::layout_Q, run.Q[1]);
    }
  }
  for (int c = 0; c < setup::NumberOfCombinations; ++c) {
    unsigned f = setup::Combinations[c][0], o = setup::Combinations[c][1];
    run.drTrace.execute(f, o);
    run.drCombine.qp = run.qp[0];
    run.drCombine.execute();
    dump(file, name("drFace", f, o), setup::layout_qp, run.qp[0]);
    fill(setup::layout_Q, run.Q[1], 2);
    run.drFaceBack.Q = run.Q[1];
    run.drFaceBack.F = run.F[0];
    run.drFaceBack.execute(f, o);
    dump(file, name("drFaceBack", f, o), setup::layout_Q, run.Q[1]);
  }
  std::fclose(file);
}

} // namespace

int main(int argc, char** argv) {
  std::size_t bytes = 256ul << 20;
  int repetitions = 5;
  unsigned seed = 1;
  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];
    if (argument == "--check" && i + 1 < argc) {
      check(argv[++i]);
      return 0;
    }
    if (argument == "--bytes" && i + 1 < argc) {
      bytes = std::strtoull(argv[++i], nullptr, 10);
    } else if (argument == "--repetitions" && i + 1 < argc) {
      repetitions = std::atoi(argv[++i]);
    } else if (argument == "--seed" && i + 1 < argc) {
      seed = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: %s [--bytes B] [--repetitions R] [--seed S] | --check FILE\n",
                   argv[0]);
      return 1;
    }
  }
  benchmark(bytes, repetitions, seed);
  return 0;
}
