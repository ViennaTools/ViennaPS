#!/usr/bin/env python3
"""Time the coverage solve as a share of a feature-scale run.

Runs one geometry over three mechanisms spanning the range of sizes, once from
the reaction file and once from the hand-written class for the chemistry that
has one, on each available flux engine. Writes a summary and a LaTeX table.

    python3 benchmark.py                  # from the build's example directory
    python3 benchmark.py --repeats 7
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys

DRIVER = './surfaceChemistry'
GEOM = ['--width', '80', '--depth', '120', '--grid', '1.5',
        '--thickness', '5', '--rays', '2000']

# label, mechanism, extra flags
ROWS = [
    ('silane_inputs', 'silane_inputs', []),
    ('sige_stack', 'sige_stack', []),
    ('diamond', 'diamond', []),
    ('ar_sputter', 'ar_sputter', []),
    ('sf6o2', 'sf6o2', []),
    ('sf6o2, hand-written', 'sf6o2', ['--handwritten']),
    ('polymer_etch', 'polymer_etch', []),
    ('gaas_cvd', 'gaas_cvd', []),
]


ENV = dict(os.environ)


def run(args):
    # every child gets the same explicit thread count, so the share reported
    # is not a function of whatever OpenMP would have defaulted to
    p = subprocess.run([DRIVER] + args, capture_output=True, text=True, env=ENV)
    return p.returncode, p.stdout + p.stderr


def fraction(mech, extra, engine):
    """Share of the run spent in the coverage solve, as a percentage."""
    args = ['-r', 'reactions/%s.mechanism.json' % mech] + extra + GEOM + \
           ['--profile', '--out', '/tmp/psbench']
    if engine == 'gpu':
        args.append('--gpu')
    code, out = run(args)
    if code != 0:
        return None
    m = re.search(r'fraction\s*:\s*([0-9.eE+-]+)\s*%', out)
    return float(m.group(1)) if m else None


def per_point(mech, extra):
    """Cost of one coverage solve at one point, in nanoseconds."""
    code, out = run(['-r', 'reactions/%s.mechanism.json' % mech] + extra +
                    ['--bench', '200000'])
    if code != 0:
        return None
    m = re.search(r'per point\s*:\s*([0-9.eE+-]+)\s*ns', out)
    return float(m.group(1)) if m else None


def wall(mech, extra, gpu):
    """Wall time of one full run, in seconds."""
    import time
    args = ['-r', 'reactions/%s.mechanism.json' % mech] + extra + GEOM + \
           ['--out', '/tmp/psbench']
    if gpu:
        args.append('--gpu')
    t0 = time.perf_counter()
    code, _ = run(args)
    return time.perf_counter() - t0 if code == 0 else None


def size(mech):
    with open('reactions/%s.mechanism.json' % mech) as f:
        d = json.load(f)
    return len(d['reactions']), len(d['coverages'])


def machine():
    info = {'platform': platform.platform(),
            'python': platform.python_version(),
            'OMP_NUM_THREADS': ENV.get('OMP_NUM_THREADS', 'unset')}
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    info['cpu'] = line.split(':', 1)[1].strip()
                    break
        info['cores'] = str(os.cpu_count())
    except OSError:
        pass
    try:
        g = subprocess.run(['nvidia-smi', '--query-gpu=name',
                            '--format=csv,noheader'],
                           capture_output=True, text=True)
        if g.returncode == 0:
            info['gpu'] = g.stdout.strip().splitlines()[0]
    except FileNotFoundError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--threads', type=int, default=None,
                    help='OMP threads; defaults to OMP_NUM_THREADS, else all '
                         'available. Recorded either way.')
    a = ap.parse_args()

    threads = (a.threads if a.threads is not None
               else int(ENV['OMP_NUM_THREADS']) if 'OMP_NUM_THREADS' in ENV
               else os.cpu_count())
    ENV['OMP_NUM_THREADS'] = str(threads)
    print('threads  : %d%s' % (threads,
          '' if a.threads is not None or 'OMP_NUM_THREADS' in os.environ
          else '  (every available processor; pass --threads to pin it)'))

    if not os.path.exists(DRIVER):
        sys.exit('run this from the build directory that holds %s' % DRIVER)

    engines = ['cpu']
    if fraction('sf6o2', [], 'gpu') is not None:
        engines.append('gpu')
    else:
        print('no device engine in this build, reporting the host engine only')

    results = {}
    for label, mech, extra in ROWS:
        line = '%-22s ' % label
        for engine in engines:
            vals = [fraction(mech, extra, engine) for _ in range(a.repeats)]
            vals = [v for v in vals if v is not None]
            results[(label, engine)] = statistics.median(vals) if vals else None
            line += ' %-3s %s' % (engine, '%8.4f %%' % results[(label, engine)]
                                  if vals else '  failed')
        print(line)

    speedup = None
    if 'gpu' in engines:
        host = [v for v in (wall('sf6o2', [], False)
                            for _ in range(a.repeats)) if v]
        dev = [v for v in (wall('sf6o2', [], True)
                           for _ in range(a.repeats)) if v]
        if host and dev:
            speedup = statistics.median(host) / statistics.median(dev)
            print('device engine is %.1fx faster in wall time' % speedup)

    bench = {}
    for label, mech, extra in [('file', 'sf6o2', []),
                               ('class', 'sf6o2', ['--handwritten'])]:
        vals = [per_point(mech, extra) for _ in range(a.repeats)]
        vals = [v for v in vals if v is not None]
        bench[label] = statistics.median(vals) if vals else None
        print('per point, %-6s %s' % (label,
              '%.1f ns' % bench[label] if vals else 'failed'))

    info = machine()
    with open('benchmark_results.txt', 'w') as f:
        for k, v in info.items():
            f.write('%-16s %s\n' % (k, v))
        f.write('repeats          %d (median reported)\n\n' % a.repeats)
        f.write('share of the run spent in the coverage solve, per cent\n')
        for (label, engine), v in results.items():
            f.write('  %-22s %-3s %s\n' % (label, engine,
                    '%.4f' % v if v is not None else 'failed'))
        if speedup:
            f.write('\ndevice engine wall time, relative to host: 1/%.1f\n'
                    % speedup)
        f.write('\ncost of one solve at one point, ns\n')
        for k, v in bench.items():
            f.write('  %-22s %s\n' % (k, '%.1f' % v if v is not None else 'failed'))
        if bench.get('file') and bench.get('class'):
            f.write('  %-22s %.1f\n' % ('ratio', bench['file'] / bench['class']))

    gpu = 'gpu' in engines

    def cell(label, engine):
        v = results.get((label, engine))
        return '---' if v is None else '$%.3f$' % v

    with open('benchmark_table.tex', 'w') as f:
        f.write('%% generated by benchmark.py -- do not edit by hand\n')
        f.write('\\begin{table}[htbp]\n\\centering\n\\footnotesize\n')
        f.write('\\begin{tabular}{@{}lcc%s@{}}\n\\toprule\n'
                % ('rr' if gpu else 'r'))
        f.write('mechanism & reactions & coverages & host'
                + (' & device' if gpu else '') + ' \\\\\n\\midrule\n')
        for label, mech, extra in ROWS:
            nr, nc = size(mech)
            name = ('\\texttt{%s}, hand-written' % mech.replace('_', '\\_')
                    if ',' in label else
                    '\\texttt{%s}' % label.replace('_', '\\_'))
            f.write('%s & %d & %d & %s%s \\\\\n'
                    % (name, nr, nc, cell(label, 'cpu'),
                       ' & ' + cell(label, 'gpu') if gpu else ''))
        f.write('\\bottomrule\n\\end{tabular}\n')
        f.write('\\caption{The share of a feature-scale run spent solving the '
                'coverage balance, as a percentage, on the host flux engine'
                + (' and on the device engine' if gpu else '') +
                '. Each row is the median of %d run%s of one geometry, a trench '
                '$80$\\,nm wide and $120$\\,nm deep on a $1.5$\\,nm grid at '
                '$2000$ rays per surface point, advanced by $5$\\,nm. The third '
                'row solves the same chemistry from the hand-written closed '
                'form of ViennaPS in place of the reaction file. The device '
                'engine leaves the solve on the host and shortens everything '
                'around it, so the same absolute cost is a larger share of a '
                'smaller total.}\n'
                % (a.repeats, '' if a.repeats == 1 else 's'))
        f.write('\\label{tab:cost}\n\\end{table}\n')

    # the figures the prose quotes, as macros, so the text cannot go stale
    with open('benchmark_numbers.tex', 'w') as f:
        f.write('%% generated by benchmark.py -- do not edit by hand\n')

        def macro(name, value, fmt='%.0f'):
            f.write('\\newcommand{\\%s}{%s}\n'
                    % (name, '??' if value is None else fmt % value))

        macro('costPerPointFile', bench.get('file'))
        macro('costPerPointClass', bench.get('class'), '%.1f')
        r = (bench['file'] / bench['class']
             if bench.get('file') and bench.get('class') else None)
        macro('costRatio', r)
        macro('costShareLargest', results.get(('gaas_cvd', 'cpu')), '%.2f')
        macro('costShareFile', results.get(('sf6o2', 'cpu')), '%.2f')
        macro('costShareClass', results.get(('sf6o2, hand-written', 'cpu')),
              '%.2f')
        macro('costShareFileGPU', results.get(('sf6o2', 'gpu')), '%.2f')
        macro('costShareLargestGPU', results.get(('gaas_cvd', 'gpu')), '%.2f')
        macro('costEngineSpeedup', speedup, '%.1f')

    print('\nwrote benchmark_results.txt, benchmark_table.tex '
          'and benchmark_numbers.tex')


if __name__ == '__main__':
    main()
