#!/usr/bin/env python3
"""The machine a timing run was taken on, as LaTeX the paper can include.

A timing is only meaningful beside the machine that produced it, and a machine
described from memory drifts away from the one measured. This is therefore
called by benchmark.py at the end of its run, so `machine.tex` and
`benchmark_numbers.tex` are written by the same invocation on the same host.

    python3 machine.py [BUILD_DIR]      write machine.tex and print it

BUILD_DIR is the directory holding CMakeCache.txt, used for the compiler and
the flags the driver was actually built with. It defaults to the build tree an
in-tree build produces.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUILD = os.path.normpath(
    os.path.join(HERE, os.pardir, os.pardir, 'build'))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ''


def first(pattern, text, group=1, default=''):
    m = re.search(pattern, text, re.M)
    return m.group(group).strip() if m else default


def cpu():
    info = sh('lscpu')
    model = first(r'^Model name:\s*(.+)$', info)
    if not model:
        model = first(r'^model name\s*:\s*(.+)$', open('/proc/cpuinfo').read())
    sockets = first(r'^Socket\(s\):\s*(\d+)', info, default='1')
    per_socket = first(r'^Core\(s\) per socket:\s*(\d+)', info, default='')
    cores = str(int(sockets) * int(per_socket)) if per_socket else ''
    # the logical count as the kernel reports it. A hybrid part gives its
    # performance cores two threads and its efficiency cores one, so cores
    # times threads-per-core overcounts it
    threads = first(r'^CPU\(s\):\s*(\d+)', info) or str(os.cpu_count())
    mhz = first(r'^CPU max MHz:\s*([\d.]+)', info)
    ghz = '%.1f' % (float(mhz) / 1000.) if mhz else ''
    return model, cores, threads, ghz


def memory():
    kb = first(r'^MemTotal:\s*(\d+)', open('/proc/meminfo').read())
    return str(round(int(kb) / 1048576)) if kb else ''


def gpu():
    out = sh('nvidia-smi --query-gpu=name,driver_version,memory.total '
             '--format=csv,noheader')
    if not out or ',' not in out:
        return '', '', ''
    name, driver, mem = [x.strip() for x in out.split('\n')[0].split(',')]
    return name, driver, mem.replace(' MiB', '')


def compiler(build):
    cache = os.path.join(build, 'CMakeCache.txt')
    cxx, build_type = '', ''
    if os.path.exists(cache):
        text = open(cache, errors='replace').read()
        cxx = first(r'^CMAKE_CXX_COMPILER:\w+=(.+)$', text)
        build_type = first(r'^CMAKE_BUILD_TYPE:\w+=(.+)$', text)
    if not cxx:
        cxx = sh('which c++') or 'c++'
    ver = sh('"%s" --version' % cxx).split('\n')[0] if cxx else ''
    # "g++ (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0" -> "g++ 12.3.0"
    name = os.path.basename(cxx)
    num = first(r'(\d+\.\d+\.\d+)\s*$', ver)
    return ('%s %s' % (name, num)).strip(), (build_type or 'Release')


def fields(build):
    model, cores, threads, ghz = cpu()
    gname, gdriver, gmem = gpu()
    cxx, build_type = compiler(build)
    osname = first(r'^PRETTY_NAME="?([^"\n]+)"?', open('/etc/os-release').read())
    return {
        'machineCPU': model,
        'machineCores': cores,
        'machineThreads': threads,
        'machineClock': ghz,
        'machineRAM': memory(),
        'machineGPU': gname,
        'machineGPUDriver': gdriver,
        'machineGPUMemory': gmem,
        'machineCUDA': first(r'release ([\d.]+)', sh('nvcc --version')),
        'machineOS': osname,
        'machineKernel': sh('uname -r'),
        'machineCompiler': cxx,
        'machineBuildType': build_type,
        'machineThreadsUsed': os.environ.get(
            'OMP_NUM_THREADS', str(os.cpu_count())),
    }


def main():
    build = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BUILD
    f = fields(build)
    lines = ['%% written by machine.py on the host the timings were taken on',
             '%% do not edit by hand']
    for k, v in f.items():
        lines.append('\\newcommand{\\%s}{%s}' % (k, v or 'unknown'))
    out = '\n'.join(lines) + '\n'
    with open('machine.tex', 'w') as fh:
        fh.write(out)
    print(out, end='')
    missing = [k for k, v in f.items() if not v]
    if missing:
        print('\n%% not detected: %s' % ', '.join(missing))
    return 0


if __name__ == '__main__':
    sys.exit(main())
