"""Run bmk.py in separate batch-size and length sweeps, isolating every variant/mode.

python sweep_bmk.py --bmk bmk.py --output sweep_results
python sweep_bmk.py --bmk bmk.py --output sweep_results --resume
SVG plots need only the standard library; install matplotlib for PNG plots.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def read_result(path, job):
    result = json.loads(path.read_text(encoding='utf-8'))
    rows = result['results']
    if len(rows) != 1:
        raise ValueError('Expected one result per isolated process')
    row = rows[0]
    if row['variant'] != job['variant'] or row['mode'] != job['mode']:
        raise ValueError('Result variant/mode does not match job')
    ms = float(row['median_ms'])
    if not math.isfinite(ms) or ms <= 0:
        raise ValueError('Invalid median_ms')
    cfg = result['config']
    for key, expected in [('batch_size', job['batch_size']), ('seq_len', job['length']),
                          ('context_len', job['length'])]:
        if cfg[key] != expected:
            raise ValueError(f'Result {key} does not match job')
    return row


FIELDS = ['sweep_axes', 'batch_size', 'length', 'mode', 'variant', 'status', 'median_ms',
          'min_ms', 'max_ms', 'tokens_per_second', 'speedup_vs_standard',
          'gpu_parameter_mib', 'cpu_parameter_mib', 'offload_gpu_buffer_mib',
          'offload_pinned_mib', 'wall_seconds', 'exit_code', 'log', 'result', 'error']


def summarize(output, records):
    rows = []
    standards = {(r['batch_size'], r['length'], r['mode']): r['metrics']['median_ms']
                 for r in records if r['status'] == 'ok' and r['variant'] == 'standard'}
    for r in records:
        row = {k: r.get(k, '') for k in FIELDS}
        if r['status'] == 'ok':
            m = r['metrics']
            row.update({k: m[k] for k in FIELDS if k in m})
            tokens = r['batch_size'] * (r['length'] if r['mode'] == 'prefill' else 1)
            row['tokens_per_second'] = tokens * 1000 / m['median_ms']
            base = standards.get((r['batch_size'], r['length'], r['mode']))
            row['speedup_vs_standard'] = base / m['median_ms'] if base is not None else ''
        rows.append(row)
    temp = output / 'summary.csv.tmp'
    with temp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output / 'summary.csv')
    write_json(output / 'summary.json', records)
    for axis in ('batch', 'length'):
        with (output / f'{axis}_sweep.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(row for row in rows if axis in row['sweep_axes'].split(','))


def plot_results(output, records):
    """Two figures, each with prefill/decode panels. Failures are gaps, not zero."""
    from html import escape
    import textwrap

    colors = {'standard': '#3878bf', 'ma_gpu': '#159b79', 'ma_offload': '#df7930'}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    for axis in ('batch', 'length'):
        selected = [r for r in records if axis in r.get('sweep_axes', '').split(',')]
        if not selected:
            continue
        key = 'batch_size' if axis == 'batch' else 'length'
        fixed_key = 'length' if axis == 'batch' else 'batch_size'
        fixed_values = sorted({r[fixed_key] for r in selected})
        xs = sorted({r[key] for r in selected})
        labels = [str(x) if axis == 'batch' else f'{x // 1024}k' if x % 1024 == 0 else str(x) for x in xs]
        xlabel = 'Batch size' if axis == 'batch' else 'Length (tokens; 1k = 1024)'
        title = ('Batch-size sweep' if axis == 'batch' else 'Length sweep') + f' | fixed {fixed_key}=' + ','.join(map(str, fixed_values))
        note = 'Median latency; whiskers = min/max across rounds (not confidence intervals). Missing/failed runs are not zero.'
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="720">',
               '<rect width="1400" height="720" fill="white"/>',
               '<style>text{font-family:Arial,sans-serif;fill:#263449}</style>',
               f'<text x="32" y="38" font-size="24">{escape(title)}</text>',
               f'<text x="32" y="66" font-size="13">{escape(note)}</text>']
        if plt is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
            fig.suptitle(title, fontsize=15)
            fig.text(0.5, 0.925, note, ha='center', fontsize=8)
        for panel, mode in enumerate(('prefill', 'decode')):
            entries = [r for r in selected if r['mode'] == mode]
            series = {}
            missing = []
            for variant in colors:
                lookup = {r[key]: r for r in entries if r['variant'] == variant}
                points = []
                for x in xs:
                    r = lookup.get(x)
                    if r is None or r['status'] != 'ok':
                        points.append(None)
                        if r is not None:
                            missing.append(f"{variant}@{x}: {r['status']}")
                    else:
                        m = r['metrics']
                        mid = float(m['median_ms'])
                        points.append((mid, min(mid, float(m.get('min_ms', mid))),
                                       max(mid, float(m.get('max_ms', mid)))))
                series[variant] = points
            ymax = max([pt[2] for pts in series.values() for pt in pts if pt is not None] or [1]) * 1.15
            has_data = any(pt is not None for pts in series.values() for pt in pts)
            left, top, width, height = 85 + panel*690, 130, 555, 365
            xpixel = lambda i: left + 30 + (width-60)*i/max(len(xs)-1, 1)
            ypixel = lambda v: top + height - v/ymax*height
            svg.append(f'<text x="{left}" y="108" font-size="19">{mode.capitalize()}</text>')
            for i in range(6):
                val = ymax*i/5
                yy = ypixel(val)
                svg.append(f'<path d="M{left} {yy} h{width}" stroke="#e0e5ec"/>')
                svg.append(f'<text x="{left-10}" y="{yy+4}" text-anchor="end" font-size="12">{val:.2f}</text>')
            svg.append(f'<text transform="translate({left-57},{top+height/2}) rotate(-90)" text-anchor="middle" font-size="13">Latency (ms)</text>')
            for i, label in enumerate(labels):
                svg.append(f'<text x="{xpixel(i)}" y="{top+height+25}" text-anchor="middle" font-size="13">{label}</text>')
            svg.append(f'<text x="{left+width/2}" y="{top+height+52}" text-anchor="middle" font-size="13">{escape(xlabel)}</text>')
            if not has_data:
                svg.append(f'<text x="{left+width/2}" y="{top+height/2}" text-anchor="middle" font-size="17">No successful measurements</text>')
                svg.append(f'<text x="{left+width/2}" y="{top+height/2+25}" text-anchor="middle" font-size="12">Check status/error in summary.csv and runs/*.log</text>')
            for variant, pts in series.items():
                color = colors[variant]
                previous = None
                for i, pt in enumerate(pts):
                    if pt is None:
                        previous = None
                        continue
                    mid, low, high = pt
                    xx, yy = xpixel(i), ypixel(mid)
                    if previous is not None:
                        svg.append(f'<path d="M{previous[0]} {previous[1]} L{xx} {yy}" fill="none" stroke="{color}" stroke-width="2.5"/>')
                    svg.append(f'<path d="M{xx} {ypixel(low)} V{ypixel(high)} M{xx-4} {ypixel(low)} h8 M{xx-4} {ypixel(high)} h8" stroke="{color}"/>')
                    svg.append(f'<circle cx="{xx}" cy="{yy}" r="4" fill="{color}"><title>{variant}, {labels[i]}: {mid:.4f} ms [{low:.4f}, {high:.4f}]</title></circle>')
                    previous = xx, yy
                if plt is not None and any(pt is not None for pt in pts):
                    y = [pt[0] if pt else float('nan') for pt in pts]
                    low = [pt[0]-pt[1] if pt else 0 for pt in pts]
                    high = [pt[2]-pt[0] if pt else 0 for pt in pts]
                    axes[panel].errorbar(range(len(xs)), y, yerr=[low, high],
                                         label=variant, color=color, marker='o', capsize=3)
            failure_note = '; '.join(missing) or ('No results for this mode.' if not entries else '')
            for line_no, line in enumerate(textwrap.wrap(failure_note, width=79)):
                svg.append(f'<text x="{left}" y="{570+line_no*15}" font-size="11">{escape(line)}</text>')
            if plt is not None:
                ax = axes[panel]
                ax.set(title=mode.capitalize(), xlabel=xlabel, ylabel='Latency (ms)', ylim=(0, ymax))
                ax.set_xticks(range(len(xs)), labels)
                ax.grid(axis='y', alpha=0.25)
                if not has_data:
                    ax.text(0.5, 0.5, 'No successful measurements\nCheck summary.csv and runs/*.log',
                            transform=ax.transAxes, ha='center', va='center', fontsize=11)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()
                ax.text(0, -0.22, '\n'.join(textwrap.wrap(failure_note, 70)),
                        transform=ax.transAxes, va='top', fontsize=7)
        for i, (name, color) in enumerate(colors.items()):
            xx = 460+i*185
            svg.append(f'<path d="M{xx} 694 h28" stroke="{color}" stroke-width="3"/><text x="{xx+35}" y="698" font-size="13">{name}</text>')
        svg.append('</svg>')
        (output / f'{axis}_sweep.svg').write_text('\n'.join(svg), encoding='utf-8')
        if plt is not None:
            fig.subplots_adjust(top=0.83, bottom=0.3, wspace=0.28)
            fig.savefig(output / f'{axis}_sweep.png', dpi=180)
            plt.close(fig)
    print('Plots saved: *_sweep.svg' + (' and *_sweep.png' if plt is not None else ' (install matplotlib for PNG)'), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bmk', type=Path, default=Path(__file__).with_name('bmk.py'))
    p.add_argument('--output', type=Path, default=Path('sweep_results'))
    p.add_argument('--batch-sizes', nargs='+', type=int, default=[8, 16, 32, 64])
    p.add_argument('--lengths', nargs='+', type=int, default=[2048, 4096, 8192, 16384])
    p.add_argument('--fixed-length', type=int, default=2048,
                   help='length held fixed during the batch-size sweep')
    p.add_argument('--fixed-batch-size', type=int, default=8,
                   help='batch size held fixed during the length sweep')
    p.add_argument('--sweep', choices=['both', 'batch', 'length'], default='both')
    p.add_argument('--modes', nargs='+', choices=['prefill', 'decode'], default=['prefill', 'decode'])
    p.add_argument('--variants', nargs='+', choices=['standard', 'ma_gpu', 'ma_offload'],
                   default=['standard', 'ma_gpu', 'ma_offload'])
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--logits-to-keep', type=int, default=0)
    p.add_argument('--resume', action='store_true', help='skip valid successful jobs with identical configuration/code')
    p.add_argument('--dry-run', action='store_true', help='write plan without running benchmarks')
    p.add_argument('--plot-only', action='store_true', help='plot existing summary.json without running benchmarks')
    p.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[],
                   help='additional bmk arguments; put this option last')
    args = p.parse_args()
    if args.plot_only:
        output = args.output.resolve()
        plot_results(output, json.loads((output / 'summary.json').read_text(encoding='utf-8')))
        return 0
    if any(n <= 0 for n in args.batch_sizes + args.lengths + [args.fixed_length, args.fixed_batch_size]) or args.repeats < 1 or args.rounds < 1:
        p.error('batch sizes, lengths, repeats and rounds must be positive')
    if args.warmup < 0 or args.logits_to_keep < 0:
        p.error('warmup/logits-to-keep must be nonnegative')
    reserved = {'--mode', '--variants', '--batch-size', '--seq-len', '--context-len',
                '--json', '--warmup', '--repeats', '--rounds', '--logits-to-keep',
                '--profile-dir', '--kernel-trace'}
    # Reject abbreviations as argparse in bmk.py also accepts these.
    for arg in args.extra_args:
        name = arg.split('=', 1)[0]
        if name.startswith('--') and any(x.startswith(name) for x in reserved):
            p.error(f'{name} is controlled by the sweep or excluded from timing runs')
    bmk = args.bmk.resolve()
    if not bmk.is_file():
        p.error(f'Benchmark script not found: {bmk}')
    output = args.output.resolve()
    (output / 'runs').mkdir(parents=True, exist_ok=True)
    script_hashes = {bmk.name: hashlib.sha256(bmk.read_bytes()).hexdigest()}
    helper = bmk.with_name('ma_profile.py')
    if helper.exists():
        script_hashes[helper.name] = hashlib.sha256(helper.read_bytes()).hexdigest()
    configurations = {}
    if args.sweep in ('both', 'batch'):
        for batch in dict.fromkeys(args.batch_sizes):
            configurations.setdefault((batch, args.fixed_length), []).append('batch')
    if args.sweep in ('both', 'length'):
        for length in dict.fromkeys(args.lengths):
            configurations.setdefault((args.fixed_batch_size, length), []).append('length')
    jobs = []
    for (batch, length), axes in configurations.items():
        for mode in dict.fromkeys(args.modes):
            for variant in dict.fromkeys(args.variants):
                key = f'bs{batch}_len{length}_{mode}_{variant}'
                result = output / 'runs' / f'{key}.json'
                command = [sys.executable, '-u', str(bmk), '--mode', mode,
                           '--variants', variant, '--batch-size', str(batch),
                           '--seq-len', str(length), '--context-len', str(length),
                           '--warmup', str(args.warmup), '--repeats', str(args.repeats),
                           '--rounds', str(args.rounds), '--logits-to-keep', str(args.logits_to_keep),
                           '--json', str(result), *args.extra_args]
                signature = hashlib.sha256(json.dumps(
                    [command, script_hashes], sort_keys=True).encode()).hexdigest()
                jobs.append(dict(key=key, batch_size=batch, length=length, sweep_axes=','.join(axes),
                                 mode=mode, variant=variant, command=command,
                                 signature=signature, result=str(result),
                                 log=str(result.with_suffix('.log'))))
    write_json(output / 'plan.json', dict(script_hashes=script_hashes, jobs=jobs))
    print(f'{len(jobs)} isolated runs; output: {output}', flush=True)
    print('Length = prefill sequence length AND decode prefix length; logits-to-keep=' + str(args.logits_to_keep), flush=True)
    if args.dry_run:
        return
    records = []
    interrupted = False
    for number, job in enumerate(jobs, 1):
        checkpoint = output / 'runs' / (job['key'] + '.status.json')
        result_path = Path(job['result'])
        if args.resume and checkpoint.exists():
            try:
                previous = json.loads(checkpoint.read_text(encoding='utf-8'))
                if previous['signature'] == job['signature'] and previous['status'] == 'ok':
                    previous['metrics'] = read_result(result_path, job)
                    previous['sweep_axes'] = job['sweep_axes']
                    records.append(previous)
                    summarize(output, records)
                    print(f'[{number}/{len(jobs)}] SKIP {job["key"]}', flush=True)
                    continue
            except (ValueError, KeyError, OSError, TypeError):
                pass
        # Remove only this generated job's stale result before retrying.
        result_path.unlink(missing_ok=True)
        record = {**job, 'status': 'running', 'metrics': {}}
        write_json(checkpoint, record)
        print(f'[{number}/{len(jobs)}] RUN {job["key"]}', flush=True)
        begin = time.perf_counter()
        tail = deque(maxlen=80)
        saw_oom = False
        process = None
        exit_code = None
        try:
            with Path(job['log']).open('w', encoding='utf-8') as log:
                log.write('COMMAND (argument list): ' + json.dumps(job['command']) + '\n')
                process = subprocess.Popen(job['command'], stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True,
                                           encoding='utf-8', errors='replace')
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    tail.append(line.rstrip())
                    lowered = line.lower()
                    saw_oom |= any(x in lowered for x in ['out of memory', 'outofmemoryerror', 'cannot allocate memory'])
                exit_code = process.wait()
            if exit_code == 0:
                record['metrics'] = read_result(result_path, job)
                record['status'] = 'ok'
            else:
                record['status'] = 'oom' if saw_oom else 'failed'
                record['error'] = '\n'.join(tail)[-5000:]
        except KeyboardInterrupt:
            interrupted = True
            record['status'] = 'interrupted'
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    exit_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record['status'] = 'failed'
            record['error'] = repr(exc)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            record['wall_seconds'] = time.perf_counter() - begin
            record['exit_code'] = exit_code
            write_json(checkpoint, record)
            records.append(record)
            summarize(output, records)
        metric = f" {record['metrics']['median_ms']:.3f} ms" if record['status'] == 'ok' else ''
        print(f'  {record["status"].upper()}{metric} ({record["wall_seconds"]:.1f}s)', flush=True)
        if record.get('error'):
            print('\n'.join(record['error'].splitlines()[-3:]), flush=True)
        if record['status'] == 'failed' and 'ModuleNotFoundError:' in record.get('error', ''):
            print('Missing Python module: fix the dependency, then rerun with --resume. Remaining jobs were not run.', flush=True)
            break
        if interrupted:
            print('Stopped. Run the same command with --resume to continue.', flush=True)
            break
    counts = {s: sum(r['status'] == s for r in records) for s in ['ok', 'oom', 'failed', 'interrupted']}
    print(json.dumps(counts) + '\nSummary: ' + str(output / 'summary.csv'), flush=True)
    plot_results(output, records)
    return 130 if interrupted else (1 if counts['failed'] or counts['oom'] else 0)


if __name__ == '__main__':
    sys.exit(main())
