"""Replot existing sweep summaries with consistent scales and styling; no CUDA needed."""
import argparse
import json
import math
from pathlib import Path


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_results(output, records):
    """One row of four panels: bsz prefill/decode, then length prefill/decode."""
    from html import escape
    from collections import Counter

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    normalized = []
    for r in records:
        if 'sweep_axes' in r:
            normalized.append(r)
        elif r.get('sweep') in ('bsz', 'batch', 'length'):
            axis = 'batch' if r['sweep'] in ('bsz', 'batch') else 'length'
            length = r['prefill_length'] if r['mode'] == 'prefill' else r['context_length']
            normalized.append(dict(
                sweep_axes=axis, batch_size=r['batch'], length=length,
                mode=r['mode'], variant=r['variant'], status=r['status'],
                metrics={'median_ms': r.get('latency_ms'),
                         'min_ms': r.get('latency_low_ms'),
                         'max_ms': r.get('latency_high_ms')},
            ))
        else:
            raise ValueError('Unrecognized summary format: expected sweep_axes or sweep=bsz/length')
    records = normalized
    styles = {'standard': ('#8055B5', 'Baseline', 'o'),
              'ma_gpu': ('#2878B5', 'MA', 's'),
              'ma_offload': ('#299768', 'MA Offload', '^')}

    def point(r):
        if r is None or r['status'] != 'ok':
            return None
        m = r.get('metrics', {})
        v = m.get('median_ms')
        if v is None or not math.isfinite(float(v)) or float(v) <= 0:
            return None
        mid = float(v)
        low, high = m.get('min_ms'), m.get('max_ms')
        low = float(low) if low is not None and math.isfinite(float(low)) else mid
        high = float(high) if high is not None and math.isfinite(float(high)) else mid
        return mid, min(mid, low), max(mid, high)

    # One scale specification for BOTH charts, computed from all loaded results.
    scales = {}
    for mode in ('prefill', 'decode'):
        values = [point(r) for r in records if r['mode'] == mode]
        peak = max([v[2] for v in values if v is not None] or [1])
        factor = 1000 if peak >= 1000 else 1
        raw_step = peak / factor * 1.1 / 4
        power = 10 ** math.floor(math.log10(raw_step))
        step = next(n*power for n in (1, 2, 2.5, 5, 10) if n*power >= raw_step)
        top = math.ceil(peak/factor*1.1/step)*step
        ticks = [i*step for i in range(round(top/step)+1)]
        scales[mode] = dict(factor=factor, unit='s' if factor == 1000 else 'ms',
                            top=top, ticks=ticks)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import NullLocator
    except ImportError:
        plt = None
    from html import escape
    W, H = 2200, 550
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#26313D}.muted{fill:#7A8591}</style>',
           '<text x="70" y="42" font-size="27" font-weight="600">Attention inference scaling</text>']
    if plt is not None:
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'pdf.fonttype': 42, 'svg.fonttype': 'none'})
        fig = plt.figure(figsize=(22, 5.5), facecolor='white')
        fig.text(70/W, 1-42/H, 'Attention inference scaling', fontsize=20, weight='semibold')
        handles = [Line2D([], [], color=c, label=l, marker=m, markerfacecolor=c,
                          markeredgewidth=1.5, linewidth=2) for c,l,m in styles.values()]
        fig.legend(handles=handles, loc='center right', bbox_to_anchor=(.97,.943),
                   frameon=False, ncol=3, handlelength=2.2, columnspacing=2)

    def marker(x, y, color, shape):
        attr = f'fill="{color}" stroke="{color}" stroke-width="1.2"'
        if shape == 'o': return f'<circle cx="{x}" cy="{y}" r="4" {attr}/>'
        if shape == 's': return f'<rect x="{x-4}" y="{y-4}" width="8" height="8" {attr}/>'
        return f'<path d="M{x} {y-5} L{x+5} {y+4} L{x-5} {y+4} Z" {attr}/>'

    for x, (color, label, shape) in zip([1650,1810,1970], styles.values()):
        svg.append(f'<path d="M{x} 37 h28" stroke="{color}" stroke-width="2.5"/>')
        svg.append(marker(x+14,37,color,shape))
        svg.append(f'<text x="{x+37}" y="42" font-size="14">{label}</text>')
    svg.append('<path d="M1100 94 V515" stroke="#E0E6EC"/>')
    if plt is not None:
        fig.add_artist(Line2D([.5,.5],[1-515/H,1-94/H],transform=fig.transFigure,color='#E0E6EC',linewidth=.8))
    caption_parts = []
    exclusions = []
    for group, axis in enumerate(('batch', 'length')):
        selected = [r for r in records if axis in r['sweep_axes'].split(',')]
        key = 'batch_size' if axis == 'batch' else 'length'
        fixed_key = 'length' if axis == 'batch' else 'batch_size'
        fixed = sorted({r[fixed_key] for r in selected})
        xs = sorted({r[key] for r in selected})
        if not xs: xs = [8,16,32,64] if axis == 'batch' else [2048,4096,8192,16384]
        if any(x<=0 for x in xs): raise ValueError('Sweep values must be positive')
        labels = [str(x) if axis=='batch' else f'{x//1024}k' if x%1024==0 else str(x) for x in xs]
        title = 'Batch size sweep' if axis=='batch' else 'Length sweep'
        subtitle = ('Fixed sequence / context length: '+', '.join(f'{x:,}' for x in fixed)+' tokens') if axis=='batch' else ('Fixed batch size: '+', '.join(map(str,fixed)))
        caption_parts.append(subtitle + '.')
        gx = 70+group*1100
        svg.append(f'<text x="{gx}" y="92" font-size="19" font-weight="600">{title}</text>')
        if plt is not None:
            fig.text(gx/W,1-92/H,title,fontsize=14,weight='semibold')
        for mi,mode in enumerate(('prefill','decode')):
            left,top,width,height = 80+group*1100+mi*535,155,440,290
            sc = scales[mode]
            logs=[math.log2(x) for x in xs]
            lo,hi=min(logs)-.15,max(logs)+.15
            if len(xs)==1:lo,hi=logs[0]-.5,logs[0]+.5
            px=lambda i:left+(logs[i]-lo)/(hi-lo)*width
            py=lambda v:top+height-v/sc['top']*height
            xlabel='Batch size (log₂)' if axis=='batch' else 'Length in tokens (log₂)'
            svg.append(f'<text x="{left}" y="130" font-size="17" font-weight="600">{mode.capitalize()}</text>')
            for tick in sc['ticks']:
                yy=py(tick)
                svg.append(f'<path d="M{left} {yy} h{width}" stroke="#E8EDF1"/>')
                svg.append(f'<text x="{left-12}" y="{yy+4}" text-anchor="end" font-size="12" class="muted">{tick:g}</text>')
            for i,label in enumerate(labels):
                svg.append(f'<text x="{px(i)}" y="470" text-anchor="middle" font-size="12" class="muted">{label}</text>')
            svg.append(f'<text x="{left+width/2}" y="498" text-anchor="middle" font-size="12">{xlabel}</text>')
            svg.append(f'<text transform="translate({left-50},{top+height/2}) rotate(-90)" text-anchor="middle" font-size="12">Latency ({sc["unit"]})</text>')
            if plt is not None:
                ax=fig.add_axes([left/W,1-(top+height)/H,width/W,height/H])
                ax.set_xscale('log',base=2);ax.set_xlim(2**lo,2**hi);ax.set_ylim(0,sc['top'])
                ax.set_xticks(xs,labels);ax.xaxis.set_minor_locator(NullLocator())
                ax.set_yticks(sc['ticks'],[f'{t:g}' for t in sc['ticks']])
                ax.tick_params(length=0,pad=8,colors='#74808B',labelsize=9)
                for spine in ax.spines.values():spine.set_visible(False)
                ax.grid(axis='y',color='#E8EDF1',linewidth=.7);ax.set_axisbelow(True)
                ax.set_xlabel(xlabel,labelpad=12,fontsize=9)
                ax.set_ylabel(f'Latency ({sc["unit"]})',labelpad=10,fontsize=9)
                fig.text(left/W,1-130/H,mode.capitalize(),fontsize=12,weight='semibold')
            found=False;failed=0
            for variant,(color,label,shape) in styles.items():
                lookup={}
                for r in selected:
                    if r['mode']==mode and r['variant']==variant:
                        if r[key] in lookup: raise ValueError(f'Duplicate measurement: {axis}/{mode}/{variant}/{r[key]}')
                        lookup[r[key]]=r
                pts=[]
                for x in xs:
                    r=lookup.get(x);pt=point(r)
                    pts.append(tuple(v/sc['factor'] for v in pt) if pt else None)
                    if r and pt is None:failed+=1
                prev=None
                for i,pt in enumerate(pts):
                    if pt is None:prev=None;continue
                    found=True;mid,low,high=pt;xx,yy=px(i),py(mid)
                    if prev:svg.append(f'<path d="M{prev[0]} {prev[1]} L{xx} {yy}" fill="none" stroke="{color}" stroke-width="2.5"/>')
                    svg.append(f'<path d="M{xx} {py(low)} V{py(high)} M{xx-3} {py(low)} h6 M{xx-3} {py(high)} h6" stroke="{color}" opacity=".5"/>')
                    svg.append(marker(xx,yy,color,shape));prev=xx,yy
                if plt is not None:
                    ax.errorbar(xs,[p[0] if p else float('nan') for p in pts],
                                yerr=[[p[0]-p[1] if p else 0 for p in pts],[p[2]-p[0] if p else 0 for p in pts]],
                                color=color,marker=shape,markerfacecolor=color,markeredgewidth=1.4,
                                linewidth=1.8,markersize=5,elinewidth=.7,capsize=2)
            if not found:
                svg.append(f'<text x="{left+width/2}" y="330" text-anchor="middle" font-size="14">No successful measurements</text>')
                if plt is not None:ax.text(.5,.5,'No successful measurements',transform=ax.transAxes,ha='center',fontsize=10)
            if failed:
                exclusions.append(f'{title}, {mode}: {failed} failed or invalid points omitted.')
    caption = (
        'Figure. Inference latency of Standard attention, GPU-resident MA, and offloaded MA. '
        'The left pair of panels varies batch size; the right pair varies sequence/context length. '
        'Within each pair, prefill is shown on the left and decode on the right. '
        + ' '.join(caption_parts) + ' '
        + f'Prefill latency is measured in {scales["prefill"]["unit"]}; decode latency in {scales["decode"]["unit"]}. '
        + 'Points show median latency; whiskers indicate the observed minimum and maximum across measurement rounds, not confidence intervals. '
        + 'The horizontal axes use a base-2 logarithmic scale; 1k denotes 1024 tokens. '
        + 'Panels for the same mode share vertical limits and units. Lower latency is better. '
        + ' '.join(exclusions)
    ).strip()
    (output/'sweep_combined_caption.txt').write_text(caption+'\n',encoding='utf-8')
    svg.append('</svg>')
    path=output/'sweep_combined.svg';path.write_text('\n'.join(svg),encoding='utf-8')
    if plt is not None:
        fig.savefig(output/'sweep_combined.png',dpi=220,facecolor='white')
        fig.savefig(output/'sweep_combined.pdf',facecolor='white')
        plt.close(fig)
    print(f'Saved {path}' + (' and PNG/PDF' if plt is not None else ' (install matplotlib for PNG/PDF)'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='One horizontal figure: bsz on the left, length on the right.')
    parser.add_argument('--inputs',nargs='+',type=Path,required=True,help='Result directories or summary.json files')
    parser.add_argument('--output',type=Path,default=Path('figures'))
    args=parser.parse_args();records=[]
    for path in args.inputs:
        path=path/'summary.json' if path.is_dir() else path
        records.extend(json.loads(path.read_text(encoding='utf-8')))
    plot_results(args.output,records)
