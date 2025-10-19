#!/usr/bin/env python3
import sys, struct, math

def unpack_rgb_float(f):
    try:
        u = struct.unpack('!I', struct.pack('!f', float(f)))[0]
    except Exception:
        return None
    r = (u >> 16) & 255
    g = (u >> 8) & 255
    b = u & 255
    return [r, g, b]

def parse_header(lines):
    hdr, meta = [], {}
    for i, ln in enumerate(lines):
        s = ln.strip()
        hdr.append(ln)
        if s.startswith('FIELDS'): meta['FIELDS'] = s.split()[1:]
        elif s.startswith('SIZE'): meta['SIZE'] = [int(x) for x in s.split()[1:]]
        elif s.startswith('TYPE'): meta['TYPE'] = s.split()[1:]
        elif s.startswith('COUNT'): meta['COUNT'] = [int(x) for x in s.split()[1:]]
        elif s.upper().startswith('DATA ASCII'):
            meta['DATA_LINE'] = i
            return meta, hdr
    raise RuntimeError('Not ASCII PCD')

def rewrite_header(hdr, meta):
    def row(tag, arr): return f"{tag} " + " ".join(str(x) for x in arr) + "\n"
    out = []
    for ln in hdr:
        s = ln.strip()
        if   s.startswith('FIELDS'): out.append(row('FIELDS', meta['FIELDS']))
        elif s.startswith('SIZE'):   out.append(row('SIZE', meta['SIZE']))
        elif s.startswith('TYPE'):   out.append(row('TYPE', meta['TYPE']))
        elif s.startswith('COUNT'):  out.append(row('COUNT', meta['COUNT']))
        else:                        out.append(ln if ln.endswith("\n") else ln + "\n")
    return out

def main(inp, outp):
    with open(inp, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().splitlines()

    meta, hdr = parse_header(lines)
    data_start = meta['DATA_LINE'] + 1
    rows = [ln for ln in lines[data_start:] if ln.strip()]

    fields = meta['FIELDS'][:]
    ncols = len(fields)
    col = {f:i for i,f in enumerate(fields)}
    has_rgb = 'rgb' in col
    has_triplet = all(f in col for f in ('r','g','b'))

    # Decide target color representation: explicit r,g,b
    if not has_triplet:
        # Append r,g,b at the end; we’ll fill values row-wise
        fields += ['r','g','b']
        meta['FIELDS'] = fields
        if 'SIZE' in meta:  meta['SIZE']  += [1,1,1]
        if 'TYPE' in meta:  meta['TYPE']  += ['U','U','U']
        if 'COUNT' in meta: meta['COUNT'] += [1,1,1]
        had_triplet_before = False
    else:
        had_triplet_before = True

    # If rgb existed, remember its index and we’ll drop it after filling r,g,b
    idx_rgb = col.get('rgb', None)

    # Recompute col map for output (after header change)
    out_col = {f:i for i,f in enumerate(fields)}
    idx_r, idx_g, idx_b = out_col['r'], out_col['g'], out_col['b']

    out_rows = []
    for ln in rows:
        vals = ln.split()
        # pad to at least original number of cols
        if len(vals) < ncols:
            vals += ['0'] * (ncols - len(vals))

        # derive r,g,b
        rgb = None
        if has_triplet and had_triplet_before:
            # use existing triplet
            def f2(x):
                try: return float(x)
                except: return float('nan')
            r0 = f2(vals[col['r']]); g0 = f2(vals[col['g']]); b0 = f2(vals[col['b']])
            if not math.isfinite(r0) and not math.isfinite(g0) and not math.isfinite(b0):
                rgb = [255,255,255]
            elif abs(r0)<1e-9 and abs(g0)<1e-9 and abs(b0)<1e-9:
                rgb = [255,255,255]
            else:
                rgb = [int(r0), int(g0), int(b0)]
        elif has_rgb and idx_rgb is not None and idx_rgb < len(vals):
            unpacked = unpack_rgb_float(vals[idx_rgb])
            rgb = unpacked if unpacked else [255,255,255]
            if rgb == [0,0,0]:
                rgb = [255,255,255]
        else:
            rgb = [255,255,255]

        # extend vals if needed to have r,g,b positions
        if len(vals) < len(fields):
            vals += ['0'] * (len(fields) - len(vals))
        vals[idx_r], vals[idx_g], vals[idx_b] = map(str, rgb)

        # optionally remove packed rgb field by zeroing it (safer to retain columns count)
        if idx_rgb is not None and idx_rgb < len(vals):
            vals[idx_rgb] = '0'

        out_rows.append(' '.join(vals))

    # If you want to remove 'rgb' entirely from header, uncomment:
    # if 'rgb' in meta['FIELDS']:
    #     j = meta['FIELDS'].index('rgb')
    #     for key in ('FIELDS','SIZE','TYPE','COUNT'):
    #         meta[key].pop(j)
    #     # and remove column j from each row (more work)

    new_hdr = rewrite_header(hdr, meta)
    with open(outp, 'w', encoding='utf-8') as f:
        f.writelines(new_hdr)
        f.write('DATA ascii\n')
        f.write('\n'.join(out_rows))
        f.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: fix_colors_pcd.py input_ascii.pcd output_ascii.pcd", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])

