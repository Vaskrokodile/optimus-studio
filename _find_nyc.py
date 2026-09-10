import json, os, hashlib
base = r'E:\iloptimus-home\learning'
seen = {}
for sid in sorted(os.listdir(base)):
    sf = os.path.join(base, sid, 'session.json')
    if not os.path.exists(sf):
        continue
    try:
        d = json.load(open(sf, encoding='utf-8'))
    except Exception:
        continue
    q = d.get('query', '')
    if 'new york' in q.lower() or 'nyc' in q.lower() or 'cityscape' in q.lower() or 'skyscraper' in q.lower():
        for kind in ['baseline', 'adapted', 'framework']:
            key = f'{kind}_artifact_path'
            bp = d.get(key, '')
            if bp and os.path.exists(bp):
                data = open(bp, 'rb').read()
                h = hashlib.md5(data).hexdigest()[:8]
                if h not in seen:
                    seen[h] = (sid, kind, len(data), bp)
                    print(f'NEW {kind}: sid={sid}, md5={h}, sz={len(data)}, path={bp}')
                else:
                    print(f'DUP {kind}: sid={sid}, md5={h} (same as {seen[h][0]})')

# Also check for any index.html in adapted/framework dirs not in session.json
print('\n--- Checking for HTML files not in session.json ---')
for sid in sorted(os.listdir(base)):
    sf = os.path.join(base, sid, 'session.json')
    if not os.path.exists(sf):
        continue
    try:
        d = json.load(open(sf, encoding='utf-8'))
    except Exception:
        continue
    q = d.get('query', '')
    if 'new york' in q.lower() or 'nyc' in q.lower() or 'cityscape' in q.lower() or 'skyscraper' in q.lower():
        for sub in ['adapted', 'framework']:
            d2 = os.path.join(base, sid, sub)
            if os.path.isdir(d2):
                for f in os.listdir(d2):
                    if f.endswith('.html'):
                        p = os.path.join(d2, f)
                        data = open(p, 'rb').read()
                        h = hashlib.md5(data).hexdigest()[:8]
                        if h not in seen:
                            seen[h] = (sid, sub, len(data), p)
                            print(f'NEW {sub}: sid={sid}, md5={h}, sz={len(data)}, file={f}')
