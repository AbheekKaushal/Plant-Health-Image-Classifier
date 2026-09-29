"""Create an auditable, stratified leaf-group split; remove pixel duplicates."""
import argparse, hashlib, json, random
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageOps

def prepare(root, leaf_map, out, prefix, seed):
    mapping = json.loads(Path(leaf_map).read_text()) if leaf_map else {}
    groups = defaultdict(list); seen = {}; duplicates = 0; unmapped = 0
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    for label in classes:
        for p in sorted((root / label).iterdir()):
            if p.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
            with Image.open(p) as im:
                rgb = ImageOps.exif_transpose(im).convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label: raise ValueError('Identical image has conflicting labels')
                duplicates += 1; continue
            seen[digest] = label
            key = p.stem.split('___')[-1].split('copy')[0].strip().lower()
            options = mapping.get(key, [])
            matches = [v for v in options if v.startswith(label+':::')]
            if len(matches) == 1: group = matches[0]
            elif len(options) == 1: group = options[0]
            else:
                group = label+':::'+key; unmapped += 1
            groups[(label,group)].append({'path':str(p.relative_to(root)), 'label':label, 'group':group,'sha256':digest})
    rows=[]; rng=random.Random(seed)
    for label in classes:
        keys = sorted(k for k in groups if k[0]==label); rng.shuffle(keys)
        if len(keys)<10: raise ValueError(f'{label}: need at least 10 independent groups')
        n=len(keys); train_end=int(n*.70); val_end=int(n*.85)
        for i,key in enumerate(keys):
            split='train' if i<train_end else 'validation' if i<val_end else 'test'
            rows.extend(dict(r,split=split) for r in groups[key])
    result={'seed':seed,'classes':classes,'duplicates_removed':duplicates,'unmapped_images':unmapped,'rows':rows}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'images':len(rows),'splits':Counter(r['split'] for r in rows),'duplicates_removed':duplicates,'unmapped':unmapped}))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--leaf-map'); p.add_argument('--out',type=Path,default=Path('reports/manifest.json')); p.add_argument('--prefix',default='Tomato'); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); prepare(a.data,a.leaf_map,a.out,a.prefix,a.seed)
