"""Print text outputs of the executed notebook (for review); flags errors/tracebacks and reports output sizes."""
import base64
import sys

import nbformat

nb = nbformat.read(sys.argv[1] if len(sys.argv) > 1 else 'dz_topic_8_Джафарлі_Фарід.ipynb', as_version=4)
MAX_TEXT = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
n_err = n_img = img_bytes = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    size = sum(len(str(o)) for o in c.get('outputs', []))
    print(f'\n===== cell {i} ({size / 1e3:.0f} kB of outputs) =====')
    print(c.source.splitlines()[0][:100] if c.source else '')
    print('-----')
    for o in c.get('outputs', []):
        if o.output_type == 'error':
            n_err += 1
            print('ERROR:', o.ename, o.evalue)
        elif o.output_type == 'stream':
            text = o.text
            if len(text) > MAX_TEXT:
                text = text[:MAX_TEXT // 2] + f'\n[... {len(text) - MAX_TEXT} chars ...]\n' + text[-MAX_TEXT // 2:]
            print(f'[{o.name}]', text)
        else:
            data = o.get('data', {})
            fmt = next((f for f in ('image/png', 'image/jpeg') if f in data), None)
            if fmt:
                n_img += 1
                size = len(base64.b64decode(data[fmt]))
                img_bytes += size
                print(f'[{fmt} {size / 1e6:.2f} MB]')
            elif 'text/plain' in data:
                print(data['text/plain'][:MAX_TEXT])
print(f'\nerrors: {n_err}, images: {n_img} ({img_bytes / 1e6:.1f} MB)')
