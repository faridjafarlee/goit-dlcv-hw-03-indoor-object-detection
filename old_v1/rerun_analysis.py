"""Re-execute all cells of the (re)built notebook EXCEPT the long training cell, whose saved outputs are copied from
an already executed notebook. Used after editing analysis / markdown cells, so that the 1-hour training does not have
to be repeated. The training cell's source must be identical in both notebooks; the analysis cells then work with the
artefacts that this training produced (runs/yolov9s_indoor/, weights/best.pt).

Usage (from this folder):
    ../.venv/bin/python build_notebook.py --out new.ipynb
    ../.venv/bin/python rerun_analysis.py new.ipynb dz_topic_8_Джафарлі_Фарід.ipynb   # 2nd = executed notebook
    mv new.ipynb dz_topic_8_Джафарлі_Фарід.ipynb
"""
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

TRAIN_MARK = "print(f'Модель: {CFG.BASE_MODEL}.pt"

path = Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
old = nbformat.read(sys.argv[2], as_version=4)

old_train = next(c for c in old.cells if c.cell_type == 'code' and c.source.startswith(TRAIN_MARK))
idx = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and c.source.startswith(TRAIN_MARK))
assert nb.cells[idx].source == old_train.source, 'training cell source changed -> the whole notebook must be re-run'

log = ''.join(o.get('text', '') for o in old_train.outputs if o.output_type == 'stream')
minutes = float(re.search(r'Час навчання: ([0-9.]+) хв', log).group(1))
restore = f"""RUN_DIR = CFG.PROJECT / CFG.NAME
LAST_PT = RUN_DIR / 'weights' / 'last.pt'
BEST_WEIGHTS = CFG.WEIGHTS_DIR / 'best.pt'
TRAIN_MINUTES = {minutes}
restarts = []
assert BEST_WEIGHTS.exists() and (RUN_DIR / 'results.csv').exists()
"""

client = NotebookClient(nb, timeout=None, kernel_name='python3', coalesce_streams=True,
                        resources={'metadata': {'path': str(path.parent)}})
t0 = time.time()
try:
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            count = client.code_cells_executed + 1
            if i == idx:                      # restore the training state instead of training again
                src = cell.source
                cell.source = restore
                client.execute_cell(cell, i, execution_count=count)
                cell.source = src
                cell.outputs = old_train.outputs
            else:
                client.execute_cell(cell, i, execution_count=count)
finally:
    nbformat.write(nb, path)
    print(f'{path.name}: analysis cells re-executed in {(time.time() - t0) / 60:.1f} min')
