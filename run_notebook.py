"""Execute a notebook in place with nbclient (like `jupyter nbconvert --execute --inplace`), but with
`coalesce_streams=True`: consecutive stream outputs are merged and carriage returns of progress bars are
applied, so only the final state of every Ultralytics progress bar is saved.

Usage (from this folder):  ../.venv/bin/python run_notebook.py dz_topic_8_Джафарлі_Фарід.ipynb
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name='python3', coalesce_streams=True,
                        resources={'metadata': {'path': str(path.parent)}})
t0 = time.time()
try:
    client.execute()
finally:
    nbformat.write(nb, path)  # keep outputs even if a cell fails
    print(f'{path.name}: executed in {(time.time() - t0) / 60:.1f} min')
