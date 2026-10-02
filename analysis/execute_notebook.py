"""Execute and checkpoint every cell, including figures and tables."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
os.environ['JUPYTER_PATH'] = str(root/'work/jupyter/share/jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root/'work/jupyter/runtime')
os.environ['IPYTHONDIR'] = str(root/'work/ipython')
os.environ['MPLCONFIGDIR'] = str(root/'work/matplotlib')
path = root/'notebooks/feature_engineering/s6e10_feature_analysis.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name='kaggle-analysis',
                        resources={'metadata': {'path': str(root)}})
try:
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                print(f'Executing cell {i}', flush=True)
                client.execute_cell(cell, i)
                nbformat.write(nb, path)
                print(f'Completed cell {i}', flush=True)
finally:
    nbformat.write(nb, path)
