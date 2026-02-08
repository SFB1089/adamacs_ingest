# Notebook Python Script Exports

`notebooks/py_scripts/` stores plain Python exports of notebooks for easier review and diffing.

The exported scripts are not the primary execution path; they are a review artifact.

## Refresh exports with Jupytext
From the repository root:

```bash
python -m pip install jupytext
jupytext --to py notebooks/*.ipynb
mv notebooks/*.py notebooks/py_scripts/
```

If you use the Jupytext Jupyter extension, notebook/script pairs can also be synced automatically on save.
