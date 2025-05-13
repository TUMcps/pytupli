---
myst:
  html_meta:
    "description": "Quick start guide for installing and setting up PyTupli"
    "keywords": "PyTupli, installation, quickstart, poetry, pip"
---

# Quick-start

## Installing PyTupli

You can install PyTupli using pip:

```{code-block} bash
pip install pytupli
```

Or if you're using Poetry:

```{code-block} bash
poetry add pytupli
```

For local development in editable mode,
navigate to the package directory and run:

```{code-block} bash
poetry install
```

### Optional Dependencies

PyTupli has several optional dependency groups that can be installed based on your needs:

- **Server Components**: To install dependencies for running the PyTupli server:
  ```{code-block} bash
  poetry install --with server
  ```

- **Documentation**: To build the documentation:
  ```{code-block} bash
  poetry install --with docs
  ```

- **Testing**: To run tests:
  ```{code-block} bash
  poetry install --with tests
  ```

You can combine multiple groups:
```{code-block} bash
poetry install --with server,docs,tests
```
