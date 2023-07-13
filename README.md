# pysumoflux

Python implementation of the SUMOFLUX method for flux-ratio analysis.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
(sudo) pip install pysumoflux
```

I use `pytorch` quite a bit to test whether the jacobians I compute agree 
with what comes out of `torch.autograd`. Once sparse matrices are fully supported
by `torch`, I also plan to move all computation to the GPU. Depending on whether 
your machine has a GPU, the installation of `torch` is different. Please follow 
the instructions on [pytorch.org](https://pytorch.org/). 

## Usage

```python
import sumoflux
```

## Contributing
I know nothing about working in groups...

## License
[MIT](https://choosealicense.com/licenses/mit/)