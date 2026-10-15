# FMOP - Flux Models of Oxygenic Photosynthesis

WARNING: this is some really, really ugly code!! Here be dragons, I apologise for the state of it, etc.

This code uses [tellurium](https://tellurium.readthedocs.io/en/latest/index.html) to set up and simulate simple models of *possible* but not *extant* photosynthesis.
The point is to investigate whether oxygenic photosynthesis as it is found on Earth is a "better" (in the sense of how much carbon it can reduce and also in robustness to fluctuations in light) way of doing things than others we could imagine.

## Setup

Pretty simple - install tellurium as described in the docs above and then clone this repo.
I have just used the version of Spyder that is provided and it works fine so far except for Spyder's lack of vim bindings in the editor.
They also provide a notebook-style frontend; I haven't used that but have included a simple `example.ipynb` which works fine in standard Jupyter.

## Comments

This repo actually contains two different but related python scripts which deal with a set of possible schemes for oxygenic photosynthesis (in `schemes.py`) and another separate script which considers the effect of input wavelength on a very simple cyclic photosystem (`limits.py`). Both are set up to just press play in the tellurium version of spyder that's provided, basically. lazy but effective enough.

The code itself is hopefully self-explanatory, at least so far.
Tellurium uses the [antimony](https://github.com/sys-bio/antimony/) specification language and I've defined the various photosynthetic schemes using that.
Both scripts use long python f-strings that insert the values of various parameters like reaction rates and stoichiometries as it goes; this might not be the most elegant way of doing things really, and we might not even need the flexibility of defining the parameters this way, but it works.

The big string in `schemes.py` is organised into four main blocks: first are a few functions which I've defined to allow for different light input scenarios.
Currently there are periodic (day/night cycle), switch (light is on at t = 0 and switches off at some time t > 0 that we can set), or constant illumination options.
These are set by the `params["light"]` dict member, which you can set, it will build the required function call and insert it into the model for you.
Secondly there are a few little models which I guess you can think of as submodels: certain components that are reused between schemes, like a maintenance reaction that consumes units of stored chemical energy, respiration which uses reduced carbon to produce units of chemical energy, etc.
These are defined first.
Then there are the schemes themselves, again with a comment block explaining how they work.
They contain the photochemical reactions and then call the submodels to complete the chemical reaction network.
Finally there's a section that sets all the variables and initialises the concentrations of all the possible species; the line `A: scheme_{scheme}(...)` is where your chosen scheme is inserted.
