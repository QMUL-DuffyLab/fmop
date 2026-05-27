# FMOP - Flux Models of Oxygenic Photosynthesis

This code uses [tellurium](https://tellurium.readthedocs.io/en/latest/index.html) to set up and simulate simple models of *possible* but not *extant* photosynthesis.
The point is to investigate whether oxygenic photosynthesis as it is found on Earth is a "better" (in the sense of how much carbon it can reduce and also in robustness to fluctuations in light) way of doing things than others we could imagine.

## Setup

Pretty simple - install tellurium as described in the docs above and then clone this repo.
I have just used the version of Spyder that is provided and it works fine so far except for Spyder's lack of vim bindings in the editor.
There's also a separate GUI `.ipynb` thing that they have set up; I assume if you use that you can just
```
import main

scheme = "7b"
results = main.run(scheme, Tstar)
```
in a notebook cell or something like that, but I haven't checked.

## Comments

The code itself is pretty short and hopefully self-explanatory, at least so far.
The main problem is setting up 14 different possible chemical reaction networks which share some similarities but differ in non-trivial ways, while also substituting in various parameters that should be settable by the user.
There are a common core of reactions, but there are a lot of variations and they all bolt together in different ways depending on the scheme so it's difficult to cut them up into properly reusable parts, and even if you could you'd still have to go through and make sure the species are explicitly connected to each other in each of the sub-reactions so you'd end up with a load of bespoke code per scheme anyway.
Additionally, because of how the underlying [antimony](https://github.com/sys-bio/antimony/) specification language works, we can't just bung the whole set in a file and read it back in, because we need an f-string to set the relevant parameters.
Basically the upshot is that I've ended up with a 500-line f-string in main.py that contains all the data for all the models, and then a wrapper function for the user to provide a scheme string and a stellar effective temperature and that just returns the results.
This is *really ugly and I hate it* but I have (not yet) figured out a better option.

## TO DO

- [ ] Check notebook setup and provide an example .ipynb for use in the tellurium viewer or jupyter or wherever
- [ ] Add notes on the allowed stellar temperatures and scheme strings to the `run()` docstring
- [ ] Figure out a better way of composing the component functions of different schemes to cut down on duplicated code
- [ ] See if it's possible to use python string `.format()` method to insert quantities at runtime instead of setting all the parameters in the file and then f-stringing them in
- [ ] Settle on where the `run()` wrapper abstraction should end - should it include the plots? Do the plots and return the filenames? What's the intended workflow here
