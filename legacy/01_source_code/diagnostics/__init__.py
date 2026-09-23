"""
Diagnostic instrumentation package for NeuroScan.

Everything in this package is OPTIONAL and OFF by default (see config.py).
Nothing here changes model architecture, loss functions, optimizer behavior,
or training schedule. When all flags in `config.py` are False (the default),
importing and using this package has zero effect on training outputs.

This package exists to help decide between two candidate research directions
(adaptive slice sampling vs. multi-objective loss optimization) by making the
existing system's internal behavior observable, not by changing that behavior.
"""
