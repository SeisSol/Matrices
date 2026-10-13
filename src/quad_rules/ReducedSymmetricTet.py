from quad_rules.TabulatedTet import TabulatedTet


class ReducedSymmetricTet(TabulatedTet):
    """Fully symmetric rules with fewer points, found for SeisSol (degrees 12 and 14)."""

    def __init__(self):
        super().__init__("reduced-symmetric")
