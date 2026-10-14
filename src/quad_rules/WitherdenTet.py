from quad_rules.TabulatedTet import TabulatedTet


class WitherdenTet(TabulatedTet):
    """Fully symmetric rules of Witherden (2023, from PyFR), degrees 10 to 20 except 14."""

    def __init__(self):
        super().__init__("witherden")
