from quad_rules.TabulatedTet import TabulatedTet


class WitherdenVincentTet(TabulatedTet):
    """Fully symmetric rules of Witherden and Vincent (2015), degrees 1 to 10."""

    def __init__(self):
        super().__init__("witherden-vincent")
