from quad_rules.TabulatedTri import TabulatedTri


class WitherdenVincentTri(TabulatedTri):
    """Fully symmetric rules of Witherden and Vincent (2015), degrees 1 to 20 (without 3)."""

    def __init__(self):
        super().__init__("witherden-vincent")
