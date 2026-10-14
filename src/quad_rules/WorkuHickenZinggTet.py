from quad_rules.TabulatedTet import TabulatedTet


class WorkuHickenZinggTet(TabulatedTet):
    """Fully symmetric rules of Worku, Hicken and Zingg (2024), via SummationByParts.jl (degrees 1 to 21)."""

    def __init__(self):
        super().__init__("worku-hicken-zingg")
