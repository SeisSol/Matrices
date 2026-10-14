from quad_rules.TabulatedTet import TabulatedTet


class OrbitSearchTet(TabulatedTet):
    """Fully symmetric rules found for SeisSol by a search over orbit structures (degree 10)."""

    def __init__(self):
        super().__init__("orbit-search")
