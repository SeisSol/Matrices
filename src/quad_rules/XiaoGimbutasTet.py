from quad_rules.TabulatedTet import TabulatedTet


class XiaoGimbutasTet(TabulatedTet):
    """Rules of Xiao and Gimbutas (2010), degrees 1 to 15; mostly not symmetric."""

    def __init__(self):
        super().__init__("xiao-gimbutas")
