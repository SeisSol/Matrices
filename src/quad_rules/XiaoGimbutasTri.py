from quad_rules.TabulatedTri import TabulatedTri


class XiaoGimbutasTri(TabulatedTri):
    """Fully symmetric rules of Xiao and Gimbutas (2010), degrees 1 to 30."""

    def __init__(self):
        super().__init__("xiao-gimbutas")
