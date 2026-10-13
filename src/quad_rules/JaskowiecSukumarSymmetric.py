from quad_rules.TabulatedTet import TabulatedTet


class JaskowiecSukumarSymmetric(TabulatedTet):
    """Fully symmetric rules of Jaskowiec and Sukumar (2021), degrees 1 to 20."""

    def __init__(self):
        super().__init__("jaskowiec-sukumar-symmetric")
