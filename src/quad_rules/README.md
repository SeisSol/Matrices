# Some quadrature rules for line segments, triangles and tetrahedra

* The reference element in 1D is the line segment [-1, 1].
* The reference element in 2D is the triangle spaned by the nodes (-1, -1), (1, 0), (0, 1).
* The reference element in 3D is the tetrahedron spaned by the nodes (-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1).

Use as
```
nodes, weights = WitherdenVincentTri.WitherdenVincentTri().find_best_rule(5)
new_corners = np.array([[0, 0], [1, 0], [0, 1]])

def f(x):
    return x[0] * x[1] ** 2

nodes_, weights_ = transform(nodes, weights, new_corners)
result = quad(nodes_, weights_, f)
```

With some help from https://github.com/TEAR-ERC/tandem/tree/e7b245aef0e8c8fbf045afce22592b08f626e806/src/quadrules
