from abc import ABC


class QuadRule(ABC):
    def find_best_rule(self, order):
        if hasattr(self, "data"):
            available_orders = sorted(self.data.keys())
            best_order = min(i for i in available_orders if i >= order)
            return self.data[best_order]
        else:
            return self.compute_nodes_and_weights(order)


if __name__ == "__main__":
    import quad_rules.Dunavant as Dunavant
    import quad_rules.WitherdenVincentTri as WitherdenVincentTri
    import matplotlib.pyplot as plt

    dunavant_points, dunavant_weights = Dunavant.Dunavant().find_best_rule(12)
    wv_points, wv_weights = WitherdenVincentTri.WitherdenVincentTri().find_best_rule(12)
    plt.scatter(
        dunavant_points[:, 0],
        dunavant_points[:, 1],
        s=100 * dunavant_weights,
        label="dunavant",
    )
    plt.scatter(
        wv_points[:, 0], wv_points[:, 1], s=100 * wv_weights, label="witherden vincent"
    )
    plt.plot([-1, 1, -1, -1], [-1, -1, 1, -1])
    plt.legend()
    plt.show()
