from abc import ABC


class QuadRule(ABC):
    def find_best_rule(self, order):
        if hasattr(self, "data"):
            available_orders = sorted(self.data.keys())
            best_order = min(i for i in available_orders if i >= order)
            return self.data[best_order]
        else:
            return self.compute_nodes_and_weights(order)
