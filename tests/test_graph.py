import unittest

from app import GRAPH_EDGES, GRAPH_NODES
from utils.model import LABELS


class GraphTests(unittest.TestCase):
    def test_every_model_label_has_connected_graph_node(self):
        node_by_label = {node.get("label_code"): node["id"] for node in GRAPH_NODES if node.get("label_code")}
        endpoints = {item for edge in GRAPH_EDGES for item in edge}
        self.assertLessEqual(set(LABELS), set(node_by_label))
        for label in LABELS:
            self.assertIn(node_by_label[label], endpoints)


if __name__ == "__main__":
    unittest.main()
