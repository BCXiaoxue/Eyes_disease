import unittest

from utils.rag import build_context, explain_query, rag_status, rebuild_index, retrieve


class RagTests(unittest.TestCase):
    def test_clinical_source_ranking(self):
        cases = [
            ("糖尿病视网膜病变 糖网 黄斑水肿", "diabetic_retinopathy.md"),
            ("青光眼 眼压升高 视野缺损 RNFL OCT", "glaucoma.md"),
            ("白内障 眩光 视物模糊 裂隙灯", "cataract.md"),
        ]
        for query, expected_source in cases:
            with self.subTest(query=query):
                results = retrieve(query, n_results=3)
                self.assertTrue(results)
                self.assertEqual(results[0]["source"], expected_source)
                self.assertEqual(len({result["source"] for result in results}), len(results))
                self.assertTrue(all(result["citation_id"] for result in results))

    def test_red_flag_forces_referral_source_first(self):
        results = retrieve("突然视力下降 大量飞蚊 闪光 幕布感", n_results=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "referral_standards.md")

    def test_structured_evidence_and_context(self):
        evidence = explain_query("青光眼 眼压升高", n_results=2)
        self.assertEqual(evidence["result_count"], 2)
        self.assertIn("glaucoma.md", evidence["sources"])
        self.assertEqual(len(evidence["citation_ids"]), 2)
        context = build_context("青光眼 眼压升高", n_results=2)
        self.assertIn("眼科知识库参考", context)
        self.assertIn("[R1]", context)
        self.assertIn("引用ID", context)

    def test_status_and_rebuild(self):
        self.assertTrue(rag_status()["available"])
        rebuilt = rebuild_index()
        self.assertTrue(rebuilt["ok"])
        self.assertGreaterEqual(rebuilt["chunks"], 8)


if __name__ == "__main__":
    unittest.main()
