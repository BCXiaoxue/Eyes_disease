import unittest

from utils.consult import build_consult_messages


class ConsultTests(unittest.TestCase):
    def test_build_messages_keeps_system_context_and_recent_rounds(self):
        history = []
        for index in range(8):
            history.extend(
                [
                    {"role": "user", "content": f"问题 {index}"},
                    {"role": "assistant", "content": f"回答 {index}"},
                ]
            )
        messages = build_consult_messages(history, "病例上下文", "[R1] 检索材料", max_rounds=6)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "system")
        self.assertIn("病例上下文", messages[1]["content"])
        self.assertEqual(len(messages[2:]), 12)
        self.assertNotIn("问题 0", str(messages))
        self.assertIn("问题 7", str(messages))

if __name__ == "__main__":
    unittest.main()
