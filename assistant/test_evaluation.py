import unittest

from assistant.evaluation import parse_tool_call, score_probe


class EvaluationTests(unittest.TestCase):
    def test_tagged_tool_call_and_parameters(self):
        probe = {"kind": "tool", "expected_tool": "system.volume.set", "expected_parameters": {"level": 30}}
        output = '<tool_call>{"name":"system.volume.set","arguments":{"level":30}}</tool_call>'
        self.assertEqual(parse_tool_call(output)["name"], "system.volume.set")
        self.assertEqual(score_probe(output, probe, {"system.volume.set"})[0], True)

    def test_wrong_tool_is_not_a_pass(self):
        probe = {"kind": "tool", "expected_tool": "time.now", "expected_parameters": {}}
        ok, reason = score_probe('{"name":"date.now","arguments":{}}', probe, {"time.now", "date.now"})
        self.assertFalse(ok)
        self.assertEqual(reason, "wrong_tool")

    def test_chat_rejects_tool_calls(self):
        probe = {"kind": "chat", "chosen_text": "Hello!"}
        self.assertEqual(score_probe('<tool_call>{"name":"time.now"}</tool_call>', probe, {"time.now"})[1], "unexpected_tool")


if __name__ == "__main__":
    unittest.main()
