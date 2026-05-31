"""Phase 6 tests: config, promotion, runner, monitoring end-to-end.
Run via: python -m unittest discover -s tests
"""

import os
import tempfile
import unittest

from alpha_pipeline.config import Config, load_yaml
from alpha_pipeline.monitor import health
from alpha_pipeline.journal import Journal


class TestConfig(unittest.TestCase):
    def test_parses_real_config(self):
        cfg = Config.load("config.yaml")
        self.assertEqual(cfg.section("risk")["max_drawdown"], 0.15)
        self.assertEqual(cfg.section("gate")["min_t_stat"], 3.0)
        self.assertTrue(len(cfg.section("universe")["names"]) >= 40)
        self.assertIsInstance(cfg.section("execution")["commission_bps"], float)

    def test_scalar_types(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "c.yaml")
        with open(path, "w") as f:
            f.write("a:\n  b: 3\n  c: 1.5\n  d: true\n  e: [x, y, z]\n  f: hello\n")
        d = load_yaml(path)
        self.assertEqual(d["a"]["b"], 3)
        self.assertEqual(d["a"]["c"], 1.5)
        self.assertIs(d["a"]["d"], True)
        self.assertEqual(d["a"]["e"], ["x", "y", "z"])
        self.assertEqual(d["a"]["f"], "hello")


class TestPaperTradingRun(unittest.TestCase):
    def test_demo_runs_and_trades(self):
        from run import run

        cfg = Config.load("config.yaml")
        workdir = tempfile.mkdtemp()
        report = run(cfg, days=400, seed=3, workdir=workdir)
        # The drift demo should promote momentum and trade many cycles.
        self.assertGreater(report.n_cycles, 100)
        self.assertFalse(report.halted)
        self.assertEqual(report.n_critical_events, 0)
        # Registry persisted to disk.
        self.assertTrue(os.path.exists(os.path.join(workdir, "registry.json")))

    def test_monitor_on_empty_journal(self):
        tmp = tempfile.mkdtemp()
        with Journal(os.path.join(tmp, "j.db")) as j:
            r = health(j)
            self.assertEqual(r.n_cycles, 0)
            self.assertEqual(r.nav, 0.0)


class TestPromoteScript(unittest.TestCase):
    def test_promote_via_gate(self):
        from scripts.promote_alpha import main

        tmp = tempfile.mkdtemp()
        reg_path = os.path.join(tmp, "reg.json")
        # momentum on the wide drift universe should pass and exit 0.
        code = main(["momentum", "--registry", reg_path, "--days", "400", "--seed", "3"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(reg_path))

    def test_unknown_alpha_rejected(self):
        from scripts.promote_alpha import main

        code = main(["does_not_exist"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
