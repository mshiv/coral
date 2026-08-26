"""Provenance parsing must retain the simulation's working directory."""
from pathlib import Path
import tempfile
import unittest
from coral.analysis.chapter_figure_bundle import par_inputs


class ParameterInputsTest(unittest.TestCase):
    def test_symlinked_par_resolves_fields_in_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template, run = root/'template', root/'run'
            template.mkdir(); run.mkdir()
            (template/'case.par').write_text('DEMfile terrain.asc\nsim_time 172800\n')
            (template/'terrain.asc').write_text('wrong source')
            (run/'terrain.asc').write_text('actual staged source')
            (run/'case.par').symlink_to(template/'case.par')
            fields, record = par_inputs(run/'case.par')
            self.assertEqual(Path(fields['demfile']), run/'terrain.asc')
            self.assertEqual(record['files']['demfile']['bytes'],20)

    def test_duplicate_and_missing_fields_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'case.par'
            p.write_text('DEMfile a\nDEMfile b\n')
            with self.assertRaises(ValueError):
                par_inputs(p)
            p.write_text('DEMfile missing.asc\n')
            with self.assertRaises(FileNotFoundError):
                par_inputs(p)


if __name__ == '__main__':
    unittest.main()
