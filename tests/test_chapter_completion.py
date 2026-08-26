import tempfile
import unittest
import json
from pathlib import Path
import numpy as np
from coral.viz.fig_archived_forcing import boundary_series, hvar_points
from coral.analysis.chapter_physics_capsule import frame_sequence, same_grid
from coral.analysis.chapter_figure_bundle import par_inputs


class CompletionTests(unittest.TestCase):
    def test_hvar_only_and_streamed_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            bci=Path(tmp)/'case.bci'
            bci.write_text('P -81 32 HVAR bc1\nP -81 33 QVAR river\n')
            self.assertEqual(hvar_points(bci),[('bc1',-81.,32.)])
            bdy=Path(tmp)/'case.bdy'
            bdy.write_text('Comment\n\n\nbc1\n2 seconds\n1 0\n2 3600\n\nriver\n2 seconds\n3 0\n4 3600\n')
            data=boundary_series(bdy,{'bc1'})
            np.testing.assert_array_equal(data['bc1'],[[1,0],[2,3600]])
            with self.assertRaises(ValueError):
                boundary_series(bdy,{'missing'})
            bdy.write_text('Comment\nbc1\n2 seconds\n1 0\n')
            with self.assertRaises(ValueError):
                boundary_series(bdy,{'bc1'})

    def test_frame_schedule_and_bad_archive(self):
        paths=[Path(f'run-{i:04d}.wd') for i in range(97)]
        ordered,times=frame_sequence(reversed(paths),86400,259200,1800)
        self.assertEqual(ordered,paths)
        np.testing.assert_equal(times[[0,-1]],[86400,259200])
        with self.assertRaises(ValueError):
            frame_sequence(paths[:-1],86400,259200,1800)
        with self.assertRaises(ValueError):
            frame_sequence(paths,86400,259200,999999)

    def test_numeric_not_text_header_comparison(self):
        h=dict(nrows=10,ncols=20,xllcorner=-81.22,yllcorner=31.7,cellsize=.0003)
        same_grid(h,{**h,'xllcorner':-81.21999999999999})
        with self.assertRaises(ValueError):
            same_grid(h,{**h,'ncols':21})

    def test_collect_and_plot_small_baseline(self):
        from coral.analysis.chapter_physics_capsule import collect, plot
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/'baseline'; run.mkdir()
            results=run/'results'; results.mkdir()
            header='ncols 2\nnrows 2\nxllcorner -81\nyllcorner 31\ncellsize .01\nNODATA_value -9999\n'
            (run/'dem.asc').write_text(header+'2 3\n-1 1\n')
            par=run/'case.par'
            par.write_text('DEMfile dem.asc\ntstart 0\nsim_time 2\nsaveint 1\ndirroot results\nresroot test\n')
            (results/'test.max').write_text(header+'1 2\n3 4\n')
            for i in range(3):
                (results/f'test-{i:04d}.wd').write_text(header+f'{i} 0\n0 0\n')
            _, info=par_inputs(par)
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'runs':{'30m':info}}))
            out=root/'export'; collect(manifest,root,out)
            with np.load(out/'baseline_event.npz') as data:
                np.testing.assert_array_equal(data['wet_land_cells'],[0,1,1])
                self.assertEqual(data['frames'].shape,(3,2,2))
            plot(out/'baseline_event.npz',root/'event.png')
            self.assertTrue((root/'event.pdf').is_file())


if __name__=='__main__':
    unittest.main()
