"""Plot-only changes must preserve raster support and the meaning of zero responses."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from coral.viz.fig_model_inputs import mask_to_dem, display_extent
from coral.analysis.paired_chapter_summary import retained_fraction
from coral.analysis.chapter_figure_bundle import file_record, previous_inputs
from coral.analysis.replot_chapter_pairs import compare_metrics


class FigureRevisionTests(unittest.TestCase):
    def test_active_mask_does_not_fill_or_mutate(self):
        dem = np.array([[1., np.nan], [-9999., 2.]])
        field = np.array([[.1, .2], [.3, .4]])
        result = mask_to_dem(field, dem)
        np.testing.assert_allclose(result, [[.1, np.nan], [np.nan, .4]], equal_nan=True)
        self.assertEqual(field[0,1], .2)
        with self.assertRaises(ValueError):
            mask_to_dem(field[:, :1], dem)

    def test_display_crop_is_explicit_and_does_not_change_header(self):
        h = dict(ncols=10, nrows=20, xllcorner=-81., yllcorner=31., cellsize=.01)
        self.assertEqual(display_extent(h), [-81., -80.9, 31., 31.2])
        self.assertEqual(display_extent(h,31.1), [-81., -80.9, 31.1, 31.2])
        self.assertEqual(h['nrows'],20)
        with self.assertRaises(ValueError):
            display_extent(h,32.)

    def test_zero_integral_is_undefined_not_flat_zero(self):
        self.assertTrue(np.isnan(retained_fraction([0,0,0])).all())
        np.testing.assert_allclose(retained_fraction([10,5,0]), [1,.5,0])
        with self.assertRaises(ValueError):
            retained_fraction([-1,0])

    def test_rebuild_checks_previous_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            par=root/'case.par'; par.write_text('DEMfile terrain.asc\n')
            dem=root/'terrain.asc'; dem.write_text('saved terrain')
            manifest=root/'bundle.json'
            manifest.write_text(json.dumps(dict(runs={'4m':dict(files={
                'par':file_record(par),'demfile':file_record(dem)})})))
            self.assertEqual(previous_inputs(manifest)['par4'],par)
            dem.write_text('changed terrain')
            with self.assertRaises(ValueError):
                previous_inputs(manifest)

    def test_replot_rejects_changed_metrics(self):
        old=dict(wet_cells=100,footprint_cells=10,improved_cells=7,worsened_cells=3,
                 benefit_m3=2.5,adverse_m3=.3)
        compare_metrics(old,dict(old))
        with self.assertRaises(ValueError):
            compare_metrics(old,{**old,'benefit_m3':25.})

    def test_input_figure_outputs_common_mask_metadata(self):
        from coral.viz.fig_model_inputs import build
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            header='ncols 3\nnrows 2\nxllcorner -81\nyllcorner 31\ncellsize .01\nNODATA_value -9999\n'
            dem=root/'dem.asc'; dem.write_text(header+'1 2 -9999\n3 4 5\n')
            field=root/'n.asc'; field.write_text(header+'.1 .2 .3\n.4 .5 .6\n')
            out=root/'inputs.png'
            build(dem,out,manning=field,infil=field,infilcap=field,publication=True)
            report=json.loads(out.with_suffix('.json').read_text())
            self.assertEqual(report['inactive_dem_cells'],1)
            self.assertEqual(report['panel_statistics'][1]['valid_fraction_of_active_dem'],1)
            self.assertFalse(report['display_only_crop'])
            self.assertTrue(out.with_suffix('.pdf').is_file())

    def test_compact_emulator_uses_recorded_population(self):
        from coral.analysis.emulator_holdout_composite import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); export=root/'fields.npz'; out=root/'composite.png'
            meta=dict(members=[dict(selection='worst',rmse_m=.1,name='test-member')],
                      holdout_rmse_m=[.01,.02,.1])
            np.savez_compressed(export,worst_error=np.array([[.01,-.02],[.1,np.nan]]),
                                metadata_json=np.asarray(json.dumps(meta)))
            with patch('sys.argv',['plot','--case','Test='+str(export),'--out',str(out)]):
                main()
            self.assertTrue(out.with_suffix('.pdf').is_file())
            result=json.loads(out.with_suffix('.json').read_text())
            self.assertEqual(result['cases'][0]['source']['holdout_rmse_m'],[.01,.02,.1])


if __name__=='__main__':
    unittest.main()
