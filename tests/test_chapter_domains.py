from pathlib import Path
import tempfile
import unittest
from coral.viz.fig_chapter_domains import bounds, geoclaw_bounds, track_points, coastline_lines


class DomainSourcesTest(unittest.TestCase):
    def test_bounds_from_inputs(self):
        self.assertEqual(bounds(dict(xllcorner=-81,yllcorner=31,ncols=4,nrows=5,cellsize=.1)),
                         [-81,-80.6,31,31.5])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'claw.data'
            p.write_text('-90 10 =: lower\n-60 50 =: upper\n')
            self.assertEqual(geoclaw_bounds(p),[-90,-60,10,50])

    def test_matthew_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'track.dat'
            p.write_text('AL, 14, 2016100800, 00, BEST, 0, 310N, 800W\n'
                         'AL, 14, 2016100806, 00, BEST, 0, 320N, 790W\n')
            self.assertEqual(track_points(p).tolist(),[[-80,31],[-79,32]])
            p.write_text('AL, 05, 2019090400, 00, BEST, 0, 310N, 800W\n')
            with self.assertRaises(ValueError):
                track_points(p)

    def test_local_coastline(self):
        p=Path.home()/'.local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp'
        if not p.is_file():
            self.skipTest('optional Natural Earth cache not present')
        line=next(coastline_lines(p))
        self.assertEqual(line.shape[1],2)
        self.assertGreater(line.shape[0],1)


if __name__=='__main__':
    unittest.main()
