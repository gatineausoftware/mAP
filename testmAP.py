import unittest
from mAP import getThreshold
from BoundingBox import BoundingBox
import numpy as np



class mAPTest(unittest.TestCase):

    def setUp(self):
        self.bb1 = BoundingBox(0, 50, 50, 100)
        self.bb2 = BoundingBox(30, 50, 40, 80)

    def test1(self):
        a = np.full_like(np.arange(10, dtype=np.double), 0.1)
        self.assertEqual({0.5: 0.1, 1: 0.1}, getThreshold(a, 10, 2))

    def test2(self):
        a = np.asarray(range(1,101)) / 100.0
        b = np.asarray((range(1,11))) / 10.0
        c = {(v+1)/10.0: k for v, k in enumerate(b)}
        self.assertEquals(c, getThreshold(a, 100, 10))

    def testOverlapX(self):
        bb1 = BoundingBox(0, 50, 0, 50)
        bb2 = BoundingBox(30, 80, 0, 50)
        self.assertEquals(20, bb1.x_overlap(bb2))


    def testOverlapY1(self):
        bb1 = BoundingBox(0, 50, 0, 50)
        bb2 = BoundingBox(30, 80, 0, 50)
        self.assertEquals(50, bb1.y_overlap(bb2))


    def testOverlapY2(self):
        bb1 = BoundingBox(0, 50, 50, 100)
        bb2 = BoundingBox(30, 80, 90, 105)
        self.assertEquals(10, bb1.y_overlap(bb2))

    def testOverlapX2(self):
        self.assertEqual(20, self.bb1.x_overlap(self.bb2))

    def testOverlapY3(self):
        self.assertEqual(30, self.bb1.y_overlap(self.bb2))

    def testIntersection(self):
        self.assertEqual(600, self.bb1.intersection(self.bb2))

    def testUnion(self):
        self.assertEqual(2700, self.bb1.union(self.bb2))


    def testIOU(self):
        self.assertEqual((600./2700), self.bb1.iou(self.bb2))

if __name__ == '__main__':
    unittest.main()
