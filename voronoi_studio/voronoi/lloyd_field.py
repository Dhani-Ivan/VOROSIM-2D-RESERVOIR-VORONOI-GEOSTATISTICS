"""Lloyd relaxation via a scipy-Voronoi ``Field`` (reference implementation).

This module is an **optional, self-contained** Lloyd-relaxation backend that
sits alongside :meth:`VoronoiEngine._lloyd_step`.  The default engine keeps
using its mirror-padding + shapely pipeline; this ``Field`` class is only used
when a caller explicitly opts in with ``lloyd_backend="field"``.

Attribution
-----------
The ``Field`` class below is adapted from Douglas Duhaime's public gist:

    "Lloyd's algorithm implementation" — Douglas Duhaime (@duhaime)
    https://gist.github.com/duhaime/3e781194ebaccc28351a5d53989caa70

Background on the algorithm: https://en.wikipedia.org/wiki/Lloyd%27s_algorithm

Minimal adaptations from the original gist (kept as small as possible so the
attribution stays faithful):

* fixed the bounding-box ``y`` axis (the gist reads ``points[:, 0]`` twice);
* moved the "store this region" append out of the inner vertex loop so a region
  is recorded once, after all its vertices pass the bounding-box test;
* added a tiny ``relax()`` convenience wrapper.

Nothing here alters the existing Voronoi/render/export framework.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.spatial import Voronoi


class Field:
    """Create a Voronoi map that can be used to run Lloyd relaxation on an
    array of 2D points.

    For background, see: https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
    Adapted from https://gist.github.com/duhaime/3e781194ebaccc28351a5d53989caa70
    """

    def __init__(self, points=None):
        """Store the points and bounding box of the points to which Lloyd
        relaxation will be applied.

        @param [arr] points: a numpy array with shape n, 2, where n is number
            of points
        """
        if points is None or not len(points):
            raise Exception('points should be a numpy array with shape n,2')
        points = np.asarray(points, dtype=np.float64)
        x = points[:, 0]
        y = points[:, 1]
        self.bounding_box = [min(x), max(x), min(y), max(y)]
        self.points = points
        self.build_voronoi()

    def build_voronoi(self):
        """Build a Voronoi map from self.points. For background on
        self.voronoi attrs, see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html
        """
        eps = sys.float_info.epsilon
        self.voronoi = Voronoi(self.points)
        self.filtered_regions = []  # list of regions with vertices inside Voronoi map
        for region in self.voronoi.regions:
            inside_map = True  # is this region inside the Voronoi map?
            for index in region:  # index = the idx of a vertex in the current region
                # check if index is inside Voronoi map (indices == -1 are outside map)
                if index == -1:
                    inside_map = False
                    break
                # check if the current coordinate is in the Voronoi map's bounding box
                else:
                    coords = self.voronoi.vertices[index]
                    if not (self.bounding_box[0] - eps <= coords[0] and
                            self.bounding_box[1] + eps >= coords[0] and
                            self.bounding_box[2] - eps <= coords[1] and
                            self.bounding_box[3] + eps >= coords[1]):
                        inside_map = False
                        break
            # store the region if it has vertices and is inside Voronoi map
            if region != [] and inside_map:
                self.filtered_regions.append(region)

    def find_centroid(self, vertices):
        """Find the centroid of a Voronoi region described by `vertices`, and
        return a np array with the x and y coords of that centroid.

        The equation for the method used here to find the centroid of a 2D
        polygon is given here:
        https://en.wikipedia.org/wiki/Centroid#Of_a_polygon

        @params: np.array `vertices` a numpy array with shape n,2
        @returns np.array a numpy array that defines the x, y coords
            of the centroid described by `vertices`
        """
        area = 0
        centroid_x = 0
        centroid_y = 0
        for i in range(len(vertices) - 1):
            step = (vertices[i, 0] * vertices[i + 1, 1]) - (vertices[i + 1, 0] * vertices[i, 1])
            area += step
            centroid_x += (vertices[i, 0] + vertices[i + 1, 0]) * step
            centroid_y += (vertices[i, 1] + vertices[i + 1, 1]) * step
        area /= 2
        centroid_x = (1.0 / (6.0 * area)) * centroid_x
        centroid_y = (1.0 / (6.0 * area)) * centroid_y
        return np.array([[centroid_x, centroid_y]])

    def relax_points(self):
        """Moves each point to the centroid of its cell in the Voronoi map to
        "relax" the points (i.e. jitter them so as to spread them out within
        the space).
        """
        centroids = []
        for region in self.filtered_regions:
            vertices = self.voronoi.vertices[region + [region[0]], :]
            centroid = self.find_centroid(vertices)  # get the centroid of these verts
            centroids.append(list(centroid[0]))
        self.points = np.asarray(centroids, dtype=np.float64)  # new point positions
        self.build_voronoi()  # rebuild the voronoi map given new point positions
        return self.points


def relax(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Run ``iterations`` passes of Lloyd relaxation over ``points``.

    Thin convenience wrapper around :class:`Field`.  Returns an ``(M, 2)`` array
    of relaxed positions (``M`` may be < ``N`` because regions touching the
    convex hull are dropped by the filtered-region logic above).
    """
    field = Field(points=points)
    for _ in range(int(max(0, iterations))):
        field.relax_points()
    return np.asarray(field.points, dtype=np.float64)
