"""Village scene generator + meshcat visualization helpers.

Adapted from Fast Path Planning (https://github.com/cvxgrp/fastpathplanning) by
the Stanford University Convex Optimization Group, licensed under Apache-2.0. See
the ``LICENSE`` file in this directory for the full license text and attribution.
Modified for pybmt: the unused ``to_csd_plant`` (csdecomp) and ``fpp_to_pybezier``
helpers were removed; the rest of the scene generator + visualizers are reproduced.
"""

import colorsys
import itertools
import random
from fractions import Fraction

import numpy as np
from matplotlib.colors import to_hex as _to_hex
from meshcat import Visualizer
from meshcat.geometry import Box, Cylinder, MeshLambertMaterial, Sphere, TriangularMeshGeometry, tf
from meshcat.transformations import translation_matrix

# import mcubes
from pydrake.all import (
    HPolyhedron,
    Hyperrectangle,
    MathematicalProgram,
    SnoptSolver,
    SurfaceTriangle,
    VPolytope,
    le,
)
from scipy.spatial import ConvexHull

to_hex = lambda rgb: "0x" + _to_hex(rgb)[1:]


def outer_box(i, j, x, y, r, zmin, zmax):
    """Four free-space boxes surrounding an obstacle at (x, y) with radius r,
    within the unit cell [i, i+1] x [j, j+1], between zmin and zmax."""
    L = [[i, j, zmin], [x + r, j, zmin], [i, y + r, zmin], [i, j, zmin]]
    U = [[x - r, j + 1, zmax], [i + 1, j + 1, zmax], [i + 1, j + 1, zmax], [i + 1, y - r, zmax]]
    return L, U


def infinite_hues():
    yield Fraction(0)
    for k in itertools.count():
        i = 2**k  # zenos_dichotomy
        for j in range(1, i, 2):
            yield Fraction(j, i)


def hue_to_hsvs(h: Fraction):
    # tweak values to adjust scheme
    for s in [Fraction(6, 10)]:
        for v in [Fraction(6, 10), Fraction(9, 10)]:
            yield (h, s, v)


def rgb_to_css(rgb) -> str:
    uint8tuple = map(lambda y: int(y * 255), rgb)
    return tuple(uint8tuple)


def css_to_html(css):
    return f"<text style=background-color:{css}>&nbsp;&nbsp;&nbsp;&nbsp;</text>"


def n_colors(n=33, rgbs_ret=False):
    hues = infinite_hues()
    hsvs = itertools.chain.from_iterable(hue_to_hsvs(hue) for hue in hues)
    rgbs = (colorsys.hsv_to_rgb(*hsv) for hsv in hsvs)
    csss = (rgb_to_css(rgb) for rgb in rgbs)
    to_ret = list(itertools.islice(csss, n)) if rgbs_ret else list(itertools.islice(csss, n))
    return to_ret


def n_colors_random(n=33, rgbs_ret=False):
    colors = n_colors(100 * n, rgbs_ret)
    return random.sample(colors, n)


class EnvironmentVisualizer(Visualizer):

    def __init__(self):
        super().__init__()
        self["/Background"].set_property("visible", False)
        self.L = []
        self.U = []

    def cube(self, name, c, r, color=(1, 0, 0), opacity=1):
        c = np.array(c)
        self.L.append(c - r)
        self.U.append(c + r)
        return _cube(self, name, c, r, color, opacity)

    def box(self, name, l, u, color=(1, 0, 0), opacity=1):
        l = np.array(l)
        u = np.array(u)
        self.L.append(l)
        self.U.append(u)
        return _box(self, name, l, u, color, opacity)

    def box_visual(self, name, l, u, color=(1, 0, 0), opacity=1):
        l = np.array(l)
        u = np.array(u)
        # self.L.append(l)
        # self.U.append(u)
        return _box(self, name, l, u, color, opacity)


def _cube(vis, name, c, r, color, opacity):
    c = np.array(c, dtype=float)
    color = to_hex(color)
    material = MeshLambertMaterial(color=color, opacity=opacity)
    cube = vis[name]
    cube.set_object(Box(2 * r * np.ones(3)), material)
    cube.set_transform(translation_matrix(c))
    return cube


def _box(vis, name, l, u, color, opacity):
    l = np.array(l, dtype=float)
    u = np.array(u, dtype=float)
    color = to_hex(color)
    material = MeshLambertMaterial(color=color, opacity=opacity)
    box = vis[name]
    box.set_object(Box(u - l), material)
    c = (u - l) / 2
    box.set_transform(translation_matrix(l + c))
    return box


def _sphere(vis, name, pos, radius, color, opacity):
    pos = np.array(pos, dtype=float)
    color = to_hex(color)
    material = MeshLambertMaterial(color=color, opacity=opacity)
    box = vis[name]
    box.set_object(Sphere(radius), material)
    box.set_transform(translation_matrix(pos))
    return box


class Village(EnvironmentVisualizer):
    def __init__(self):
        super().__init__()
        self["/Grid"].set_property("visible", False)

    def ground(self, side, color=(1, 1, 1)):
        l = [-0.5, -0.5, -0.02]
        u = [side + 0.5, side + 0.5, -0.01]
        ground = self.box("ground", l, u, color, self.opacity)
        return ground

    def bush(self, name, c, r, h):
        c = np.array(c, dtype=float)
        l = [c[0] - r, c[1] - r, 0]
        u = [c[0] + r, c[1] + r, h]
        rand = 0.4 * (np.random.rand(3) - 0.5)
        color = (np.max([rand[0], 0]), 0.5 + rand[1], np.max([rand[2], 0]))
        bush = self.box(name, l, u, color, self.opacity)
        return bush

    def tree(self, name, c, r, r_trunk):
        c = np.array(c, dtype=float)
        l = [c[0] - r_trunk, c[1] - r_trunk, 0]
        u = [c[0] + r_trunk, c[1] + r_trunk, c[2] - r]
        color = (0.7, 0.35, 0)
        trunk = self.box(name + "tree/_trunk", l, u, color, self.opacity)
        rand = 0.4 * (np.random.rand(3) - 0.5)
        color = (0.2 + rand[0], 0.8 + rand[1], 0.2 + rand[2])
        foliage = self.cube(name + "tree/_foliage", c, r, color, self.opacity)
        return trunk, foliage

    def building(self, name, c, r, n):

        c = np.array(c, dtype=float)
        r = np.array(r, dtype=float)
        n = np.array(n, dtype=int)

        h = 2 * r[2]
        l = [c[0] - r[0], c[1] - r[1], 0]
        u = [c[0] + r[0], c[1] + r[1], h]
        color = (0.8, 0.8, 0.8)
        body = self.box(name + "building/_body", l, u, color, self.opacity)

        roof_ratio = 1 / 50
        l[2] = h
        u[2] = h * (1 + roof_ratio)
        color = (0.7, 0.0, 0.0)
        roof = self.box_visual(name + "building/_roof", l, u, color, self.opacity)

        windows = []
        eps = 1e-2
        wcolor = (0.3, 0.6, 1)
        fcolor = (0, 0, 0)
        d = 2 * r / n  # Distance between windows in all directions.
        wr = d / 3.5  # Window radius in all directions.
        f = wr / 5  # Frame size in all directions.

        dw = np.array([wr[0], r[1] + 2 * eps, wr[2]])
        df = [f[0], -eps, f[2]]
        for i in range(n[0]):
            for j in range(n[2]):
                cij = np.array([c[0] - r[0] + d[0] * (i + 0.5), c[1], d[2] * (j + 0.5)])
                lij = cij - dw
                uij = cij + dw
                windows.append(
                    self.box_visual(
                        name + f"window/_window_x_{i}_{j}", lij, uij, wcolor, self.opacity
                    )
                )
                lij -= df
                uij += df
                windows.append(
                    self.box_visual(
                        name + f"window/_frame_x_{i}_{j}", lij, uij, fcolor, self.opacity
                    )
                )

        dw = np.array([r[0] + 2 * eps, wr[1], wr[2]])
        df = [-eps, f[1], f[2]]
        for i in range(n[1]):
            for j in range(n[2]):
                cij = np.array([c[0], c[1] - r[1] + d[1] * (i + 0.5), d[2] * (j + 0.5)])
                lij = cij - dw
                uij = cij + dw
                windows.append(
                    self.box_visual(
                        name + f"window/_window_y_{i}_{j}", lij, uij, wcolor, self.opacity
                    )
                )

                lij -= df
                uij += df
                windows.append(
                    self.box_visual(
                        name + f"window/_frame_y_{i}_{j}", lij, uij, fcolor, self.opacity
                    )
                )

        return body, roof, windows

    def build(
        self,
        village_height=5,
        village_side=19,
        building_every=5,
        density=0.3,
        opacity=1,
        seed=12,
        r_uav=0.1,
    ):
        np.random.seed(seed)
        self.density = density
        self.opacity = opacity
        self.r_uav = r_uav
        self.L_dom = [-0.5, -0.5, 0]
        self.U_dom = [village_side + 0.5, village_side + 0.5, village_height]
        self._safe_L = []
        self._safe_U = []

        assert (village_side + 1) % building_every == 0

        def direction():
            I = np.eye(2)
            directions = np.vstack((I, -I))
            return directions[np.random.randint(0, 4)]

        def walk(m):
            d1 = direction()
            starts = [np.zeros(2)]
            ends = [d1]
            blocks = [np.zeros(2), d1]
            for i in range(m):
                d2 = direction()
                blocks.append(blocks[-1] + d2)
                if all(d2 == d1):
                    ends[-1] += d1
                else:
                    starts.append(ends[-1])
                    ends.append(starts[-1] + d2)
                    d1 = d2
            return np.array(starts), np.array(ends), np.array(blocks)

        def _building(i, j, m):
            starts, ends, blocks = walk(m)
            offset = np.array([i, j]) + 0.5
            starts += offset
            ends += offset
            blocks += [i, j]
            for k, (s, e) in enumerate(zip(starts, ends)):
                c = (s + e) / 2
                d = np.abs(e - s) + 1
                r = list(0.5 * d) + [village_height / 2]
                n = list(d) + [village_height]
                self.building(f"building/building_{i}_{j}_{k}", c, r, n)
            return blocks

        def _tree(i, j):
            name = f"tree/tree_{i}_{j}"
            r = 0.5 - r_uav  # Radius of foliage (shrunk by UAV radius).
            r_trunk = 0.1
            low = [i + 0.5, j + 0.5, 1]
            high = [i + 0.5, j + 0.5, village_height - 0.5]
            c = np.random.uniform(low, high)
            self.tree(name, c, r, r_trunk)
            Lij, Uij = outer_box(i, j, c[0], c[1], r_trunk + r_uav, 0, c[2] - 0.5)  # Around trunk.
            Lij.append([i, j, c[2] + 0.5])  # Above foliage.
            Uij.append([i + 1, j + 1, village_height])
            self._safe_L += Lij
            self._safe_U += Uij

        def _bush(i, j):
            name = f"bush/bush_{i}_{j}"
            r = np.random.uniform(low=0.1, high=0.35)  # Radius.
            h = 4 * r  # Height.
            c = np.array([i + 0.5, j + 0.5])
            self.bush(name, c, r, h)
            Lij, Uij = outer_box(i, j, c[0], c[1], r + r_uav, 0, h + r_uav)  # Around bush.
            Lij.append([i, j, h + r_uav])  # Above bush.
            Uij.append([i + 1, j + 1, village_height])
            self._safe_L += Lij
            self._safe_U += Uij

        self.delete()

        # village ground
        ground_color = (0.7, 1, 0.7)
        ground = self.ground(village_side, ground_color)

        # buildings
        blocks = []
        for i in range(village_side):
            for j in range(village_side):
                if (i + 1) % building_every == 0 and (j + 1) % building_every == 0:
                    blocks.append(_building(i, j, 4))
        blocks = [tuple(b) for b in np.vstack(blocks)]

        # trees, bushes, and empty cells
        for i in range(village_side):
            for j in range(village_side):
                if (i, j) not in blocks:
                    r = np.random.rand()
                    if r > 1 - self.density:
                        if r > 1 - 0.5 * self.density:
                            _bush(i, j)
                        else:
                            _tree(i, j)
                    else:
                        # Empty cell: entire cell is free space.
                        self._safe_L.append([i, j, 0])
                        self._safe_U.append([i + 1, j + 1, village_height])

        self.obstacles = [Hyperrectangle(l, u) for l, u in zip(self.L, self.U)]
        # plot_collision(self, self.L, self.U)

        # Border safe boxes (1-unit margin around the village grid, full height)
        m = 1.0
        s = float(village_side)
        h = float(village_height)
        self._safe_L += [[-m, -m, 0], [-m, s, 0], [-m, 0, 0], [s, 0, 0]]
        self._safe_U += [[s + m, 0, h], [s + m, s + m, h], [0, s, h], [s + m, s, h]]

        self.safe_L = np.array(self._safe_L)
        self.safe_U = np.array(self._safe_U)

        self.domain = HPolyhedron.MakeBox(np.array(self.L_dom) - 1, np.array(self.U_dom) + 1)
        # self.domain = HPolyhedron.MakeBox(np.array(self.L_dom), np.array(self.U_dom))

    def plot_regions(
        self,
        regions,
        ellipses=None,
        region_suffix="",
        colors=None,
        wireframe=False,
        opacity=0.7,
        fill=True,
        line_width=10,
        darken_factor=0.2,
        el_opacity=0.3,
    ):
        if colors is None:
            colors = n_colors_random(len(regions), rgbs_ret=True)

        for i, region in enumerate(regions):
            c = tuple([col / 255 for col in colors[i]])  # ,opacity)
            prefix = f"/cfree/regions{region_suffix}/{i}"
            name = prefix  # + "/hpoly"
            if region.ambient_dimension() == 3:
                # self.plot_hpoly3d(name, region,
                #                   c, wireframe = wireframe, resolution = 30, opacity = opacity)
                self.plot_hpoly3d_2(name, region, c, wireframe=wireframe, opacity=opacity)

    def plot_plane(
        self, name, a, b, bounds=None, color=(0.5, 0.5, 1.0), opacity=0.5, thickness=0.01
    ):
        """
        Plot a plane defined by a^T x + b = 0 using a transformed slim box.

        Args:
            name: Name for the plane object in meshcat
            a: Normal vector of the plane (3D numpy array)
            b: Scalar offset (plane equation: a^T x + b = 0)
            bounds: Optional bounds for the plane size. If None, uses domain bounds.
                    Can be [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            color: RGB tuple for plane color
            opacity: Transparency of the plane
            thickness: How thick to make the plane (very small for thin appearance)
        """
        import numpy as np

        a = np.array(a, dtype=float)
        a = a / np.linalg.norm(a)  # Normalize the normal vector

        # Use domain bounds if not specified
        if bounds is None:
            if hasattr(self, "L_dom") and hasattr(self, "U_dom"):
                bounds = [[self.L_dom[i], self.U_dom[i]] for i in range(3)]
            else:
                bounds = [[-10, 10], [-10, 10], [-10, 10]]

        # Find a point on the plane
        # Choose the coordinate with the largest normal component to avoid division by small numbers
        max_idx = np.argmax(np.abs(a))
        plane_point = np.zeros(3)
        if abs(a[max_idx]) > 1e-8:
            plane_point[max_idx] = -b / a[max_idx]

        # Create two orthogonal vectors in the plane
        # First vector: project a unit vector onto the plane
        if max_idx == 0:
            v1 = np.array([0, 1, 0])
        else:
            v1 = np.array([1, 0, 0])

        # Project v1 onto the plane: v1 - (v1 · a)a
        v1 = v1 - np.dot(v1, a) * a
        v1 = v1 / np.linalg.norm(v1)

        # Second vector: cross product of normal and first vector
        v2 = np.cross(a, v1)
        v2 = v2 / np.linalg.norm(v2)

        # Determine plane size based on bounds
        # Project bounds corners onto the plane and find the extent
        corners = []
        for x in bounds[0]:
            for y in bounds[1]:
                for z in bounds[2]:
                    corner = np.array([x, y, z])
                    # Project corner onto plane
                    projected = corner - np.dot(corner - plane_point, a) * a
                    corners.append(projected)

        corners = np.array(corners)

        # Find the extent in the plane coordinate system
        u_coords = [np.dot(corner - plane_point, v1) for corner in corners]
        v_coords = [np.dot(corner - plane_point, v2) for corner in corners]

        u_size = max(u_coords) - min(u_coords)
        v_size = max(v_coords) - min(v_coords)

        # Make the plane a bit larger to ensure it covers the bounds
        u_size *= 1.2
        v_size *= 1.2

        # Create the box dimensions (very thin in normal direction)
        box_size = [u_size, v_size, thickness]

        # Create rotation matrix to align box with plane
        # Box's local z-axis should align with plane's normal
        # Box's local x-axis should align with v1
        # Box's local y-axis should align with v2
        rotation_matrix = np.column_stack([v1, v2, a])

        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = plane_point

        # Create the box
        color_hex = to_hex(color)
        material = MeshLambertMaterial(color=color_hex, opacity=opacity)
        plane_obj = self[name]
        plane_obj.set_object(Box(box_size), material)
        plane_obj.set_transform(transform)

        return plane_obj

    def plot_hpoly3d_2(self, name, region, color, wireframe, opacity):
        region = HPolyhedron(region.A(), region.b())
        verts = VPolytope(region).vertices().T
        hull = ConvexHull(verts)
        triangles = []
        for s in hull.simplices:
            triangles.append(s)
        tri_drake = [SurfaceTriangle(*t) for t in triangles]
        obj = self[name]
        # objwf = self[name+'wf']
        col = to_hex(color)
        material = MeshLambertMaterial(color=col, opacity=opacity)
        obj.set_object(TriangularMeshGeometry(verts, triangles), material)
        material = MeshLambertMaterial(color=col, opacity=0.95, wireframe=True)
        # objwf.set_object(TriangularMeshGeometry(verts, triangles), material)
        # objwf.set_property("Visible", True)
        obj.set_property("Visible", True)

    def get_AABB_limits(self, hpoly, dim=3):
        max_limits = []
        min_limits = []
        A = hpoly.A()
        b = hpoly.b()

        for idx in range(dim):
            aabbprog = MathematicalProgram()
            x = aabbprog.NewContinuousVariables(dim, "x")
            cost = x[idx]
            aabbprog.AddCost(cost)
            aabbprog.AddConstraint(le(A @ x, b))
            solver = SnoptSolver()
            result = solver.Solve(aabbprog)
            min_limits.append(result.get_optimal_cost() - 0.01)
            aabbprog = MathematicalProgram()
            x = aabbprog.NewContinuousVariables(dim, "x")
            cost = -x[idx]
            aabbprog.AddCost(cost)
            aabbprog.AddConstraint(le(A @ x, b))
            solver = SnoptSolver()
            result = solver.Solve(aabbprog)
            max_limits.append(-result.get_optimal_cost() + 0.01)
        return max_limits, min_limits

    def convert_box(self, l, u):
        pos = 0.5 * (l + u)
        size = u - l
        return pos, size


to_hex = lambda rgb: "0x" + _to_hex(rgb)[1:]


def plot_collision(vis, ls, us, col=[1, 0, 0], opacity=0.5):
    for id, (l, u) in enumerate(zip(ls, us)):
        _box(vis, f"col/{id}", l, u, col, opacity=opacity)


def compute_rotation_matrix(a, b):
    # Normalize the points
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    # Calculate the rotation axis
    rotation_axis = np.cross(a, b)
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Calculate the rotation angle
    dot_product = np.dot(a, b)
    rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Construct the rotation matrix using Rodrigues' formula
    skew_matrix = np.array(
        [
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0],
        ]
    )
    rotation_matrix = (
        np.eye(3)
        + np.sin(rotation_angle) * skew_matrix
        + (1 - np.cos(rotation_angle)) * np.dot(skew_matrix, skew_matrix)
    )

    return rotation_matrix


def plot_edge(world, pt1, pt2, name, color, opacity, size=0.01):

    material = MeshLambertMaterial(color=to_hex(color), opacity=opacity)
    world["/" + name].set_object(Cylinder(np.float64(np.linalg.norm(pt1 - pt2)), size), material)

    dir = np.float64(pt2 - pt1)
    rot = compute_rotation_matrix(np.array([0, 1, 0]), dir)
    # print(rot)
    offs = rot @ np.array([0, np.linalg.norm(pt1 - pt2) / 2, 0])
    tfmat = np.zeros((4, 4))
    tfmat[:3, :3] = rot
    tfmat[:3, -1] = offs + pt1.squeeze()
    tfmat[3, 3] = 1
    world["/" + name].set_transform(tfmat)


def plot_edges(world, edges, name, color, opacity=1, size=0.01):
    for i, e in enumerate(edges):
        plot_edge(world, e[0], e[1], name + f"/e_{i}", color, opacity, size=size)


def plot_points(world, points, name, color, opacity, size=1):
    for idx, pt in enumerate(points):
        world[f"/{name}/{idx}"].set_object(
            Sphere(size),
            MeshLambertMaterial(color=to_hex((color[0], color[1], color[2])), opacity=opacity),
        )
        world[f"/{name}/{idx}"].set_transform(tf.translation_matrix(np.array(pt)))


def plot_path(env, path, name, color=[0, 0, 0], opacity=1, size=0.05):
    edges = []
    for idx in range(path.shape[0] - 1):
        edges.append([path[idx], path[idx + 1]])
    plot_edges(env, edges, f"{name}/path", color, opacity=1, size=size)
    plot_points(env, path, f"{name}/nodes", color, opacity=1, size=size * 2.5)


from pybezier import CompositeBezierCurve


def plot_composite_bezier_curve(
    env, traj: CompositeBezierCurve, name, color=[0, 0, 0], opacity=1, size=0.05
):
    wp_path = np.array([traj(t) for t in np.linspace(traj.initial_time, traj.final_time, 100)])
    edges = []
    for idx in range(wp_path.shape[0] - 1):
        edges.append([wp_path[idx], wp_path[idx + 1]])
    plot_edges(env, edges, f"{name}/path", color, opacity=1, size=size)
    segments = traj.curves
    stitching_pts = np.array([s.initial_point for s in segments] + [segments[-1].final_point])
    plot_points(env, stitching_pts, f"{name}/nodes", color, opacity=1, size=size * 3)
