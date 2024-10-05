from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import colors as pltcolors
from collections import deque, defaultdict
from dataclasses import dataclass


class TeamClassifier:

    @dataclass
    class Colors:
        bgr: tuple[int, int, int]
        lab: tuple[float, float, float]

    @staticmethod
    # returns the BGR colors (0-255) for drawing and the LAB colors(0-1) for
    # further sorting use
    def get_colors(roi: NDArray, max_clusters: int) -> \
            None or list[TeamClassifier.Colors]:

        if not roi.size:
            return None

        roi = cv.resize(roi, (min(32, roi.shape[1]), min(32, roi.shape[0])))

        # LAB is a useful color space for color stats. it is color components like HSV but
        # doesn't have the circular H which has a discontinuity at 0deg
        # an option would be use cos(H), sin(H), S, V
        roi = cv.cvtColor(roi, cv.COLOR_BGR2LAB)
        roi = roi.astype(np.float32) / 255.0
        roi = roi.reshape(-1, 3)

        labels = {}
        silhouette_scores = {}
        colors = defaultdict(lambda: list())
        cluster_centers = {}
        inertias = {}
        for num_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
            kmeans.fit_predict(roi)
            cs_lab = kmeans.cluster_centers_.copy()
            cs_bgr = (cs_lab * 255).astype(np.uint8)
            cs_bgr = np.expand_dims(cs_bgr, axis=0)
            cs_bgr = cv.cvtColor(cs_bgr, cv.COLOR_LAB2BGR)
            cs_bgr = tuple(map(tuple, np.squeeze(cs_bgr, axis=0)))
            cs_lab = tuple(map(tuple, cs_lab))
            for c_bgr, c_lab in zip(cs_bgr, cs_lab):
                colors[num_clusters].append(TeamClassifier.Colors(c_bgr, c_lab))
            labels[num_clusters] = kmeans.labels_.copy()
            silhouette_scores[num_clusters] = \
                silhouette_score(roi, kmeans.labels_) if num_clusters > 1 else 0.
            cluster_centers[num_clusters] = kmeans.cluster_centers_.copy()
            inertias[num_clusters] = kmeans.inertia_

        # the number of colors found is the best silhouette score
        # but we try the 1 cluster case just by trying to see if a large
        # inertia step occurs. this works with canned data but I haven't found
        # a 'real' data case with just one cluster
        if max_clusters == 1:
            num_colors_found = 1
        else:
            if inertias[1] < inertias[2] * 4.0:
                num_colors_found = 1
            else:
                num_colors_found = max(silhouette_scores, key=silhouette_scores.get)

        # sort colors by count, ie, most prevalent first
        counts = [np.count_nonzero(np.where(labels[num_colors_found] == i))
                  for i in range(num_colors_found)]
        order = np.argsort(counts, )[::-1]

        sorted_colors = []
        for i in order:
            sorted_colors.append(colors[num_colors_found][i])

        return sorted_colors

    def __init__(self, team_colors=dict[str, str]):

        self._kmeans = KMeans(n_clusters=2, n_init='auto')
        self._player_colors = deque(maxlen=64)
        self._team_colors_lab = {}
        self._team_colors_bgr = {}
        self._player_colors_lab = None
        for name, color in team_colors.items():
            color_bgr = (np.array(pltcolors.to_rgb(color)[::-1]) * 255.).astype(np.uint8)
            self._team_colors_bgr[name] = color_bgr
            color_bgr = np.expand_dims(color_bgr, axis=(0, 1))
            color_lab = cv.cvtColor(color_bgr, cv.COLOR_BGR2LAB).astype(np.float32)
            color_lab /= 255.
            color_lab = np.squeeze(color_lab, axis=(0, 1))
            self._team_colors_lab[name] = color_lab

    def predict(self, players_colors: list[tuple[float, float, float]]):

        self._player_colors += players_colors
        if len(self._player_colors) < 16:
            return None

        labels = self._kmeans.fit_predict(self._player_colors)[-len(players_colors):]

        # sort to closest to user-entered color
        # TODO this algorithm would be better if we tried all combinations
        # TODO (as opposed to the combinations for each pair independantly)
        self._player_colors_lab = self._kmeans.cluster_centers_.copy()
        ordered_colors = {}
        used = set()
        for team_name, color in self._team_colors_lab.items():
            dists = np.linalg.norm(self._player_colors_lab - color, axis=1)
            order = np.argsort(dists)
            for i in order:
                if i not in used:
                    ordered_colors[team_name] = i
                    used.add(i)
                    break

        teams = {team_name: np.where(labels == i)[0]
                 for team_name, i in ordered_colors.items()}

        return teams

    def uniform_color_bgr(self, name: str) -> tuple[int, int, int]:
        if name in self._team_colors_bgr:
            return tuple[int, int, int](
                tuple(map(int, self._team_colors_bgr[name]))
            )

    @staticmethod
    def draw(m, body_box, body_colors):
        if body_colors is not None:
            for i, body_color in enumerate(body_colors):
                body_color = [int(c) for c in body_color.bgr]
                body_pt1_p = body_box[0] + np.array([i * 9, i * 9])
                body_pt2_p = body_box[1] - np.array([i * 9, i * 9])
                m = cv.rectangle(m, body_pt1_p, body_pt2_p, body_color, 9)
        return m
