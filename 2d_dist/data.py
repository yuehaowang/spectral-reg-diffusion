"""2D toy distribution samplers.

get_2d_data_points(name, n_samples) returns an (n_samples, 2) float32 array
of samples from the named distribution. Randomness uses the module-level
``np.random`` state — call ``np.random.seed(...)`` first for reproducibility.
"""
import numpy as np
import sklearn.datasets


SUPPORTED = (
    "swissroll", "circles", "rings", "moons", "8gaussians",
    "pinwheel", "2spirals", "checkerboard",
)


def get_2d_data_points(name, n_samples):
    if name == "swissroll":
        x = sklearn.datasets.make_swiss_roll(n_samples=n_samples, noise=1.0)[0]
        return (x.astype("float32")[:, [0, 2]] / 5).astype("float32")

    if name == "circles":
        x = sklearn.datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.08)[0]
        return (x.astype("float32") * 3).astype("float32")

    if name == "rings":
        base = n_samples // 4
        ns = [base, base, base, n_samples - 3 * base]
        radii = [1.0, 0.75, 0.5, 0.25]
        pts = []
        for n, r in zip(ns, radii):
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pts.append(np.stack([np.cos(ang) * r, np.sin(ang) * r], axis=1))
        X = np.concatenate(pts, axis=0) * 3.0
        X = X + np.random.normal(scale=0.08, size=X.shape)
        return X.astype("float32")

    if name == "moons":
        x = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.1)[0]
        x = x.astype("float32")
        # Auto-promotes to float64 via the offset array — matches the original
        # code; the float32 cast happens later in load_data.
        return x * 2 + np.array([-1, -0.2])

    if name == "8gaussians":
        scale = 4.0
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            ( 1.0 / np.sqrt(2),  1.0 / np.sqrt(2)),
            ( 1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2),  1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        # Add point + center in float64 row-by-row; cast to float32 only
        # after the whole list is built. Matches the original byte-for-byte.
        dataset = []
        for _ in range(n_samples):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            c = centers[idx]
            point[0] += c[0]
            point[1] += c[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    if name == "pinwheel":
        radial_std, tangential_std = 0.3, 0.1
        num_classes, rate = 5, 0.25
        per_class = n_samples // num_classes
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        features = np.random.randn(num_classes * per_class, 2) * np.array(
            [radial_std, tangential_std]
        )
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), per_class)
        angles = rads[labels] + rate * np.exp(features[:, 0])
        rot = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
        ).T.reshape(-1, 2, 2)
        x = np.einsum("ti,tij->tj", features, rot)
        return (2 * np.random.permutation(x)).astype("float32")

    if name == "2spirals":
        n_per_arm = n_samples // 2
        n = np.sqrt(np.random.rand(n_per_arm, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_per_arm, 1) * 0.5
        d1y =  np.sin(n) * n + np.random.rand(n_per_arm, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x = x + np.random.randn(*x.shape) * 0.1
        return x.astype("float32")

    if name == "checkerboard":
        x1 = np.random.rand(n_samples) * 4 - 2
        x2 = (np.random.rand(n_samples) - np.random.randint(0, 2, n_samples) * 2
              + (np.floor(x1) % 2))
        return (np.stack([x1, x2], axis=1) * 2).astype("float32")

    raise ValueError(f"unknown distribution: {name!r} (supported: {SUPPORTED})")
