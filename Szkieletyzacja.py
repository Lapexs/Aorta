import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from skimage.morphology import skeletonize_3d
import networkx as nx
from scipy.interpolate import splprep, splev
import pyvista as pv

# === 1. Wczytanie obrazu .vti ===
filename = r"C:\Users\PC\Desktop\TOM projekt\0020_H_AO_COA\Images\OSMSC0111-cm.vti"
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(filename)
reader.Update()
imageData = reader.GetOutput()

# Pobierz spacing i wymiary
spacing = imageData.GetSpacing()         # (dx, dy, dz) w mm
dims = imageData.GetDimensions()         # (nx, ny, nz) w voxelach
print("Rozmiar (voxel):", dims, "Spacing (mm):", spacing)

# === 2. Zamiana na NumPy i maska binarna ===
scalars = imageData.GetPointData().GetScalars()
np_array = vtk_to_numpy(scalars).reshape(dims[::-1])  # reshape do [z, y, x]
binary = (np_array > 0).astype(np.uint8)              # 1 = aorta, 0 = tło

# === 3. Szkieletyzacja 3D ===
print("Robię skeletonize_3d...")
skeleton = skeletonize_3d(binary)  # wynik uint8

# === 4. Budowa grafu 6‑sąsiedztwa i znalezienie centerline ===
# 4.1. Koordynaty voxeli szkieletu
skel_vox = np.argwhere(skeleton > 0)       # [[z,y,x],...]
skel_pts = skel_vox[:, ::-1]               # → [[x,y,z],...]

# 4.2. Graf
G = nx.Graph()
pt2idx = {}
for idx, p in enumerate(map(tuple, skel_pts)):
    G.add_node(idx, coord=p)
    pt2idx[p] = idx

# 4.3. Dodaj krawędzie (6-sąsiedztwo)
dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
for idx, p in enumerate(skel_pts):
    for d in dirs:
        q = (p[0]+d[0], p[1]+d[1], p[2]+d[2])
        if q in pt2idx:
            G.add_edge(idx, pt2idx[q])

# 4.4. Znajdź końcówki (stopień=1) i najdłuższą trasę
endpoints = [n for n,d in G.degree() if d == 1]
max_len = 0
best_pair = (None, None)
for a in endpoints:
    lengths = nx.single_source_shortest_path_length(G, a)
    for b in endpoints:
        if b in lengths and lengths[b] > max_len:
            max_len = lengths[b]
            best_pair = (a, b)

path_idx = nx.shortest_path(G, source=best_pair[0], target=best_pair[1])
centerline_vox = np.array([G.nodes[i]['coord'] for i in path_idx])  # [[x,y,z],...]

# === 5. Przeskalowanie i wygładzenie spline’em ===
centerline_mm = centerline_vox * spacing

# Parametry spline (s = siła wygładzenia)
tck, u = splprep(centerline_mm.T, s=1.0)
u_fine = np.linspace(0, 1, len(centerline_mm)*5)
smooth_pts = np.vstack(splev(u_fine, tck)).T  # [[x,y,z],...]

# === 5a. Funkcja, by zapewnić parzystą liczbę punktów ===
def make_even_points(pts: np.ndarray) -> np.ndarray:
    if len(pts) % 2 != 0:
        return pts[:-1]
    return pts

centerline_mm = make_even_points(centerline_mm)
smooth_pts     = make_even_points(smooth_pts)

# === 6. Wizualizacja w PyVista ===
plotter = pv.Plotter()

# Surowy szkielet (półprzezroczysty)
cloud = pv.PolyData(skel_pts * spacing)
plotter.add_mesh(cloud,
                 color='lightgrey',
                 point_size=1,
                 render_points_as_spheres=True,
                 opacity=0.3)

# Surowa centerline (czerwona)
plotter.add_lines(centerline_mm,
                  color='red',
                  width=4,
                  label='Raw centerline')

# Wygładzona spline (niebieska)
plotter.add_lines(smooth_pts,
                  color='blue',
                  width=4,
                  label='Smoothed spline')

plotter.add_legend()
plotter.add_axes()
plotter.show_grid()
plotter.show(title="Centralna linia aorty")

# === 7. Długość centralnej linii ===
length = np.sum(np.linalg.norm(np.diff(centerline_mm, axis=0), axis=1))
print(f"Długość centerline: {length:.2f} mm")
