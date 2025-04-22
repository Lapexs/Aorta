import pyvista as pv
import numpy as np
from skimage.morphology import skeletonize_3d
import vtk
import csv

# === 1. Wczytaj model aorty (.vtp)
mesh = pv.read(r"C:\Users\PC\Desktop\TOM projekt\0020_H_AO_COA\Models\0111_0001.vtp")

# === 2. Ustaw rozdzielczość voxelizacji
voxel_size = 1.0  # mm
bounds = mesh.bounds
x_dim = int((bounds[1] - bounds[0]) / voxel_size)
y_dim = int((bounds[3] - bounds[2]) / voxel_size)
z_dim = int((bounds[5] - bounds[4]) / voxel_size)

# === 3. Stwórz funkcję odległości od powierzchni
implicit_distance = vtk.vtkImplicitPolyDataDistance()
implicit_distance.SetInput(mesh)  # Poprawiona metoda

# === 4. Stwórz pustą siatkę voxelową (vtkImageData)
image = vtk.vtkImageData()
image.SetDimensions(x_dim, y_dim, z_dim)
image.SetSpacing(voxel_size, voxel_size, voxel_size)
image.SetOrigin(bounds[0], bounds[2], bounds[4])
image.AllocateScalars(vtk.VTK_FLOAT, 1)

# === 5. Wypełnij voxelowy obraz wartościami odległości
for z in range(z_dim):
    for y in range(y_dim):
        for x in range(x_dim):
            pt = [
                bounds[0] + x * voxel_size,
                bounds[2] + y * voxel_size,
                bounds[4] + z * voxel_size
            ]
            d = implicit_distance.EvaluateFunction(pt)
            image.SetScalarComponentFromFloat(x, y, z, 0, d)

# === 6. Konwersja do PyVista UniformGrid
volume = pv.wrap(image)

# === 7. Zidentyfikuj dostępne skalary w point_data
print("Dostępne skalary w volume:", volume.point_data.keys())

# Zakładając, że interesuje nas skalar o nazwie "vtkValidPointMask" (jeśli nie, zmień nazwę na dostępny skalar)
mask = volume.point_data["ImageScalars"] < 0  # Binaryzacja
binary_volume = mask.reshape(volume.dimensions[::-1])  # [z, y, x]

# === 8. Skeletonizacja 3D
skeleton = skeletonize_3d(binary_volume)

# === 9. Wyciągnięcie współrzędnych skeletonu
svox = np.argwhere(skeleton)           # [z, y, x]
coords_xyz = svox[:, ::-1]              # → [x, y, z]
spacing = np.array(volume.spacing)
origin  = np.array(volume.origin)
coords_mm = coords_xyz * spacing + origin

# === 10. Oblicz odległość od ściany dla punktów skeletonu
dist_calc = vtk.vtkImplicitPolyDataDistance()
dist_calc.SetInput(mesh)  # Używamy poprawionej metody
distances = [abs(dist_calc.EvaluateFunction(pt)) for pt in coords_mm]

# === 11. Utwórz chmurę punktów z danymi
skeleton_cloud = pv.PolyData(coords_mm)
skeleton_cloud["distance_to_wall"] = distances

# === 12. Zapisz do CSV
with open("skeleton_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x", "y", "z", "distance_to_wall_mm"])
    for pt, d in zip(coords_mm, distances):
        w.writerow([*pt, d])

# === 13. Wizualizacja
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightblue", opacity=0.3)
plotter.add_mesh(
    skeleton_cloud,
    scalars="distance_to_wall",
    cmap="plasma",
    point_size=4,
    render_points_as_spheres=True
)
#plotter.add_scalar_bar(title="Odległość od ściany (mm)")
#plotter.add_axes()
plotter.add_title("Centerline + analiza dystansu do ściany")
plotter.show()
