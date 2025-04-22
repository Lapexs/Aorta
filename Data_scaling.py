import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

# === Ścieżka do pliku ===
filename = r"C:\Users\PC\Desktop\TOM projekt\0020_H_AO_COA\Images\OSMSC0111-cm.vti"

# === Wczytaj obraz VTI ===
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(filename)
reader.Update()
imageData = reader.GetOutput()

# === Odczytaj spacing i rozmiary ===
spacing = imageData.GetSpacing()
dimensions = imageData.GetDimensions()
print("Rozmiar obrazu (voxel):", dimensions)
print("Spacing (mm):", spacing)

# === Zamień dane na tablicę NumPy ===
scalars = imageData.GetPointData().GetScalars()
np_array = vtk_to_numpy(scalars).reshape(dimensions[::-1])  # odwrotna kolejność [z, y, x]

# === Znajdź punkty aorty (wartość > 0) ===
coords_voxel = np.argwhere(np_array > 0)  # [z, y, x]
coords_xyz = coords_voxel[:, ::-1]        # → [x, y, z]

# === Przeskalowanie do jednostek fizycznych ===
scaled_coords_mm = coords_xyz * spacing  # element-wise

# === Wyświetlenie przykładowych punktów ===
print("\nPrzykładowe punkty voxelowe (x, y, z):")
print(coords_xyz[:5])

print("\nPo przeskalowaniu (mm):")
print(scaled_coords_mm[:5])


import pyvista as pv
import numpy as np

# Zakładamy, że masz już zmienną: scaled_coords_mm

# === Utwórz chmurę punktów w PyVista ===
point_cloud = pv.PolyData(scaled_coords_mm)

# === Możesz też pokolorować punkty lub dodać promień ===
point_cloud['value'] = np.ones(len(scaled_coords_mm))  # np. jednolita wartość

# === Tworzenie okna wizualizacji ===
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, color='red', point_size=2, render_points_as_spheres=True)
plotter.add_axes()
plotter.show_grid()
plotter.show(title="Segmentacja Aorty 3D")

