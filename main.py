import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import triangle

def read_geojson(input_file):
    """Чтение GeoJSON файла"""
    try:
        gdf = gpd.read_file(input_file)
        return gdf
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return None

def extract_points_and_holes(gdf):
    """Извлечение всех точек и внутренних контуров (дырок) из GeoDataFrame"""
    points = []
    holes = []
    segments = []

    def extract_coords(geometry, feature_idx):
        if geometry is None:
            return

        geom_type = geometry.geom_type
        if geom_type == 'Point':
            points.append((geometry.x, geometry.y))
        elif geom_type == 'MultiPoint':
            for point in geometry.geoms:
                points.append((point.x, point.y))
        elif geom_type == 'LineString':
            points.extend(geometry.coords)
            # Добавляем сегменты для линии
            for i in range(len(geometry.coords) - 1):
                segments.append((len(points) - len(geometry.coords) + i, len(points) - len(geometry.coords) + i + 1))
        elif geom_type == 'MultiLineString':
            for line in geometry.geoms:
                start_idx = len(points)
                points.extend(line.coords)
                for i in range(len(line.coords) - 1):
                    segments.append((start_idx + i, start_idx + i + 1))
        elif geom_type == 'Polygon':
            # Внешний контур
            start_idx = len(points)
            exterior_coords = list(geometry.exterior.coords)[:-1]  # Удаляем замыкающую точку
            points.extend(exterior_coords)
            for i in range(len(exterior_coords)):
                segments.append((start_idx + i, start_idx + (i + 1) % len(exterior_coords)))
            # Внутренние контуры (дырки)
            for interior in geometry.interiors:
                interior_coords = list(interior.coords)[:-1]
                hole_points = interior_coords
                hole_center = np.mean(hole_points, axis=0)  # Центр дырки
                holes.append(hole_center)
                start_idx = len(points)
                points.extend(hole_points)
                for i in range(len(hole_points)):
                    segments.append((start_idx + i, start_idx + (i + 1) % len(hole_points)))
        elif geom_type == 'MultiPolygon':
            for poly in geometry.geoms:
                extract_coords(poly, feature_idx)
        elif geom_type == 'GeometryCollection':
            for geom in geometry.geoms:
                extract_coords(geom, feature_idx)

    for idx, geometry in enumerate(gdf.geometry):
        extract_coords(geometry, idx)

    # Удаляем дубликаты точек
    unique_points = []
    seen = set()
    point_map = {}
    new_segments = []
    idx = 0
    for point in points:
        point_tuple = (float(point[0]), float(point[1]))
        if point_tuple not in seen:
            seen.add(point_tuple)
            point_map[(point[0], point[1])] = idx
            unique_points.append(point_tuple)
            idx += 1

    # Обновляем сегменты с учетом уникальных точек
    for seg in segments:
        p1 = points[seg[0]]
        p2 = points[seg[1]]
        new_seg = (point_map[(p1[0], p1[1])], point_map[(p2[0], p2[1])])
        if new_seg[0] != new_seg[1]:  # Пропускаем сегменты, соединяющие одну и ту же точку
            new_segments.append(new_seg)

    return np.array(unique_points), holes, new_segments

def create_constrained_triangulation(points, segments, holes):
    """Создание ограниченной триангуляции Делоне с учетом дырок"""
    if len(points) < 3:
        print("Недостаточно точек для триангуляции")
        return None

    # Подготовка данных для библиотеки triangle
    tri_input = {
        'vertices': points,
        'segments': np.array(segments) if segments else None,
        'holes': np.array(holes) if holes else None
    }

    # Выполняем триангуляцию с ограничениями
    try:
        tri = triangle.triangulate(tri_input, 'p')
        return tri
    except Exception as e:
        print(f"Ошибка триангуляции: {e}")
        return None

def create_triangulation_geojson(points, triangles):
    """Создание GeoJSON с триангуляцией"""
    points_list = [[float(point[0]), float(point[1])] for point in points]
    features = []
    for i, triangle in enumerate(triangles):
        triangle_indices = [int(j) for j in triangle]
        triangle_points = [points_list[j] for j in triangle_indices]
        triangle_points.append(triangle_points[0])  # Замыкаем полигон
        feature = {
            "type": "Feature",
            "properties": {
                "id": i,
                "triangle_index": i,
                "vertex_count": len(triangle_indices)
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [triangle_points]
            }
        }
        features.append(feature)
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    return geojson

def save_geojson(geojson_data, output_file):
    """Сохранение GeoJSON в файл"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
        print(f"Файл успешно сохранен: {output_file}")
    except Exception as e:
        print(f"Ошибка сохранения файла: {e}")

def visualize_triangulation(points, triangles, original_gdf=None):
    """Визуализация триангуляции"""
    try:
        plt.figure(figsize=(12, 8))
        if original_gdf is not None:
            original_gdf.plot(ax=plt.gca(), color='blue', alpha=0.3, edgecolor='black')
        plt.plot(points[:, 0], points[:, 1], 'o', color='red', markersize=3)
        plt.triplot(points[:, 0], points[:, 1], triangles, 'g-', alpha=0.7, linewidth=0.8)
        plt.title('Ограниченная триангуляция Делоне')
        plt.xlabel('Долгота')
        plt.ylabel('Широта')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Ошибка визуализации: {e}")

def numpy_to_python(obj):
    """Рекурсивно преобразует numpy типы в стандартные Python типы"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(item) for item in obj)
    else:
        return obj

def safe_save_geojson(geojson_data, output_file):
    """Безопасное сохранение с преобразованием numpy типов"""
    geojson_safe = numpy_to_python(geojson_data)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson_safe, f, indent=2, ensure_ascii=False)
        print(f"Файл успешно сохранен: {output_file}")
        return True
    except Exception as e:
        print(f"Ошибка сохранения файла: {e}")
        return False

def main():
    input_file = "input.geojson"
    output_file = "triangulation_output.geojson"

    print("Чтение исходного файла...")
    gdf = read_geojson(input_file)
    if gdf is None:
        print("Не удалось прочитать файл")
        return

    print(f"Прочитано объектов: {len(gdf)}")
    print(f"Типы геометрий: {set(geom.geom_type for geom in gdf.geometry)}")

    print("Извлечение точек, дырок и сегментов...")
    points, holes, segments = extract_points_and_holes(gdf)
    print(f"Извлечено уникальных точек: {len(points)}")
    print(f"Извлечено дырок: {len(holes)}")
    print(f"Извлечено сегментов: {len(segments)}")

    if len(points) < 3:
        print("Недостаточно точек для триангуляции")
        return

    print("Создание ограниченной триангуляции Делоне...")
    tri = create_constrained_triangulation(points, segments, holes)
    if tri is None:
        print("Не удалось создать триангуляцию")
        return

    print(f"Создано треугольников: {len(tri['triangles'])}")

    print("Создание выходного GeoJSON...")
    triangulation_geojson = create_triangulation_geojson(points, tri['triangles'])

    print("Сохранение результата...")
    success = safe_save_geojson(triangulation_geojson, output_file)

    if success:
        print("Визуализация результата...")
        visualize_triangulation(points, tri['triangles'], gdf)
        print(f"Триангуляция завершена! Результат сохранен в {output_file}")
    else:
        print("Ошибка при сохранении файла")

if __name__ == "__main__":
    main()