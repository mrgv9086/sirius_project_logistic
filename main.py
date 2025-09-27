import json
import numpy as np
from scipy.spatial import Delaunay
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
import matplotlib.pyplot as plt


def read_geojson(input_file):
    """Чтение GeoJSON файла"""
    try:
        gdf = gpd.read_file(input_file)
        return gdf
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return None


def extract_points_from_geojson(gdf):
    """Извлечение точек из GeoDataFrame"""
    points = []

    for geometry in gdf.geometry:
        if geometry.geom_type == 'Point':
            points.append((geometry.x, geometry.y))
        elif geometry.geom_type in ['Polygon', 'MultiPolygon']:
            # Для полигонов берем точки границы
            if geometry.geom_type == 'Polygon':
                coords = list(geometry.exterior.coords)
            else:  # MultiPolygon
                coords = []
                for poly in geometry.geoms:
                    coords.extend(list(poly.exterior.coords))

            points.extend(coords)
        elif geometry.geom_type == 'LineString':
            points.extend(list(geometry.coords))
        elif geometry.geom_type == 'MultiPoint':
            for point in geometry.geoms:
                points.append((point.x, point.y))

    # Удаляем дубликаты и преобразуем в список стандартных float
    unique_points = []
    seen = set()

    for point in points:
        # Преобразуем numpy типы в стандартные Python float
        point_tuple = (float(point[0]), float(point[1]))
        if point_tuple not in seen:
            seen.add(point_tuple)
            unique_points.append(point_tuple)

    return np.array(unique_points)


def create_triangulation(points):
    """Создание триангуляции Делоне"""
    if len(points) < 3:
        print("Недостаточно точек для триангуляции")
        return None

    # Выполняем триангуляцию
    tri = Delaunay(points)
    return tri


def create_triangulation_geojson(points, triangles):
    """Создание GeoJSON с триангуляцией"""
    features = []

    # Преобразуем points в список стандартных Python float
    points_list = []
    for point in points:
        points_list.append([float(point[0]), float(point[1])])

    for i, triangle in enumerate(triangles):
        # Преобразуем индексы треугольника в стандартные Python int
        triangle_indices = [int(j) for j in triangle]

        # Создаем полигон для каждого треугольника
        triangle_points = [points_list[j] for j in triangle_indices]
        # Замыкаем полигон (первая точка = последняя)
        triangle_points.append(triangle_points[0])

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

        # Отображаем исходные данные если есть
        if original_gdf is not None:
            original_gdf.plot(ax=plt.gca(), color='blue', alpha=0.3, edgecolor='black')

        # Отображаем точки
        plt.plot(points[:, 0], points[:, 1], 'o', color='red', markersize=3)

        # Отображаем треугольники
        plt.triplot(points[:, 0], points[:, 1], triangles, 'g-', alpha=0.7, linewidth=0.8)

        plt.title('Триангуляция Делоне')
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
    # Преобразуем все numpy типы в стандартные Python типы
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

    # Чтение исходного GeoJSON
    print("Чтение исходного файла...")
    gdf = read_geojson(input_file)

    if gdf is None:
        print("Не удалось прочитать файл")
        return

    print(f"Прочитано объектов: {len(gdf)}")
    print(f"Типы геометрий: {set(geom.geom_type for geom in gdf.geometry)}")

    # Извлечение точек
    print("Извлечение точек...")
    points = extract_points_from_geojson(gdf)
    print(f"Извлечено уникальных точек: {len(points)}")

    if len(points) < 3:
        print("Недостаточно точек для триангуляции")
        return

    # Создание триангуляции
    print("Создание триангуляции Делоне...")
    tri = create_triangulation(points)

    if tri is None:
        print("Не удалось создать триангуляцию")
        return

    print(f"Создано треугольников: {len(tri.simplices)}")

    # Создание GeoJSON с триангуляцией
    print("Создание выходного GeoJSON...")
    triangulation_geojson = create_triangulation_geojson(points, tri.simplices)

    # Сохранение результата (используем безопасный метод)
    print("Сохранение результата...")
    success = safe_save_geojson(triangulation_geojson, output_file)

    if success:
        # Визуализация
        print("Визуализация результата...")
        visualize_triangulation(points, tri.simplices, gdf)

        print(f"Триангуляция завершена! Результат сохранен в {output_file}")
    else:
        print("Ошибка при сохранении файла")


# Упрощенная версия для отладки
def simple_triangulation():
    """Упрощенная версия для тестирования"""
    input_file = "input.geojson"
    output_file = "simple_triangulation.geojson"

    try:
        # Чтение данных
        gdf = gpd.read_file(input_file)
        print(f"Прочитано объектов: {len(gdf)}")

        # Сбор всех координат
        all_coords = []
        for geom in gdf.geometry:
            if hasattr(geom, 'exterior'):
                all_coords.extend(list(geom.exterior.coords))
            elif hasattr(geom, 'coords'):
                all_coords.extend(list(geom.coords))
            elif geom.geom_type == 'Point':
                all_coords.append((geom.x, geom.y))

        # Удаление дубликатов
        unique_coords = list(set((round(x, 6), round(y, 6)) for x, y in all_coords))
        points = np.array(unique_coords)

        print(f"Уникальных точек: {len(points)}")

        if len(points) < 3:
            print("Недостаточно точек")
            return

        # Триангуляция
        tri = Delaunay(points)
        print(f"Создано треугольников: {len(tri.simplices)}")

        # Создание GeoJSON
        features = []
        for i, simplex in enumerate(tri.simplices):
            triangle_coords = []
            for idx in simplex:
                triangle_coords.append([float(points[idx][0]), float(points[idx][1])])
            triangle_coords.append(triangle_coords[0])  # Замыкаем полигон

            feature = {
                "type": "Feature",
                "properties": {"id": i},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [triangle_coords]
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        # Сохранение
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2)

        print(f"Успешно сохранено в {output_file}")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    # Основной запуск
    main()

    # Если основной метод не работает, попробуйте упрощенную версию:
    # print("\n--- Запуск упрощенной версии ---")
    # simple_triangulation()