import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import triangle


# Функция для чтения GeoJSON файла
def read_geojson(input_file):
    """Чтение GeoJSON файла и возврат GeoDataFrame.

    Args:
        input_file (str): Путь к входному GeoJSON файлу.

    Returns:
        GeoDataFrame: Объект GeoDataFrame с геометриями или None в случае ошибки.
    """
    try:
        gdf = gpd.read_file(input_file)  # Читаем GeoJSON файл с помощью GeoPandas
        return gdf
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")  # Выводим сообщение об ошибке, если чтение не удалось
        return None


# Функция для извлечения точек, дырок и сегментов из GeoDataFrame
def extract_points_and_holes(gdf):
    """Извлечение всех точек, дырок и сегментов из GeoDataFrame.

    Args:
        gdf (GeoDataFrame): Входной GeoDataFrame с геометриями.

    Returns:
        tuple: Кортеж из трех элементов:
            - np.array: Массив уникальных точек (координат).
            - list: Список координат центров дырок.
            - list: Список сегментов (пар индексов точек).
    """
    points = []  # Список для хранения всех точек
    holes = []  # Список для хранения центров дырок
    segments = []  # Список для хранения сегментов (ребер)

    # Внутренняя функция для обработки геометрий
    def extract_coords(geometry, feature_idx):
        """Рекурсивная обработка геометрий для извлечения точек, дырок и сегментов.

        Args:
            geometry: Объект геометрии из GeoDataFrame.
            feature_idx (int): Индекс объекта для идентификации.
        """
        if geometry is None:
            return

        geom_type = geometry.geom_type  # Получаем тип геометрии
        if geom_type == 'Point':
            points.append((geometry.x, geometry.y))  # Добавляем координаты точки
        elif geom_type == 'MultiPoint':
            for point in geometry.geoms:
                points.append((point.x, point.y))  # Добавляем координаты всех точек из MultiPoint
        elif geom_type == 'LineString':
            points.extend(geometry.coords)  # Добавляем все координаты линии
            # Формируем сегменты для линии
            for i in range(len(geometry.coords) - 1):
                segments.append((len(points) - len(geometry.coords) + i, len(points) - len(geometry.coords) + i + 1))
        elif geom_type == 'MultiLineString':
            for line in geometry.geoms:
                start_idx = len(points)  # Запоминаем начальный индекс точек линии
                points.extend(line.coords)  # Добавляем координаты линии
                for i in range(len(line.coords) - 1):
                    segments.append((start_idx + i, start_idx + i + 1))  # Формируем сегменты
        elif geom_type == 'Polygon':
            # Обработка внешнего контура полигона
            start_idx = len(points)
            exterior_coords = list(geometry.exterior.coords)[:-1]  # Удаляем замыкающую точку
            points.extend(exterior_coords)  # Добавляем координаты внешнего контура
            for i in range(len(exterior_coords)):
                segments.append((start_idx + i, start_idx + (i + 1) % len(exterior_coords)))  # Формируем сегменты
            # Обработка внутренних контуров (дырок)
            for interior in geometry.interiors:
                interior_coords = list(interior.coords)[:-1]  # Удаляем замыкающую точку
                hole_points = interior_coords
                hole_center = np.mean(hole_points, axis=0)  # Вычисляем центр дырки
                holes.append(hole_center)  # Добавляем центр дырки
                start_idx = len(points)
                points.extend(hole_points)  # Добавляем координаты дырки
                for i in range(len(hole_points)):
                    segments.append((start_idx + i, start_idx + (i + 1) % len(hole_points)))  # Формируем сегменты
        elif geom_type == 'MultiPolygon':
            for poly in geometry.geoms:
                extract_coords(poly, feature_idx)  # Рекурсивно обрабатываем каждый полигон
        elif geom_type == 'GeometryCollection':
            for geom in geometry.geoms:
                extract_coords(geom, feature_idx)  # Рекурсивно обрабатываем каждую геометрию

    # Обрабатываем все геометрии в GeoDataFrame
    for idx, geometry in enumerate(gdf.geometry):
        extract_coords(geometry, idx)

    # Удаляем дубликаты точек
    unique_points = []
    seen = set()
    point_map = {}
    idx = 0
    for point in points:
        point_tuple = (float(point[0]), float(point[1]))  # Преобразуем в кортеж для хеширования
        if point_tuple not in seen:
            seen.add(point_tuple)
            point_map[(point[0], point[1])] = idx  # Сопоставляем координаты с индексом
            unique_points.append(point_tuple)
            idx += 1

    # Обновляем сегменты с учетом уникальных точек
    new_segments = []
    for seg in segments:
        p1 = points[seg[0]]
        p2 = points[seg[1]]
        new_seg = (point_map[(p1[0], p1[1])], point_map[(p2[0], p2[1])])
        if new_seg[0] != new_seg[1]:  # Пропускаем сегменты, соединяющие одну и ту же точку
            new_segments.append(new_seg)

    return np.array(unique_points), holes, new_segments


# Функция для создания ограниченной триангуляции Делоне
def create_constrained_triangulation(points, segments, holes):
    """Создание ограниченной триангуляции Делоне с учетом дырок.

    Args:
        points (np.array): Массив координат точек.
        segments (list): Список сегментов (ребер).
        holes (list): Список координат центров дырок.

    Returns:
        dict: Результат триангуляции от библиотеки triangle или None в случае ошибки.
    """
    if len(points) < 3:
        print("Недостаточно точек для триангуляции")
        return None

    # Подготовка данных для библиотеки triangle
    tri_input = {
        'vertices': points,  # Координаты вершин
        'segments': np.array(segments) if segments else None,  # Сегменты (ребра)
        'holes': np.array(holes) if holes else None  # Центры дырок
    }

    # Выполняем триангуляцию с ограничениями
    try:
        tri = triangle.triangulate(tri_input, 'p')  # 'p' указывает на ограниченную триангуляцию
        return tri
    except Exception as e:
        print(f"Ошибка триангуляции: {e}")
        return None


# Функция для создания GeoJSON с триангуляцией
def create_triangulation_geojson(points, triangles):
    """Создание GeoJSON с треугольниками триангуляции.

    Args:
        points (np.array): Массив координат точек.
        triangles (list): Список треугольников (индексы вершин).

    Returns:
        dict: GeoJSON словарь, содержащий треугольники как полигоны.
    """
    points_list = [[float(point[0]), float(point[1])] for point in points]  # Преобразуем точки в список
    features = []
    for i, triangle in enumerate(triangles):
        triangle_indices = [int(j) for j in triangle]  # Получаем индексы вершин треугольника
        triangle_points = [points_list[j] for j in triangle_indices]  # Координаты вершин
        triangle_points.append(triangle_points[0])  # Замыкаем полигон
        feature = {
            "type": "Feature",
            "properties": {
                "id": i,  # Уникальный идентификатор
                "triangle_index": i,  # Индекс треугольника
                "vertex_count": len(triangle_indices)  # Количество вершин
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [triangle_points]  # Координаты полигона
            }
        }
        features.append(feature)
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    return geojson


# Функция для сохранения GeoJSON в файл
def save_geojson(geojson_data, output_file):
    """Сохранение GeoJSON в файл.

    Args:
        geojson_data (dict): GeoJSON словарь для сохранения.
        output_file (str): Путь к выходному файлу.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)  # Сохраняем с отступами и поддержкой UTF-8
        print(f"Файл успешно сохранен: {output_file}")
    except Exception as e:
        print(f"Ошибка сохранения файла: {e}")


# Функция для визуализации триангуляции
def visualize_triangulation(points, triangles, original_gdf=None):
    """Визуализация триангуляции с использованием matplotlib.

    Args:
        points (np.array): Массив координат точек.
        triangles (list): Список треугольников (индексы вершин).
        original_gdf (GeoDataFrame, optional): Исходный GeoDataFrame для наложения.
    """
    try:
        plt.figure(figsize=(12, 8))  # Создаем новое окно с заданным размером
        if original_gdf is not None:
            original_gdf.plot(ax=plt.gca(), color='blue', alpha=0.3, edgecolor='black')  # Рисуем исходные геометрии
        plt.plot(points[:, 0], points[:, 1], 'o', color='red', markersize=3)  # Рисуем точки
        plt.triplot(points[:, 0], points[:, 1], triangles, 'g-', alpha=0.7, linewidth=0.8)  # Рисуем треугольники
        plt.title('Ограниченная триангуляция Делоне')  # Заголовок графика
        plt.xlabel('Долгота')  # Подпись оси X
        plt.ylabel('Широта')  # Подпись оси Y
        plt.grid(True, alpha=0.3)  # Добавляем сетку
        plt.axis('equal')  # Устанавливаем равные масштабы осей
        plt.tight_layout()  # Оптимизируем расположение
        plt.show()  # Показываем график
    except Exception as e:
        print(f"Ошибка визуализации: {e}")


# Функция для преобразования numpy типов в стандартные Python типы
def numpy_to_python(obj):
    """Рекурсивно преобразует numpy типы в стандартные Python типы.

    Args:
        obj: Объект, который нужно преобразовать.

    Returns:
        Преобразованный объект с Python типами.
    """
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


# Функция для безопасного сохранения GeoJSON
def safe_save_geojson(geojson_data, output_file):
    """Безопасное сохранение GeoJSON с преобразованием numpy типов.

    Args:
        geojson_data (dict): GeoJSON словарь для сохранения.
        output_file (str): Путь к выходному файлу.

    Returns:
        bool: True, если сохранение успешно, иначе False.
    """
    geojson_safe = numpy_to_python(geojson_data)  # Преобразуем numpy типы
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson_safe, f, indent=2, ensure_ascii=False)
        print(f"Файл успешно сохранен: {output_file}")
        return True
    except Exception as e:
        print(f"Ошибка сохранения файла: {e}")
        return False


# Основная функция программы
def main():
    """Основная функция для выполнения триангуляции и сохранения результата."""
    input_file = "input.geojson"  # Входной файл
    output_file = "triangulation_output.geojson"  # Выходной файл

    print("Чтение исходного файла...")
    gdf = read_geojson(input_file)  # Читаем GeoJSON
    if gdf is None:
        print("Не удалось прочитать файл")
        return

    print(f"Прочитано объектов: {len(gdf)}")  # Выводим количество объектов
    print(f"Типы геометрий: {set(geom.geom_type for geom in gdf.geometry)}")  # Выводим типы геометрий

    print("Извлечение точек, дырок и сегментов...")
    points, holes, segments = extract_points_and_holes(gdf)  # Извлекаем точки, дырки и сегменты
    print(f"Извлечено уникальных точек: {len(points)}")
    print(f"Извлечено дырок: {len(holes)}")
    print(f"Извлечено сегментов: {len(segments)}")

    if len(points) < 3:
        print("Недостаточно точек для триангуляции")
        return

    print("Создание ограниченной триангуляции Делоне...")
    tri = create_constrained_triangulation(points, segments, holes)  # Выполняем триангуляцию
    if tri is None:
        print("Не удалось создать триангуляцию")
        return

    print(f"Создано треугольников: {len(tri['triangles'])}")  # Выводим количество треугольников

    print("Создание выходного GeoJSON...")
    triangulation_geojson = create_triangulation_geojson(points, tri['triangles'])  # Создаем GeoJSON

    print("Сохранение результата...")
    success = safe_save_geojson(triangulation_geojson, output_file)  # Сохраняем результат

    if success:
        print("Визуализация результата...")
        visualize_triangulation(points, tri['triangles'], gdf)  # Визуализируем триангуляцию
        print(f"Триангуляция завершена! Результат сохранен в {output_file}")
    else:
        print("Ошибка при сохранении файла")


# Запуск программы
if __name__ == "__main__":
    main()