# __init__.py для узла ComfyUI PixelArtScaler (Auto-Scale & Crop based on edge-aware detection)
# Следует алгоритму из статьи https://habr.com/ru/articles/930462/
# и оригинальному JS-коду, используя edge-aware detection + runs-based voting + optimal crop.

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
import logging
import time
from collections import Counter
import math
from scipy.ndimage import sobel, generic_filter
from scipy import ndimage

# Настройка логгирования
logging.basicConfig(level=logging.INFO) # Можно изменить на WARNING/INFO/DEBUG
logger = logging.getLogger(__name__)

class PixelArtScaler:
    """
    Узел ComfyUI для пикселизации RGB изображений.
    Алгоритм: Edge-Aware Scale Detection -> Optimal Crop -> Quantization -> Downscale.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Определяет входные параметры узла.
        """
        return {
            "required": {
                "image": ("IMAGE",),  # Входное изображение из ComfyUI (тензор B,H,W,C)
                "max_colors": ("INT", {
                    "default": 16,
                    "min": 2,
                    "max": 256,
                    "step": 1,
                    "display": "number"
                }),
                "cleanup_jaggies": ("BOOLEAN", {"default": True}),
                "downscale_method": (["dominant", "nearest"], {"default": "dominant"}),
                # Основной метод теперь edge-aware
                "scale_detection_method": (["edge_aware"], {"default": "edge_aware"}),
                # Параметры для edge-aware detection
                "ea_tile_grid_size": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Размер сетки для выбора информативных тайлов (NxN)."
                }),
                "ea_min_peak_distance": ("INT", {
                    "default": 5, # Зависит от ожидаемого масштаба
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Минимальное расстояние между пиками в профиле."
                }),
                "ea_peak_prominence_factor": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Минимальная 'поминка' пика как доля от макс. значения профиля."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING") # Возвращает тензор изображения и строку манифеста
    RETURN_NAMES = ("pixel_art_image", "manifest")
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"

    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """Конвертирует тензор ComfyUI (B,H,W,C) в PIL Image (RGB)."""
        try:
            image_tensor = image_tensor.squeeze(0) # -> (H, W, C)
            image_tensor = torch.clamp(image_tensor, 0, 1)
            image_np = (image_tensor.numpy() * 255).astype(np.uint8)

            h, w, c = image_np.shape

            if c == 1:
                pil_image = Image.fromarray(image_np[:, :, 0], mode='L').convert('RGB')
                logger.debug(f"Converted grayscale image ({h}x{w}x{c}) to RGB")
            elif c == 3:
                pil_image = Image.fromarray(image_np, mode='RGB')
                logger.debug(f"Converted RGB image ({h}x{w}x{c})")
            elif c == 4:
                pil_image = Image.fromarray(image_np[:, :, :3], mode='RGB')
                logger.debug(f"Converted RGBA image ({h}x{w}x{c}) to RGB (alpha ignored)")
            else:
                raise ValueError(f"Unsupported number of channels ({c}) in input tensor. Expected 1, 3, or 4.")

            return pil_image
        except Exception as e:
            logger.error(f"Error in tensor_to_pil: {e}")
            raise

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Конвертирует PIL Image (RGB) в тензор ComfyUI (B,H,W,C)."""
        image_np = np.array(pil_image.convert('RGB')).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        return image_tensor

    def edge_aware_detect(self, img: Image.Image, tile_grid_size: int = 3, min_peak_distance: int = 5, peak_prominence_factor: float = 0.1) -> int:
        """
        Определяет масштаб на основе анализа границ (edge-aware detection).
        Аналог edgeAwareDetect -> runsBasedDetect из JS-кода.
        """
        logger.info("Detecting pixel scale using edge-aware method...")
        img_gray = img.convert('L') # Конвертируем в градации серого
        np_gray = np.array(img_gray, dtype=np.float32)

        h, w = np_gray.shape

        # --- 1. Выбор информативных тайлов ---
        # Разбиваем изображение на tile_grid_size x tile_grid_size тайлов
        tile_h = h // tile_grid_size
        tile_w = w // tile_grid_size

        if tile_h <= 1 or tile_w <= 1:
            logger.warning("Image too small for tiling, using whole image.")
            informative_tiles = [np_gray]
        else:
            informative_tiles = []
            tile_variances = []
            for i in range(tile_grid_size):
                for j in range(tile_grid_size):
                    y1, y2 = i * tile_h, (i + 1) * tile_h
                    x1, x2 = j * tile_w, (j + 1) * tile_w
                    tile = np_gray[y1:y2, x1:x2]
                    if tile.size > 0:
                        var = np.var(tile)
                        tile_variances.append((var, tile))

            # Сортируем по дисперсии и берем топ 50% или минимум 1
            tile_variances.sort(key=lambda x: x[0], reverse=True)
            num_tiles_to_use = max(1, len(tile_variances) // 2)
            informative_tiles = [tile for _, tile in tile_variances[:num_tiles_to_use]]
            logger.debug(f"Selected {len(informative_tiles)} informative tiles based on variance.")

        # --- 2. Анализ границ и профилей для каждого тайла ---
        all_distances = []

        for idx, tile in enumerate(informative_tiles):
            tile_h, tile_w = tile.shape
            if tile_h < 10 or tile_w < 10: # Пропустить слишком маленькие тайлы
                continue
            logger.debug(f"Analyzing tile {idx+1}/{len(informative_tiles)} (size {tile_w}x{tile_h})")

            # --- 2a. Фильтр Собеля ---
            # Горизонтальные градиенты (Sobel X)
            sobel_x = sobel(tile, axis=1, mode='constant') # Используем scipy
            # Вертикальные градиенты (Sobel Y)
            sobel_y = sobel(tile, axis=0, mode='constant')

            # Абсолютные значения градиентов
            abs_sobel_x = np.abs(sobel_x)
            abs_sobel_y = np.abs(sobel_y)

            # --- 2b. Создание профилей ---
            profile_x = np.sum(abs_sobel_x, axis=0) # (tile_w,)
            profile_y = np.sum(abs_sobel_y, axis=1) # (tile_h,)

            # --- 2c. Поиск пиков в профилях ---
            def find_peaks_simple(profile, min_dist, prom_factor):
                peaks = []
                if len(profile) < 2:
                    return peaks
                max_val = np.max(profile)
                if max_val <= 0:
                    return peaks
                min_prominence = max_val * prom_factor

                # Простой поиск пиков: точка больше соседей
                for i in range(1, len(profile) - 1):
                    if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                        # Проверка минимальной prominence
                        # (Упрощенная проверка: просто больше порога)
                        if profile[i] >= min_prominence:
                            peaks.append(i)
                
                # Фильтрация по минимальному расстоянию
                if len(peaks) <= 1:
                    return peaks
                filtered_peaks = [peaks[0]]
                for i in range(1, len(peaks)):
                    if peaks[i] - filtered_peaks[-1] >= min_dist:
                        filtered_peaks.append(peaks[i])
                return filtered_peaks

            peaks_x = find_peaks_simple(profile_x, min_peak_distance, peak_prominence_factor)
            peaks_y = find_peaks_simple(profile_y, min_peak_distance, peak_prominence_factor)
            logger.debug(f"  Tile {idx+1}: Found {len(peaks_x)} X-peaks, {len(peaks_y)} Y-peaks")

            # --- 2d. Вычисление расстояний между пиками ---
            def calculate_distances(peaks):
                distances = []
                for i in range(1, len(peaks)):
                    distances.append(peaks[i] - peaks[i-1])
                return distances

            distances_x = calculate_distances(peaks_x)
            distances_y = calculate_distances(peaks_y)
            
            all_distances.extend(distances_x)
            all_distances.extend(distances_y)

        # --- 3. Голосование и определение масштаба ---
        if not all_distances:
            logger.warning("No distances found for scale detection, defaulting to 1.")
            return 1

        # Фильтруем очень маленькие расстояния
        filtered_distances = [d for d in all_distances if d >= 2]
        if not filtered_distances:
            logger.warning("No distances >= 2 found for scale detection, defaulting to 1.")
            return 1

        logger.debug(f"Total distances collected: {len(all_distances)}, filtered: {len(filtered_distances)}")

        # --- 3a. Голосование (Counter) ---
        distance_counts = Counter(filtered_distances)
        most_common_distances = [dist for dist, _ in distance_counts.most_common(20)] # Берем топ-20
        logger.debug(f"Top 20 distances: {sorted(most_common_distances)}")

        if not most_common_distances:
            logger.warning("No common distances found for scale detection, defaulting to 1.")
            return 1

        # --- 3b. Определение масштаба ---
        # Вариант 1: Мода (наиболее частое значение)
        scale_mode = distance_counts.most_common(1)[0][0]
        logger.info(f"Detected pixel scale factor (mode): {scale_mode}")

        # Вариант 2: НОД (GCD) - более надежен, если есть кратные значения
        try:
            if len(most_common_distances) == 1:
                gcd_val = most_common_distances[0]
            else:
                # Начинаем с НОД первых двух
                gcd_val = math.gcd(most_common_distances[0], most_common_distances[1])
                # Продолжаем с остальными
                for dist in most_common_distances[2:]:
                    gcd_val = math.gcd(gcd_val, dist)
                    if gcd_val == 1:
                        break # Если НОД стал 1, дальнейший поиск бессмыслен

            scale_gcd = max(1, gcd_val)
            logger.info(f"Detected pixel scale factor (GCD): {scale_gcd}")

            # Используем GCD как результат, если он разумен
            # Иначе используем моду
            max_reasonable_scale = min(img.width, img.height) // 4 # Ограничение
            if 2 <= scale_gcd <= max_reasonable_scale:
                final_scale = scale_gcd
                logger.info(f"Using GCD scale: {final_scale}")
            elif 2 <= scale_mode <= max_reasonable_scale:
                final_scale = scale_mode
                logger.info(f"GCD was unreliable, using mode scale: {final_scale}")
            else:
                logger.warning(f"Both GCD ({scale_gcd}) and mode ({scale_mode}) scales are unreasonable. Defaulting to 1.")
                final_scale = 1

        except Exception as e:
            logger.warning(f"GCD calculation failed: {e}. Trying mode.")
            max_reasonable_scale = min(img.width, img.height) // 4
            if 2 <= scale_mode <= max_reasonable_scale:
                final_scale = scale_mode
                logger.info(f"Using mode scale after GCD failure: {final_scale}")
            else:
                logger.warning(f"Mode scale ({scale_mode}) is also unreasonable. Defaulting to 1.")
                final_scale = 1

        return final_scale

    def find_optimal_crop(self, img: Image.Image, scale: int) -> tuple[int, int]:
        """
        Находит оптимальное смещение обрезки (x, y) так, чтобы сетка масштаба совпадала с содержимым.
        Упрощённая версия, используя градиенты и профили.
        """
        logger.info(f"Finding optimal crop offset for scale {scale}...")
        if scale <= 1:
            logger.info("Scale is 1 or less, no crop offset needed.")
            return (0, 0)

        img_gray = img.convert('L')
        np_gray = np.array(img_gray, dtype=np.float32)

        # --- Вычисление градиентов Собеля ---
        sobel_x = sobel(np_gray, axis=1, mode='constant')
        sobel_y = sobel(np_gray, axis=0, mode='constant')

        abs_sobel_x = np.abs(sobel_x)
        abs_sobel_y = np.abs(sobel_y)

        # --- Создание профилей ---
        profile_x = np.sum(abs_sobel_x, axis=0) # (W,)
        profile_y = np.sum(abs_sobel_y, axis=1) # (H,)

        # --- Поиск наилучшего смещения ---
        def find_best_offset(profile, scale_candidate):
            if scale_candidate <= 1 or len(profile) < scale_candidate:
                return 0

            max_score = -1
            best_offset = 0
            # Проверяем смещения от 0 до scale-1
            for offset in range(scale_candidate):
                current_score = 0
                # Суммируем значения профиля в позициях offset, offset+scale, offset+2*scale, ...
                idx = offset
                while idx < len(profile):
                    current_score += profile[idx]
                    idx += scale_candidate
                if current_score > max_score:
                    max_score = current_score
                    best_offset = offset
            return best_offset

        best_dx = find_best_offset(profile_x, scale)
        best_dy = find_best_offset(profile_y, scale)

        logger.info(f"Optimal crop offset found: x={best_dx}, y={best_dy}")
        return (best_dx, best_dy)

    def quantize_image(self, img: Image.Image, max_colors: int) -> tuple[Image.Image, list]:
        """Квантование изображения методом K-средних."""
        logger.info(f"Quantizing image to max {max_colors} colors")
        img = img.convert('RGB')

        unique_colors = len(set(img.getdata()))
        logger.info(f"  - Original unique colors: {unique_colors}")

        if unique_colors <= max_colors:
            logger.info("  - Image already has fewer colors than max_colors, skipping quantization.")
            palette = list(set(img.getdata()))
            return img, palette

        np_img = np.array(img)
        h, w, c = np_img.shape
        data = np_img.reshape((-1, 3))

        # Используем n_init='auto' для совместимости
        kmeans = KMeans(n_clusters=max_colors, n_init='auto', random_state=0).fit(data)
        labels = kmeans.labels_
        palette_rgb = kmeans.cluster_centers_.round().astype(np.uint8)

        new_data = palette_rgb[labels]
        new_img_np = new_data.reshape((h, w, c))
        quantized_img = Image.fromarray(new_img_np, mode='RGB')

        final_palette_set = set(quantized_img.getdata())
        final_palette = list(final_palette_set)
        logger.info(f"  - Colors after quantization: {len(final_palette)}")

        return quantized_img, final_palette

    def downscale_by_dominant_color(self, img: Image.Image, scale: int) -> Image.Image:
        """Понижает дискретизацию изображения методом доминирующего цвета."""
        logger.info(f"Downscaling by {scale}x using dominant color method")
        if scale <= 1:
            return img

        orig_w, orig_h = img.size
        target_w = orig_w // scale
        target_h = orig_h // scale

        if target_w <= 0 or target_h <= 0:
            logger.error(f"Target size after downscaling is invalid: {target_w}x{target_h}. Scale factor {scale} might be too large.")
            return Image.new('RGB', (1, 1), (0, 0, 0))

        img_array = np.array(img.convert('RGB'))
        # Предполагается, что img уже обрезан до кратного размера
        # Но на всякий случай обрежем принудительно
        img_array = img_array[:target_h * scale, :target_w * scale]

        # Изменяем форму для векторизованной обработки блоков
        reshaped = img_array.reshape(target_h, scale, target_w, scale, 3)
        reshaped = reshaped.transpose(0, 2, 1, 3, 4)
        block_view = reshaped.reshape(target_h, target_w, -1, 3)

        # Векторизованное нахождение доминирующего цвета
        # block_view.shape = (target_h, target_w, scale*scale, 3)
        # Используем reshape и mode/approach по оси 2
        downsampled_array = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        for i in range(target_h):
            for j in range(target_w):
                 block = block_view[i, j] # (scale*scale, 3)
                 # Находим уникальные цвета и их количество
                 unique_colors, counts = np.unique(block, axis=0, return_counts=True)
                 if len(counts) > 0:
                     dominant_idx = np.argmax(counts)
                     downsampled_array[i, j] = unique_colors[dominant_idx]
                 # Если блок пуст (маловероятно после обрезки), остается черный (0,0,0)

        return Image.fromarray(downsampled_array, mode='RGB')

    def jaggy_cleaner(self, img: Image.Image) -> Image.Image:
        """
        Упрощенная версия удаления "jaggies" для RGB.
        """
        logger.info("Cleaning up jaggies (simplified for RGB)")
        img = img.convert('RGB')
        np_img = np.array(img)
        h, w, c = np_img.shape
        out_img = np_img.copy()

        def get_color(x, y):
            if x < 0 or x >= w or y < 0 or y >= h:
                return np.array([0, 0, 0]) # Черный за краем
            return np_img[y, x]

        def color_distance(c1, c2):
            # Используем более быстрый расчет
            return np.sum((c1.astype(np.int32) - c2.astype(np.int32)) ** 2) # Квадрат расстояния

        # Порог можно сделать параметром, 10000 это квадрат 100
        threshold_sq = 10000 

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                current_color = get_color(x, y)

                neighbors = [
                    get_color(x, y - 1), get_color(x, y + 1),
                    get_color(x - 1, y), get_color(x + 1, y)
                ]

                avg_neighbor_color = np.mean(neighbors, axis=0)
                dist_sq_to_avg = color_distance(current_color, avg_neighbor_color)

                if dist_sq_to_avg > threshold_sq:
                    out_img[y, x] = avg_neighbor_color.astype(np.uint8)

        return Image.fromarray(out_img, mode='RGB')


    def process(self, image: torch.Tensor, max_colors: int, cleanup_jaggies: bool, downscale_method: str, scale_detection_method: str, ea_tile_grid_size: int, ea_min_peak_distance: int, ea_peak_prominence_factor: float):
        """
        Основная функция обработки изображения.
        """
        logger.info("--- Starting Pixel Art Scaling Process (Edge-Aware Auto-Scale & Crop) ---")
        start_time = time.time()

        original_tensor_shape = image.shape
        pil_image = self.tensor_to_pil(image)
        logger.info(f"Original image size: {pil_image.width}x{pil_image.height}")

        # --- 1. Определение масштаба (размера пикселя) ---
        scale = 1
        if scale_detection_method == 'edge_aware':
            try:
                scale = self.edge_aware_detect(
                    pil_image,
                    tile_grid_size=ea_tile_grid_size,
                    min_peak_distance=ea_min_peak_distance,
                    peak_prominence_factor=ea_peak_prominence_factor
                )
                scale = max(1, scale)
                logger.info(f"Auto-detected pixel scale: {scale}")
            except Exception as e:
                logger.error(f"Scale detection failed: {e}. Proceeding with scale=1.")
                scale = 1
        else:
            logger.warning("Unknown scale detection method, defaulting to 'edge_aware'.")
            scale = self.edge_aware_detect(pil_image)

        # --- 1.5. Оптимальная обрезка ---
        crop_x, crop_y = 0, 0
        cropped_pil_image = pil_image
        if scale > 1:
            try:
                crop_x, crop_y = self.find_optimal_crop(pil_image, scale)
                # Вычисляем новые размеры после обрезки
                new_width = ((pil_image.width - crop_x) // scale) * scale
                new_height = ((pil_image.height - crop_y) // scale) * scale

                if new_width > 0 and new_height > 0:
                    box = (crop_x, crop_y, crop_x + new_width, crop_y + new_height)
                    cropped_pil_image = pil_image.crop(box)
                    logger.info(f"Image cropped to align with scale {scale}. New size: {cropped_pil_image.width}x{cropped_pil_image.height} (cropped from {pil_image.width}x{pil_image.height} at offset {crop_x},{crop_y})")
                else:
                     logger.warning(f"Crop resulted in invalid size ({new_width}x{new_height}). Skipping crop.")
                     cropped_pil_image = pil_image
            except Exception as e:
                 logger.error(f"Optimal crop failed: {e}. Proceeding with original image.")
                 cropped_pil_image = pil_image
        else:
            logger.info("Skipping optimal crop (scale <= 1).")


        # --- 2. Квантование цветов ---
        initial_colors = len(set(cropped_pil_image.getdata()))
        logger.info(f"Initial unique colors (after crop): {initial_colors}")
        quantized_image = cropped_pil_image
        if max_colors < 256 and initial_colors > max_colors:
             try:
                 quantized_image, _ = self.quantize_image(cropped_pil_image, max_colors)
                 logger.info(f"Image quantized to max {max_colors} colors.")
             except Exception as e:
                 logger.error(f"Quantization failed: {e}. Proceeding with cropped image.")
        else:
             logger.info("Skipping quantization.")


        # --- 3. Понижающая дискретизация (масштабирование ДО размера пикселя) ---
        final_image = quantized_image
        if scale > 1:
            try:
                if downscale_method == 'dominant':
                    final_image = self.downscale_by_dominant_color(quantized_image, scale)
                elif downscale_method == 'nearest':
                    target_w = max(1, quantized_image.width // scale)
                    target_h = max(1, quantized_image.height // scale)
                    final_image = quantized_image.resize((target_w, target_h), Image.NEAREST)
                    logger.info(f"Downscaled by {scale}x using PIL NEAREST resampling to {target_w}x{target_h}")
                else: # fallback
                    final_image = self.downscale_by_dominant_color(quantized_image, scale)

                logger.info(f"Image downscaled by factor of {scale}. Final size: {final_image.width}x{final_image.height}")
            except Exception as e:
                 logger.error(f"Downscaling failed: {e}. Proceeding with quantized/cropped image.")
                 final_image = quantized_image
        else:
            logger.info("No downscaling applied (scale <= 1).")
            final_image = quantized_image

        # --- 4. Постобработка ---
        if cleanup_jaggies and scale > 1:
            try:
                final_image = self.jaggy_cleaner(final_image)
                logger.info("Jaggy cleanup applied.")
            except Exception as e:
                 logger.error(f"Jaggy cleanup failed: {e}. Proceeding without it.")
        elif cleanup_jaggies:
            logger.info("Skipping jaggy cleanup (no downscaling performed).")


        # --- 5. Создание манифеста ---
        processing_time_ms = round((time.time() - start_time) * 1000)
        final_colors = len(set(final_image.getdata()))
        manifest = {
            "original_size": [original_tensor_shape[2], original_tensor_shape[1]], # W, H from tensor
            "final_size": [final_image.width, final_image.height],
            "processing_steps": {
                "scale_detection": {
                    "method": scale_detection_method,
                    "detected_scale": scale,
                    "params": {
                        "tile_grid_size": ea_tile_grid_size if scale_detection_method == 'edge_aware' else None,
                        "min_peak_distance": ea_min_peak_distance if scale_detection_method == 'edge_aware' else None,
                        "peak_prominence_factor": ea_peak_prominence_factor if scale_detection_method == 'edge_aware' else None,
                    }
                },
                "optimal_crop": {
                     "applied": scale > 1,
                     "offset_x": crop_x if scale > 1 else None,
                     "offset_y": crop_y if scale > 1 else None,
                },
                "color_quantization": {
                    "max_colors": max_colors,
                    "initial_colors": initial_colors,
                    "final_colors": final_colors,
                },
                "downscaling": {
                    "method": downscale_method,
                    "scale_factor": scale,
                    "applied": scale > 1,
                },
                "cleanup": {
                    "jaggy": cleanup_jaggies,
                },
            },
            "processing_time_ms": processing_time_ms,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        logger.info(f"Final manifest: {manifest}")
        logger.info("--- Pixel Art Scaling Process Completed ---")

        output_tensor = self.pil_to_tensor(final_image)
        return (output_tensor, str(manifest))

# Маппинг имен узлов для ComfyUI
NODE_CLASS_MAPPINGS = {
    "PixelArtScaler": PixelArtScaler,
}

# Маппинг отображаемых имен узлов
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtScaler": "Pixel Art Scaler (Edge-Aware Auto-Scale)",
}
