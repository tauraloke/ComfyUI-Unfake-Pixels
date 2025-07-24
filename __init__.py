# __init__.py для узла ComfyUI PixelArtScaler (для RGB изображений)

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
import logging
import time
from collections import Counter
import math
from scipy import stats

# Настройка логгирования
logging.basicConfig(level=logging.INFO)  # Можно изменить на WARNING/INFO/DEBUG
logger = logging.getLogger(__name__)


class PixelArtScaler:
    """
    Узел ComfyUI для пикселизации RGB изображений.
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
                "max_colors": (
                    "INT",
                    {
                        "default": 16,
                        "min": 2,
                        "max": 256,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "cleanup_jaggies": ("BOOLEAN", {"default": True}),
                "downscale_method": (["dominant", "nearest"], {"default": "dominant"}),
                "scale_detection_method": (["runs", "none"], {"default": "runs"}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "STRING",
    )  # Возвращает тензор изображения и строку манифеста
    RETURN_NAMES = ("pixel_art_image", "manifest")
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"

    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """Конвертирует тензор ComfyUI (B,H,W,C) в PIL Image (RGB)."""
        try:
            # Предполагаем batch size = 1 и убираем его
            image_tensor = image_tensor.squeeze(0)  # -> (H, W, C)

            # Убеждаемся, что значения в [0, 1]
            image_tensor = torch.clamp(image_tensor, 0, 1)

            # Конвертируем в numpy (H, W, C) и умножаем на 255
            image_np = (image_tensor.numpy() * 255).astype(np.uint8)

            # Проверка количества каналов
            h, w, c = image_np.shape

            if c == 1:  # Grayscale
                pil_image = Image.fromarray(image_np[:, :, 0], mode="L").convert("RGB")
                logger.debug(f"Converted grayscale image ({h}x{w}x{c}) to RGB")
            elif c == 3:  # RGB
                pil_image = Image.fromarray(image_np, mode="RGB")
                logger.debug(f"Converted RGB image ({h}x{w}x{c})")
            elif c == 4:  # RGBA - преобразуем в RGB, игнорируя альфа
                pil_image = Image.fromarray(image_np[:, :, :3], mode="RGB")
                logger.debug(
                    f"Converted RGBA image ({h}x{w}x{c}) to RGB (alpha ignored)"
                )
            else:
                raise ValueError(
                    f"Unsupported number of channels ({c}) in input tensor. Expected 1, 3, or 4."
                )

            return pil_image

        except Exception as e:
            logger.error(f"Error in tensor_to_pil: {e}")
            raise  # Перебрасываем исключение для отображения в ComfyUI

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Конвертирует PIL Image (RGB) в тензор ComfyUI (B,H,W,C)."""
        # Конвертируем в numpy массив (H, W, 3)
        image_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        # Добавляем batch dimension -> (1, H, W, 3)
        image_tensor = torch.from_numpy(image_np)[None,]
        return image_tensor

    def runs_based_detect(self, img: Image.Image) -> int:
        """Определяет масштаб на основе длин серий пикселей (оптимизированная версия)."""
        logger.info("Detecting scale using runs-based method (optimized)")
        img = img.convert("RGB")
        np_img = np.array(img)

        all_run_lens = []

        def scan_runs_vectorized(is_horizontal: bool):
            if is_horizontal:
                # np_img.shape = (H, W, 3)
                # Сравниваем соседние пиксели по оси W (горизонтально)
                # np_img[:, :-1, :] - все пиксели, кроме последнего в строке
                # np_img[:, 1:, :] - все пиксели, кроме первого в строке
                diff = np_img[:, :-1, :] != np_img[:, 1:, :]  # (H, W-1, 3) bool
                # Если хоть один из каналов R,G,B отличается, считаем это сменой цвета
                color_changes = np.any(diff, axis=2)  # (H, W-1) bool
            else:
                # Сравниваем соседние пиксели по оси H (вертикально)
                diff = np_img[:-1, :, :] != np_img[1:, :, :]  # (H-1, W, 3) bool
                color_changes = np.any(diff, axis=2)  # (H-1, W) bool

            # Добавляем True в начало каждой строки/столбца, чтобы правильно считать длины серий
            # np.diff(np.where(...)) - стандартный способ найти длины серий
            if is_horizontal:
                # Для каждой строки
                for i in range(color_changes.shape[0]):
                    # Находим индексы, где цвет меняется (True)
                    change_indices = np.concatenate(
                        (
                            [0],
                            np.where(color_changes[i])[0] + 1,
                            [color_changes.shape[1]],
                        )
                    )
                    # Вычисляем длины серий
                    run_lengths = np.diff(change_indices)
                    all_run_lens.extend(run_lengths)
            else:
                # Для каждого столбца
                for j in range(color_changes.shape[1]):
                    change_indices = np.concatenate(
                        (
                            [0],
                            np.where(color_changes[:, j])[0] + 1,
                            [color_changes.shape[0]],
                        )
                    )
                    run_lengths = np.diff(change_indices)
                    all_run_lens.extend(run_lengths)

        scan_runs_vectorized(True)  # Горизонтальные серии
        scan_runs_vectorized(False)  # Вертикальные серии

        if not all_run_lens:
            logger.warning("No runs found for scale detection, defaulting to 1.")
            return 1

        # Остальная логика GCD остается прежней
        run_counts = Counter(all_run_lens)
        most_common_runs = [run_len for run_len, _ in run_counts.most_common(10)]

        if not most_common_runs:
            logger.warning("No common runs found for scale detection, defaulting to 1.")
            return 1

        try:
            gcd_val = most_common_runs[0]
            for run_len in most_common_runs[1:]:
                gcd_val = math.gcd(gcd_val, run_len)
                if gcd_val == 1:
                    break
            scale = max(1, gcd_val)
            logger.info(f"Detected scale factor: {scale} (from GCD of top runs)")
            return scale
        except Exception as e:
            logger.warning(f"GCD calculation failed: {e}. Defaulting to 1.")
            return 1

    def quantize_image(
        self, img: Image.Image, max_colors: int
    ) -> tuple[Image.Image, list]:
        """Квантование изображения методом K-средних."""
        logger.info(f"Quantizing image to max {max_colors} colors")
        img = img.convert("RGB")

        unique_colors = len(set(img.getdata()))
        logger.info(f"  - Original unique colors: {unique_colors}")

        if unique_colors <= max_colors:
            logger.info(
                "  - Image already has fewer colors than max_colors, skipping quantization."
            )
            palette = list(set(img.getdata()))
            return img, palette

        # Конвертируем в numpy массив
        np_img = np.array(img)
        h, w, c = np_img.shape

        # Изменяем форму для кластеризации (N, 3) - R, G, B
        data = np_img.reshape((-1, 3))

        # Применяем KMeans
        kmeans = KMeans(n_clusters=max_colors, n_init=10, random_state=0).fit(data)
        labels = kmeans.labels_
        palette_rgb = kmeans.cluster_centers_.round().astype(np.uint8)  # (N, 3)

        # Создаем новое изображение
        new_data = palette_rgb[labels]
        new_img_np = new_data.reshape((h, w, c))
        quantized_img = Image.fromarray(new_img_np, mode="RGB")

        # Извлекаем палитру
        final_palette_set = set(quantized_img.getdata())
        final_palette = list(final_palette_set)
        logger.info(f"  - Colors after quantization: {len(final_palette)}")

        return quantized_img, final_palette

    def downscale_by_dominant_color(self, img: Image.Image, scale: int) -> Image.Image:
        """Понижает дискретизацию изображения методом доминирующего цвета (оптимизированная версия)."""
        logger.info(f"Downscaling by {scale}x using dominant color method (optimized)")
        if scale <= 1:
            return img

        orig_w, orig_h = img.size
        new_w = orig_w // scale
        new_h = orig_h // scale

        if new_w <= 0 or new_h <= 0:
            raise ValueError(
                f"Scale factor {scale} is too large for image size {orig_w}x{orig_h}"
            )

        img_array = np.array(img.convert("RGB"))  # Убедимся, что RGB
        # Обрезаем изображение, чтобы размеры делились на scale
        img_array = img_array[: new_h * scale, : new_w * scale]

        # Изменяем форму: (new_h, scale, new_w, scale, 3)
        reshaped = img_array.reshape(new_h, scale, new_w, scale, 3)
        # Меняем оси: (new_h, new_w, scale, scale, 3)
        reshaped = reshaped.transpose(0, 2, 1, 3, 4)
        # Снова изменяем форму: (new_h, new_w, scale*scale, 3)
        block_view = reshaped.reshape(new_h, new_w, -1, 3)

        # Находим доминирующий цвет для каждого блока
        # block_view.shape = (new_h, new_w, scale*scale, 3)
        # stats.mode по оси 2 (scale*scale)
        # mode_result = stats.mode(block_view, axis=2, keepdims=False)
        # downsampled_array = mode_result.mode # (new_h, new_w, 3)

        # stats.mode может быть медленным для больших блоков или если нет четкого режима.
        # Альтернатива: найти уникальные цвета и их количество в каждом блоке.
        downsampled_array = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for i in range(new_h):
            for j in range(new_w):
                # block_view[i, j] is (scale*scale, 3)
                block = block_view[i, j]
                # Конвертируем цвета в кортежи для подсчета
                # Это всё ещё может быть узким местом для очень больших scale
                colors_tuples = [tuple(color) for color in block]
                if colors_tuples:
                    color_counts = Counter(colors_tuples)
                    dominant_color = color_counts.most_common(1)[0][0]
                    downsampled_array[i, j] = dominant_color

        return Image.fromarray(downsampled_array, mode="RGB")

    def jaggy_cleaner(self, img: Image.Image) -> Image.Image:
        """
        Упрощенная версия удаления "jaggies" для RGB.
        Так как нет прозрачности, будем проверять контрастность цветов.
        """
        logger.info("Cleaning up jaggies (simplified for RGB)")
        img = img.convert("RGB")
        np_img = np.array(img)
        h, w, c = np_img.shape
        out_img = np_img.copy()

        def get_color(x, y):
            if x < 0 or x >= w or y < 0 or y >= h:
                return np.array([0, 0, 0])  # Черный по краям
            return np_img[y, x]

        def color_distance(c1, c2):
            # Простое евклидово расстояние
            return np.sqrt(np.sum((c1.astype(np.float32) - c2.astype(np.float32)) ** 2))

        # Проходим по пикселям, игнорируя края
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                current_color = get_color(x, y)

                # Получаем цвета соседей
                N = get_color(x, y - 1)
                S = get_color(x, y + 1)
                E = get_color(x + 1, y)
                W = get_color(x - 1, y)

                # Проверяем, сильно ли отличается от соседей
                # Это упрощенная эвристика
                neighbors = [N, S, E, W]
                avg_neighbor_color = np.mean(neighbors, axis=0)

                dist_to_avg = color_distance(current_color, avg_neighbor_color)

                # Если пиксель сильно отличается от среднего соседа, возможно, это шум
                # Этот порог можно настраивать
                if dist_to_avg > 100:
                    # Заменяем на средний цвет соседей
                    out_img[y, x] = avg_neighbor_color.astype(np.uint8)

        return Image.fromarray(out_img, mode="RGB")

    def process(
        self,
        image: torch.Tensor,
        max_colors: int,
        cleanup_jaggies: bool,
        downscale_method: str,
        scale_detection_method: str,
    ):
        """
        Основная функция обработки изображения (для RGB).
        """
        logger.info("--- Starting Pixel Art Scaling Process (RGB) ---")
        start_time = time.time()

        # --- 1. Подготовка ---
        original_tensor_shape = image.shape
        pil_image = self.tensor_to_pil(image)
        logger.info(f"Original image size: {pil_image.width}x{pil_image.height}")

        # --- 2. Определение масштаба ---
        scale = 1
        if scale_detection_method == "runs":
            try:
                scale = self.runs_based_detect(pil_image)
                scale = max(1, scale)
                logger.info(f"Detected scale: {scale}")
            except Exception as e:
                logger.error(f"Scale detection failed: {e}. Proceeding with scale=1.")
                scale = 1
        else:
            logger.info("Scale detection skipped (method='none').")

        # --- 3. Квантование цветов ---
        initial_colors = len(set(pil_image.getdata()))
        logger.info(f"Initial unique colors: {initial_colors}")
        if max_colors < 256 and initial_colors > max_colors:
            try:
                pil_image, _ = self.quantize_image(pil_image, max_colors)
                logger.info(f"Image quantized to max {max_colors} colors.")
            except Exception as e:
                logger.error(
                    f"Quantization failed: {e}. Proceeding with original image."
                )
        else:
            logger.info("Skipping quantization.")

        # --- 4. Понижающая дискретизация ---
        if scale > 1:
            try:
                if downscale_method == "dominant":
                    pil_image = self.downscale_by_dominant_color(pil_image, scale)
                elif downscale_method == "nearest":
                    new_size = (pil_image.width // scale, pil_image.height // scale)
                    pil_image = pil_image.resize(new_size, Image.NEAREST)
                    logger.info(
                        f"Downscaled by {scale}x using PIL NEAREST resampling to {new_size[0]}x{new_size[1]}"
                    )
                else:
                    pil_image = self.downscale_by_dominant_color(pil_image, scale)
            except Exception as e:
                logger.error(
                    f"Downscaling failed: {e}. Proceeding with quantized image."
                )
        else:
            logger.info("No downscaling applied (scale <= 1).")

        # --- 5. Постобработка ---
        if cleanup_jaggies:
            try:
                pil_image = self.jaggy_cleaner(pil_image)
                logger.info("Jaggy cleanup applied.")
            except Exception as e:
                logger.error(f"Jaggy cleanup failed: {e}. Proceeding without it.")

        # --- 6. Создание манифеста ---
        processing_time_ms = round((time.time() - start_time) * 1000)
        final_colors = len(set(pil_image.getdata()))
        manifest = {
            "original_size": [
                original_tensor_shape[2],
                original_tensor_shape[1],
            ],  # W, H from tensor
            "final_size": [pil_image.width, pil_image.height],
            "processing_steps": {
                "scale_detection": {
                    "method": scale_detection_method,
                    "detected_scale": (
                        scale if scale_detection_method != "none" else None
                    ),
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
                # alpha_processing удален
            },
            "processing_time_ms": processing_time_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        logger.info(f"Final manifest: {manifest}")
        logger.info("--- Pixel Art Scaling Process Completed ---")

        # --- 7. Возврат результата ---
        output_tensor = self.pil_to_tensor(pil_image)
        return (output_tensor, str(manifest))


# Маппинг имен узлов для ComfyUI
NODE_CLASS_MAPPINGS = {
    "PixelArtScaler": PixelArtScaler,
}

# Маппинг отображаемых имен узлов
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtScaler": "Pixel Art Scaler (RGB)",
}
