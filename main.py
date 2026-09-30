import av
import time
import os
import numpy as np
import cv2
import socket
import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime, timedelta
import requests
import pytz
from time import sleep
import logging
import fcntl
import threading
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Настройка логирования с ротацией
log_handlers = [
    RotatingFileHandler('camera_bot.log', maxBytes=5 * 1024 * 1024, backupCount=5),
    logging.StreamHandler()
]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)

@dataclass
class Branch:
    
    name: str
    base_url: str
    cameras_to_check: Set[int]

# Dictionary of branches with their base RTSP URLs
BRANCHES = {
    "40 лет": "rtsp://admin:6J5W-wck@87.239.29.247:554/cam/realmonitor?channel={}&subtype=00",
    "Алфавит": "rtsp://admin:6J5W-wck@82.149.223.218:554/cam/realmonitor?channel={}&subtype=00",
    "Бпарк": "rtsp://admin:6J5W-wck@185.43.198.90:554/cam/realmonitor?channel={}&subtype=00",
    "Ватутинки": "rtsp://admin:6J5W-wck@88.210.49.169:554/cam/realmonitor?channel={}&subtype=00",
    "Восток": "rtsp://admin:6J5W-wck@62.217.187.100:554/cam/realmonitor?channel={}&subtype=00",
    "Испания": "rtsp://admin:6J5W-wck@46.242.38.131:554/cam/realmonitor?channel={}&subtype=00",
    "Коммунарка": "rtsp://admin:6J5W-wck@94.102.123.173:554/cam/realmonitor?channel={}&subtype=00",
    "Луга": "rtsp://admin:6J5W-wck@185.136.76.134:554/cam/realmonitor?channel={}&subtype=00",
    "Лучи": "rtsp://admin:6J5W-wck@83.237.37.58:554/cam/realmonitor?channel={}&subtype=00",
    "Московский": "rtsp://admin:6J5W-wck@5.228.14.150:554/cam/realmonitor?channel={}&subtype=00",
    "Подольск": "rtsp://admin:6J5W-wck@95.73.83.249:554/cam/realmonitor?channel={}&subtype=00",
    "Скандинавия": "rtsp://admin:6J5W-wck@178.140.172.225:554/cam/realmonitor?channel={}&subtype=00",
    "Щербинка": "rtsp://admin:6J5W-wck@87.239.29.249:554/cam/realmonitor?channel={}&subtype=00",
    "Эталон": "rtsp://admin:6J5W-wck@87.239.29.42:554/cam/realmonitor?channel={}&subtype=00",
    "Ясенево": "rtsp://admin:6J5W-wck@95.143.218.198:554/cam/realmonitor?channel={}&subtype=00"
}

# Camera availability matrix (✅ = check, - = don't check)
CAMERA_MATRIX = {
    "40 лет": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16},
    # Алфавит — NVR 82.149.223.218; «Бпарк» в BRANCHES — другой IP, 16 каналов на площадке
    "Алфавит": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    "Бпарк": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    "Ватутинки": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    "Восток": {1, 2, 3, 4, 5, 6, 7, 8},
    "Испания": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    "Коммунарка": {1, 2, 3, 4, 5, 6, 7, 8, 9},
    "Луга": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    "Лучи": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    "Московский": {1, 2, 3, 4, 5, 6, 7, 8, 9},
    "Подольск": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
    "Скандинавия": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
    "Щербинка": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    "Эталон": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    "Ясенево": {1, 2, 3, 4, 5, 6, 7, 8}
}

# Несколько филиалов могут указывать на один IP NVR (напр. Алфавит и Бпарк).
# Параллельные RTSP-сессии на один регистратор дают таймауты/401 и ложные Offline.
_nvr_host_locks: Dict[str, threading.Lock] = {}
_nvr_host_locks_guard = threading.Lock()
OFFLINE_CONFIRMATION_ROUNDS = 1
OFFLINE_CONFIRMATION_DELAY_S = 10


def _rtsp_host_from_base_url(base_url: str) -> str:
    m = re.search(r"@([^:]+):", base_url)
    return m.group(1) if m else base_url


def _lock_for_rtsp_host(base_url: str) -> threading.Lock:
    host = _rtsp_host_from_base_url(base_url)
    with _nvr_host_locks_guard:
        if host not in _nvr_host_locks:
            _nvr_host_locks[host] = threading.Lock()
        return _nvr_host_locks[host]


def check_tcp_connection(ip: str, port: int, timeout: int = 1) -> bool:
    """
    Проверяет TCP-соединение с камерой
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def check_tcp_connection_with_retries(
    ip: str,
    port: int,
    timeout: int = 1,
    attempts: int = 3,
    delay_s: float = 0.5,
) -> bool:
    attempts = max(1, int(attempts))
    for i in range(attempts):
        if check_tcp_connection(ip, port, timeout=timeout):
            return True
        if i < attempts - 1:
            time.sleep(delay_s)
    return False

def is_black_frame(frame, threshold: float = 0.1) -> bool:
    """
    Check if a frame is black or very dark.
    
    Args:
        frame: PyAV video frame
        threshold: Threshold for considering a frame black (0-1)
                  Lower values mean more strict checking
    
    Returns:
        bool: True if frame is black, False otherwise
    """
    # Convert frame to numpy array
    img = frame.to_ndarray(format='rgb24')
    
    # Calculate mean brightness
    mean_brightness = np.mean(img) / 255.0
    
    # If mean brightness is below threshold, consider it black
    return mean_brightness < threshold


def is_distorted_frame(
    frame,
    laplacian_threshold: float = 800.0,
    periodicity_ratio: float = 60.0,
    max_saturation: float = 0.05,
    dark_brightness_threshold: float = 0.15,
    dark_lap_threshold: float = 1200.0,
) -> bool:
    """
    Проверяет, искажён ли кадр (помехи, «плывущее» изображение).
    
    Типы искажений:
    1) ЧБ + периодичность + низкая резкость (горизонтальные полосы, «плывёт»)
    2) Тёмный кадр при помехах (очень низкая яркость + низкая резкость)
    
    Returns:
        bool: True если кадр выглядит искажённым/нерабочим
    """
    img = frame.to_ndarray(format='rgb24')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray) / 255.0

    # Тёмный кадр при помехах: очень низкая яркость + нет чёткой картинки
    if brightness < dark_brightness_threshold and laplacian_var < dark_lap_threshold:
        return True

    low_sharpness = laplacian_var < laplacian_threshold
    sat_mean = np.mean(hsv[:, :, 1]) / 255.0
    is_grayscale = sat_mean < max_saturation

    row_means = np.mean(gray, axis=1).astype(np.float64)
    fft = np.fft.rfft(row_means - np.mean(row_means))
    magnitude = np.abs(fft)
    if len(magnitude) < 2:
        return low_sharpness and is_grayscale
    mag_no_dc = magnitude[1:]
    if np.mean(mag_no_dc) < 1e-6:
        return False
    ratio = np.max(mag_no_dc) / np.mean(mag_no_dc)
    strong_periodicity = ratio > periodicity_ratio

    return low_sharpness and strong_periodicity and is_grayscale

def check_camera(
    url: str,
    timeout: int = 5,
    max_attempts: int = 3,
    num_frames_to_check: int = 3,
    warmup_frames: int = 0,
) -> Tuple[bool, Dict[str, any]]:
    """
    Check if a camera is accessible and analyze image quality for interference.
    Returns tuple of (is_accessible, quality_info).
    
    Args:
        url: RTSP URL камеры
        timeout: таймаут в секундах для каждой попытки
        max_attempts: максимальное количество попыток подключения
        num_frames_to_check: количество кадров для проверки
    """
    # Увеличиваем таймауты и число попыток для Подольска, Эталона и Скандинавии,
    # а также проверяем больше кадров после короткого прогрева
    is_podolsk = "95.73.83.249" in url
    is_etalon = "87.239.29.42" in url
    is_skandinavia = "178.140.172.225" in url

    if is_podolsk:
        timeout = max(timeout, 30)
        max_attempts = max(max_attempts, 6)
        num_frames_to_check = max(num_frames_to_check, 8)
        warmup_frames = max(warmup_frames, 5)
        logging.info(f"Подольск: увеличенный таймаут {timeout}s, попыток {max_attempts}")
    elif is_etalon:
        timeout = max(timeout, 25)
        max_attempts = max(max_attempts, 5)
        num_frames_to_check = max(num_frames_to_check, 8)
        warmup_frames = max(warmup_frames, 5)
        logging.info(f"Эталон: увеличенный таймаут {timeout}s, попыток {max_attempts}")
    elif is_skandinavia:
        timeout = max(timeout, 25)
        max_attempts = max(max_attempts, 5)
        num_frames_to_check = max(num_frames_to_check, 8)
        warmup_frames = max(warmup_frames, 5)
        logging.info(f"Скандинавия: увеличенный таймаут {timeout}s, попыток {max_attempts}")
    
    # Для Подольска и Скандинавии сначала проверяем TCP-соединение.
    # Успешный TCP не означает, что камера работает: Online ставим только
    # после успешного открытия RTSP-потока и получения кадров.
    if is_podolsk or is_skandinavia:
        try:
            # Извлекаем IP и порт из URL
            ip_match = re.search(r'@([^:]+):(\d+)', url)
            if ip_match:
                ip = ip_match.group(1)
                port = int(ip_match.group(2))
                branch_name = "Подольск" if is_podolsk else "Скандинавия"
                logging.info(f"{branch_name}: проверяем TCP-соединение к {ip}:{port}")
                tcp_ok = check_tcp_connection_with_retries(ip, port, timeout=5, attempts=3, delay_s=0.5)
                if not tcp_ok:
                    logging.error(f"{branch_name}: TCP-соединение к {ip}:{port} не удалось (3 попытки)")
                    return False, {'status': 'tcp_connection_failed', 'ip': ip, 'port': port}
                else:
                    logging.info(f"{branch_name}: TCP-соединение к {ip}:{port} успешно")
        except Exception as e:
            branch_name = "Подольск" if is_podolsk else "Скандинавия"
            logging.warning(f"{branch_name}: ошибка при проверке TCP-соединения: {str(e)}")
    
    for attempt in range(max_attempts):
        try:
            # Устанавливаем параметры для RTSP
            options = {
                'rtsp_transport': 'tcp',
                'stimeout': str(timeout * 1000000),  # таймаут сокета (handshake) в мкс
                'rw_timeout': str(timeout * 1000000),  # таймаут чтения/записи в мкс
                'buffer_size': '1024000',
                'reorder_queue_size': '0',
                'max_delay': '500000'
            }
            
            if is_podolsk and attempt == 0:
                logging.info(f"Подольск: попытка подключения к {url}")
            elif is_etalon and attempt == 0:
                logging.info(f"Эталон: попытка подключения к {url}")
            elif is_skandinavia and attempt == 0:
                logging.info(f"Скандинавия: попытка подключения к {url}")
            
            # Пробуем открыть поток
            with av.open(url, options=options, timeout=timeout) as container:
                # Прогреваем поток и проверяем несколько кадров
                black_frames_count = 0
                distorted_frames_count = 0
                analysis_frames_checked = 0
                total_frames_seen = 0

                for frame in container.decode(video=0):
                    total_frames_seen += 1
                    # пропускаем первые warmup_frames кадров
                    if total_frames_seen <= warmup_frames:
                        continue

                    if analysis_frames_checked >= num_frames_to_check:
                        break

                    if is_black_frame(frame):
                        black_frames_count += 1
                    elif is_distorted_frame(frame):
                        distorted_frames_count += 1

                    analysis_frames_checked += 1
                
                frames_checked = analysis_frames_checked
                bad_frames_count = black_frames_count + distorted_frames_count

                # Если не удалось получить ни одного кадра для анализа, считаем камеру оффлайн
                # (подключение могло установиться, но поток не отдает кадры)
                if total_frames_seen == 0 or frames_checked == 0:
                    msg = f"Камера {url}: не удалось получить кадры (seen={total_frames_seen}, checked={frames_checked}) - НЕ РАБОТАЕТ"
                    if is_podolsk:
                        logging.warning(f"Подольск: {msg}")
                    elif is_etalon:
                        logging.warning(f"Эталон: {msg}")
                    elif is_skandinavia:
                        logging.warning(f"Скандинавия: {msg}")
                    else:
                        logging.warning(msg)
                    return False, {'status': 'no_frames', 'frames_seen': total_frames_seen, 'frames_checked': frames_checked}
                
                # Если все проверенные кадры тёмные, поток нельзя считать рабочим.
                if frames_checked > 0 and black_frames_count == frames_checked:
                    if is_podolsk:
                        logging.warning(f"Подольск: камера {url} показывает только тёмные кадры - НЕ РАБОТАЕТ")
                    elif is_etalon:
                        logging.warning(f"Эталон: камера {url} показывает только тёмные кадры - НЕ РАБОТАЕТ")
                    elif is_skandinavia:
                        logging.warning(f"Скандинавия: камера {url} показывает только тёмные кадры - НЕ РАБОТАЕТ")
                    else:
                        logging.warning(f"Камера {url} показывает только тёмные кадры - НЕ РАБОТАЕТ")
                    return False, {'status': 'all_black', 'frames_checked': frames_checked, 'black_frames': black_frames_count}
                
                # Если большинство кадров черные (>80%), поток нельзя считать рабочим.
                elif frames_checked > 0 and (black_frames_count / frames_checked) > 0.8:
                    if is_podolsk:
                        logging.warning(f"Подольск: камера {url} показывает в основном тёмные кадры ({black_frames_count}/{frames_checked}) - НЕ РАБОТАЕТ")
                    elif is_etalon:
                        logging.warning(f"Эталон: камера {url} показывает в основном тёмные кадры ({black_frames_count}/{frames_checked}) - НЕ РАБОТАЕТ")
                    elif is_skandinavia:
                        logging.warning(f"Скандинавия: камера {url} показывает в основном тёмные кадры ({black_frames_count}/{frames_checked}) - НЕ РАБОТАЕТ")
                    else:
                        logging.warning(f"Камера {url} показывает в основном тёмные кадры ({black_frames_count}/{frames_checked}) - НЕ РАБОТАЕТ")
                    return False, {'status': 'mostly_black', 'frames_checked': frames_checked, 'black_frames': black_frames_count}
                
                # Если большинство кадров искажены или тёмные (помехи, «плывёт»), поток нельзя считать рабочим.
                elif frames_checked > 0 and (bad_frames_count / frames_checked) > 0.8:
                    msg = f"искажённые/тёмные кадры ({bad_frames_count}/{frames_checked}, из них черных {black_frames_count}, искажённых {distorted_frames_count})"
                    if is_podolsk:
                        logging.warning(f"Подольск: камера {url} {msg} - НЕ РАБОТАЕТ")
                    elif is_etalon:
                        logging.warning(f"Эталон: камера {url} {msg} - НЕ РАБОТАЕТ")
                    elif is_skandinavia:
                        logging.warning(f"Скандинавия: камера {url} {msg} - НЕ РАБОТАЕТ")
                    else:
                        logging.warning(f"Камера {url} {msg} - НЕ РАБОТАЕТ")
                    return False, {'status': 'mostly_distorted', 'frames_checked': frames_checked, 'black_frames': black_frames_count, 'distorted_frames': distorted_frames_count}
                
                
                if attempt > 0:
                    success_msg = f"Камера {url} заработала с {attempt + 1} попытки"
                    if is_podolsk:
                        logging.info(f"Подольск: {success_msg}")
                    elif is_etalon:
                        logging.info(f"Эталон: {success_msg}")
                    elif is_skandinavia:
                        logging.info(f"Скандинавия: {success_msg}")
                    else:
                        print(success_msg)
                
                # Добавляем логирование для отладки
                if black_frames_count > 0 or distorted_frames_count > 0:
                    debug_msg = f"Камера {url}: {black_frames_count}/{frames_checked} черных, {distorted_frames_count}/{frames_checked} искажённых"
                    if is_podolsk:
                        logging.info(f"Подольск: {debug_msg}")
                    elif is_etalon:
                        logging.info(f"Эталон: {debug_msg}")
                    elif is_skandinavia:
                        logging.info(f"Скандинавия: {debug_msg}")
                    else:
                        logging.info(debug_msg)
                
                quality_info = {
                    'status': 'normal',
                    'frames_checked': frames_checked,
                    'black_frames': black_frames_count,
                    'distorted_frames': distorted_frames_count
                }
                
                if is_podolsk:
                    logging.info(f"Подольск: камера {url} работает нормально")
                elif is_etalon:
                    logging.info(f"Эталон: камера {url} работает нормально")
                elif is_skandinavia:
                    logging.info(f"Скандинавия: камера {url} работает нормально")
                
                return True, quality_info
                
        except av.error.FFmpegError as e:
            message = str(e).lower()
            should_retry = (
                'timeout' in message
                or 'end of file' in message
                or 'connection refused' in message
                or 'input/output error' in message
                or 'operation not permitted' in message
                or 'immediate exit requested' in message
                or '401' in message
                or 'unauthorized' in message
                or 'authorization failed' in message
                or 'invalid data' in message
            )

            if attempt < max_attempts - 1 and should_retry:
                retry_msg = f"Ошибка FFmpeg, повторяем: {url}, попытка {attempt + 1} из {max_attempts}: {str(e)}"
                if is_podolsk:
                    logging.warning(f"Подольск: {retry_msg}")
                elif is_etalon:
                    logging.warning(f"Эталон: {retry_msg}")
                elif is_skandinavia:
                    logging.warning(f"Скандинавия: {retry_msg}")
                else:
                    print(retry_msg)
                time.sleep(1)
                continue

            # финальная ошибка без возможности повторить
            if (
                '401' in message
                or 'unauthorized' in message
                or 'authorization failed' in message
            ):
                logging.warning(
                    "RTSP 401 с хоста, где запущен бот: если VLC с другого ПК открывает тот же URL, "
                    "частая причина — белый список IP / фильтр клиентов на NVR (серверу «не тот» источник). "
                    "Проверьте сетевые ограничения регистратора для RTSP или тот же URL с этой машины (ffprobe в контейнере)."
                )
            error_msg = f"Ошибка FFmpeg: {url} - {str(e)}"
            if is_podolsk:
                logging.error(f"Подольск: {error_msg}")
            elif is_etalon:
                logging.error(f"Эталон: {error_msg}")
            elif is_skandinavia:
                logging.error(f"Скандинавия: {error_msg}")
            else:
                print(error_msg)
            return False, {'status': 'ffmpeg_error', 'error': str(e)}
        except Exception as e:
            error_msg = f"Ошибка при проверке камеры {url}: {str(e)}"
            if is_podolsk:
                logging.error(f"Подольск: {error_msg}")
            elif is_etalon:
                logging.error(f"Эталон: {error_msg}")
            elif is_skandinavia:
                logging.error(f"Скандинавия: {error_msg}")
            else:
                print(error_msg)
            return False, {'status': 'exception', 'error': str(e)}
    
    return False, {'status': 'max_attempts_exceeded'}

def check_branch_cameras(branch_name: str, base_url: str, cameras_to_check: Set[int]) -> Dict[int, Dict[str, any]]:
    """
    Check all cameras for a specific branch.
    Returns a dictionary with camera numbers as keys and their status/quality info as values.
    """
    results: Dict[int, Dict[str, any]] = {}
    with _lock_for_rtsp_host(base_url):
        is_podolsk = "95.73.83.249" in base_url
        is_etalon = "87.239.29.42" in base_url
        is_skandinavia = "178.140.172.225" in base_url

        if is_podolsk:
            logging.info(f"Подольск: начинаем проверку {len(cameras_to_check)} камер")
        elif is_etalon:
            logging.info(f"Эталон: начинаем проверку {len(cameras_to_check)} камер")
        elif is_skandinavia:
            logging.info(f"Скандинавия: начинаем проверку {len(cameras_to_check)} камер")

        for camera_num in cameras_to_check:
            url = base_url.format(camera_num)
            if is_podolsk:
                logging.info(f"Подольск: проверяем камеру {camera_num}")
            elif is_etalon:
                logging.info(f"Эталон: проверяем камеру {camera_num}")
            elif is_skandinavia:
                logging.info(f"Скандинавия: проверяем камеру {camera_num}")

            is_accessible, quality_info = check_camera(url)
            results[camera_num] = {
                'accessible': is_accessible,
                'quality_info': quality_info
            }

            if is_podolsk:
                status = "работает" if is_accessible else "не работает"
                logging.info(f"Подольск: камера {camera_num} - {status}")
            elif is_etalon:
                status = "работает" if is_accessible else "не работает"
                logging.info(f"Эталон: камера {camera_num} - {status}")
            elif is_skandinavia:
                status = "работает" if is_accessible else "не работает"
                logging.info(f"Скандинавия: камера {camera_num} - {status}")

        for confirmation_round in range(OFFLINE_CONFIRMATION_ROUNDS):
            offline_cameras = [
                camera_num
                for camera_num, info in results.items()
                if not info['accessible']
            ]
            if not offline_cameras:
                break

            logging.info(
                "%s: контрольная перепроверка Offline камер через %s сек: %s",
                branch_name,
                OFFLINE_CONFIRMATION_DELAY_S,
                ", ".join(map(str, offline_cameras)),
            )
            time.sleep(OFFLINE_CONFIRMATION_DELAY_S)

            for camera_num in offline_cameras:
                url = base_url.format(camera_num)
                logging.info(
                    "%s: контрольная перепроверка камеры %s, круг %s/%s",
                    branch_name,
                    camera_num,
                    confirmation_round + 1,
                    OFFLINE_CONFIRMATION_ROUNDS,
                )
                is_accessible, quality_info = check_camera(url)
                if is_accessible:
                    logging.info(
                        "%s: камера %s восстановилась на контрольной перепроверке",
                        branch_name,
                        camera_num,
                    )
                results[camera_num] = {
                    'accessible': is_accessible,
                    'quality_info': quality_info
                }

        if is_podolsk:
            working_cameras = sum(1 for info in results.values() if info['accessible'])
            logging.info(f"Подольск: проверка завершена. Работает {working_cameras}/{len(cameras_to_check)} камер")
            if working_cameras == 0:
                logging.error(f"Подольск: ВНИМАНИЕ! Все камеры не работают!")
        elif is_etalon:
            working_cameras = sum(1 for info in results.values() if info['accessible'])
            logging.info(f"Эталон: проверка завершена. Работает {working_cameras}/{len(cameras_to_check)} камер")
        elif is_skandinavia:
            working_cameras = sum(1 for info in results.values() if info['accessible'])
            logging.info(f"Скандинавия: проверка завершена. Работает {working_cameras}/{len(cameras_to_check)} камер")

    return results

def send_report_to_telegram(token: str, chat_id: str, summary_text: str, disable_notification: bool = False):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        'chat_id': chat_id, 
        'text': summary_text,
        'disable_notification': disable_notification
    }
    response = requests.post(url, data=data)
    return response.ok


def main():
    # Глобальная взаимоблокировка, чтобы скрипт не запускался в двух экземплярах
    # Одновременный запуск приводил к отправке двух отчетов
    lockfile_path = '/tmp/remote_cam_bot.lock'
    try:
        lockfile = open(lockfile_path, 'w')
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lockfile.write(str(os.getpid()))
        lockfile.flush()
    except BlockingIOError:
        logging.warning("Another instance is already running. Exiting to prevent duplicate reports.")
        return
    except Exception as e:
        logging.error(f"Failed to acquire lock file {lockfile_path}: {e}")
        return
    logging.info("Script started, waiting for the next full hour...")
    TELEGRAM_TOKEN = "7993101154:AAGR2zZ_HAztxcL_LnYu-t441cfruo-tKAQ"
    TELEGRAM_CHAT_ID = "-1002760249281"
    MSK = pytz.timezone('Europe/Moscow')
    last_check_hour = -1  # Переменная для отслеживания последней проверки
    
    while True:
        now = datetime.now(MSK)
        logging.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} MSK")
        
        # Проверяем, что сейчас в нужном диапазоне (11-22 по МСК) и еще не проверяли в этот час
        if 11 <= now.hour <= 22 and now.hour != last_check_hour:
            
            logging.info(f"Starting camera check at {now.strftime('%H:%M')}")
            try:
                # Create a list to store results
                results = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_branch = {
                        executor.submit(
                            check_branch_cameras,
                            branch_name,
                            base_url,
                            CAMERA_MATRIX[branch_name]
                        ): branch_name
                        for branch_name, base_url in BRANCHES.items()
                    }
                    for future in future_to_branch:
                        branch_name = future_to_branch[future]
                        try:
                            camera_results = future.result()
                            for camera_num, camera_info in camera_results.items():
                                is_accessible = camera_info['accessible']
                                quality_info = camera_info['quality_info']
                                
                                # Определяем статус камеры
                                if not is_accessible:
                                    status = 'Offline'
                                else:
                                    status = 'Online'
                                
                                results.append({
                                    'Branch': branch_name,
                                    'Camera': camera_num,
                                    'Status': status,
                                    'Timestamp': datetime.now(MSK).strftime('%Y-%m-%d %H:%M:%S')
                                })
                        except Exception as e:
                            logging.error(f"Error checking branch {branch_name}: {str(e)}")
                
                df = pd.DataFrame(results)
                df = df.sort_values(['Branch', 'Camera'])
                # Убираем создание Excel файла
                # timestamp = datetime.now(MSK).strftime('%Y%m%d_%H%M%S')
                # excel_filename = f'camera_status_{timestamp}.xlsx'
                # df.to_excel(excel_filename, index=False)
            except Exception as e:
                logging.error(f"Error during camera check: {str(e)}")
                # Не обновляем last_check_hour, чтобы можно было повторить попытку
                sleep(60)  # Ждем минуту перед повторной попыткой
                continue
            # Print summary
            summary_lines = []
            # Создаем список филиалов с информацией о статусе для сортировки
            branch_status_list = []
            for branch in BRANCHES.keys():
                branch_results = df[df['Branch'] == branch]
                total = len(branch_results)
                online = len(branch_results[branch_results['Status'] == 'Online'])
                offline_cameras = branch_results[branch_results['Status'] == 'Offline']['Camera'].tolist()
                has_offline = len(offline_cameras) > 0
                branch_status_list.append((branch, online, total, has_offline, offline_cameras))
            
            # Сортируем: сначала филиалы с проблемами (❌), затем остальные (✅)
            branch_status_list.sort(key=lambda x: (not x[3], x[0]))  # not x[3] делает True (есть проблемы) первым, затем по алфавиту
            
            for branch, online, total, has_offline, offline_cameras in branch_status_list:
                if online == total:
                    summary_lines.append(f"✅ {branch}: {online}/{total} камер работает")
                else:
                    status_text = f" (камеры {', '.join(map(str, offline_cameras))})" if has_offline else ""
                    summary_lines.append(f"❌ {branch}: {online}/{total} камер работает{status_text}")
            summary_text = '\n'.join(summary_lines)
            logging.info("\nCamera Status Summary:")
            logging.info("=" * 50)
            logging.info(summary_text)
            # Убираем упоминание Excel файла
            # logging.info(f"Detailed results saved to: {excel_filename}")
            
            # Проверяем, есть ли проблемы с камерами
            has_problems = any(has_offline for _, _, _, has_offline, _ in branch_status_list)
            
            # Отправка в Telegram (без звука, если все камеры работают)
            logging.info(f"Sending report to Telegram, has_problems: {has_problems}")
            success = send_report_to_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, summary_text, disable_notification=not has_problems)
            if success:
                logging.info("Report sent successfully to Telegram")
            else:
                logging.error("Failed to send report to Telegram")
            
            # Обновляем время последней проверки
            last_check_hour = now.hour
            
            # Ждём до следующего часа
            now = datetime.now(MSK)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            sleep_seconds = (next_hour - now).total_seconds()
            logging.info(f"Waiting for the next full hour, sleeping for {int(sleep_seconds)} seconds...")
            sleep(max(10, sleep_seconds))
        else:
            # Ждём 60 секунд и проверяем снова
            logging.info("Not in working hours or already checked this hour, waiting 60 seconds...")
            sleep(60)

if __name__ == "__main__":
    main()
