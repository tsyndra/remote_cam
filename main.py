import av
import time
import os
import numpy as np
from typing import Dict, List, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime, timedelta
import requests
import pytz
from time import sleep

@dataclass
class Branch:
    name: str
    base_url: str
    cameras_to_check: Set[int]

# Dictionary of branches with their base RTSP URLs
BRANCHES = {
    "40 лет": "rtsp://admin:4feeDTeH@87.239.29.247:554/cam/realmonitor?channel={}&subtype=00",
    "Алфавит": "rtsp://admin:4feeDTeH@82.149.223.218:554/cam/realmonitor?channel={}&subtype=00",
    "Ватутинки": "rtsp://admin:Qwerty12345@88.210.49.169:554/cam/realmonitor?channel={}&subtype=00",
    "Восток": "rtsp://admin:4feeDTeH@62.217.190.201:554/cam/realmonitor?channel={}&subtype=00",
    "Испания": "rtsp://admin:4feeDTeH@46.242.38.131:554/cam/realmonitor?channel={}&subtype=00",
    "Коммунарка": "rtsp://admin:4feeDTeH@94.102.123.173:554/cam/realmonitor?channel={}&subtype=00",
    "Луга": "rtsp://admin:4feeDTeH@185.136.76.134:554/cam/realmonitor?channel={}&subtype=00",
    "Лучи(в работе)": "rtsp://admin:4feeDTeH@83.237.37.58:554/cam/realmonitor?channel={}&subtype=00",
    "Московский": "rtsp://admin:4feeDTeH@5.228.14.150:554/cam/realmonitor?channel={}&subtype=00",
    "Подольск": "rtsp://admin:4feeDTeH@77.51.218.182:554/cam/realmonitor?channel={}&subtype=00",
    "Скандинавия": "rtsp://admin:4feeDTeH@178.140.203.115:554/cam/realmonitor?channel={}&subtype=00",
    "Щербинка": "rtsp://admin:4feeDTeH@87.239.29.249:554/cam/realmonitor?channel={}&subtype=00",
    "Эталон(в работе)": "rtsp://admin:4feeDTeH@87.239.29.42:554/cam/realmonitor?channel={}&subtype=00",
    "Ясенево": "rtsp://admin:$Er13%kod1@95.143.218.198:554/cam/realmonitor?channel={}&subtype=00"
}

# Camera availability matrix (✅ = check, - = don't check)
CAMERA_MATRIX = {
    "40 лет": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16},
    "Алфавит": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    "Ватутинки": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    "Восток": {1, 2, 3, 4, 5, 6, 7, 8},
    "Испания": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    "Коммунарка": {1, 2, 3, 4, 5, 6, 7, 8, 9},
    "Луга": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    "Лучи(в работе)": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    "Московский": {1, 3, 4, 5, 6, 7, 8, 9, 10},
    "Подольск": {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12},
    "Скандинавия": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
    "Щербинка": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    "Эталон(в работе)": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    "Ясенево": {1, 2, 3, 4, 5, 6, 7, 8}
}

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

def check_camera(url: str, timeout: int = 5, max_attempts: int = 3, num_frames_to_check: int = 5) -> bool:
    """
    Check if a camera is accessible and showing non-black images using PyAV.
    Returns True if camera is accessible and showing non-black images, False otherwise.
    
    Args:
        url: RTSP URL камеры
        timeout: таймаут в секундах для каждой попытки
        max_attempts: максимальное количество попыток подключения
        num_frames_to_check: количество кадров для проверки на черное изображение
    """
    for attempt in range(max_attempts):
        try:
            # Устанавливаем параметры для RTSP
            options = {
                'rtsp_transport': 'tcp',
                'stimeout': str(timeout * 1000000),  # таймаут в микросекундах
                'buffer_size': '1024000',  # размер буфера
                'reorder_queue_size': '0',  # отключаем переупорядочивание пакетов
                'max_delay': '500000'  # максимальная задержка в микросекундах
            }
            
            # Пробуем открыть поток
            with av.open(url, options=options, timeout=timeout) as container:
                # Проверяем несколько кадров
                black_frames_count = 0
                frames_checked = 0
                
                for frame in container.decode(video=0):
                    if frames_checked >= num_frames_to_check:
                        break
                        
                    if is_black_frame(frame):
                        black_frames_count += 1
                    frames_checked += 1
                
                # Если все проверенные кадры черные, считаем камеру неработающей
                if frames_checked > 0 and black_frames_count == frames_checked:
                    print(f"Камера {url} показывает черное изображение")
                    return False
                    
                if attempt > 0:
                    print(f"Камера {url} заработала с {attempt + 1} попытки")
                return True  # Если удалось прочитать кадры и не все они черные
                
        except av.error.FFmpegError as e:
            if 'timeout' in str(e).lower():
                if attempt < max_attempts - 1:
                    print(f"Таймаут подключения к {url}, попытка {attempt + 1} из {max_attempts}")
                    time.sleep(1)  # Ждем секунду перед следующей попыткой
                    continue
                print(f"Таймаут подключения после {max_attempts} попыток: {url}")
            else:
                print(f"Ошибка FFmpeg: {url} - {str(e)}")
            return False
        except Exception as e:
            print(f"Ошибка при проверке камеры {url}: {str(e)}")
            return False
    
    return False

def check_branch_cameras(branch_name: str, base_url: str, cameras_to_check: Set[int]) -> Dict[int, bool]:
    """
    Check all cameras for a specific branch.
    Returns a dictionary with camera numbers as keys and their status as values.
    """
    results = {}
    for camera_num in cameras_to_check:
        url = base_url.format(camera_num)
        is_accessible = check_camera(url)
        results[camera_num] = is_accessible
    return results

def send_report_to_telegram(token: str, chat_id: str, summary_text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {'chat_id': chat_id, 'text': summary_text}
    response = requests.post(url, data=data)
    return response.ok

def main():
    print("Script started, waiting for the next full hour...")
    TELEGRAM_TOKEN = "7993101154:AAGR2zZ_HAztxcL_LnYu-t441cfruo-tKAQ"
    TELEGRAM_CHAT_ID = "-1002760249281"
    MSK = pytz.timezone('Europe/Moscow')
    while True:
        now = datetime.now(MSK)
        # Проверяем, что сейчас ровно начало часа и в нужном диапазоне
        if 11 <= now.hour <= 22 and now.minute == 0:
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
                        for camera_num, is_accessible in camera_results.items():
                            results.append({
                                'Branch': branch_name,
                                'Camera': camera_num,
                                'Status': 'Online' if is_accessible else 'Offline',
                                'Timestamp': datetime.now(MSK).strftime('%Y-%m-%d %H:%M:%S')
                            })
                    except Exception as e:
                        print(f"Error checking branch {branch_name}: {str(e)}")
            df = pd.DataFrame(results)
            df = df.sort_values(['Branch', 'Camera'])
            timestamp = datetime.now(MSK).strftime('%Y%m%d_%H%M%S')
            excel_filename = f'camera_status_{timestamp}.xlsx'
            df.to_excel(excel_filename, index=False)
            # Print summary
            summary_lines = []
            for branch in sorted(BRANCHES.keys()):
                branch_results = df[df['Branch'] == branch]
                total = len(branch_results)
                online = len(branch_results[branch_results['Status'] == 'Online'])
                if online == total:
                    summary_lines.append(f"✅ {branch}: {online}/{total} камер работает")
                else:
                    summary_lines.append(f"❌ {branch}: {online}/{total} камер работает")
            summary_text = '\n'.join(summary_lines)
            print("\nCamera Status Summary:")
            print("=" * 50)
            print(summary_text)
            print(f"\nDetailed results saved to: {excel_filename}")
            # Отправка в Telegram
            send_report_to_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, summary_text)
            # Ждём до следующего часа
            now = datetime.now(MSK)
            seconds_to_next_hour = 3600 - (now.minute * 60 + now.second)
            sleep(seconds_to_next_hour)
        else:
            # Ждём до следующего целого часа
            now = datetime.now(MSK)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            sleep_seconds = (next_hour - now).total_seconds()
            print(f"Waiting for the next full hour, sleeping for {int(sleep_seconds)} seconds...")
            sleep(max(10, sleep_seconds))

if __name__ == "__main__":
    main()
