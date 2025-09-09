#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import pytz

from main import BRANCHES, CAMERA_MATRIX, check_branch_cameras


def run_check(selected_branches: list[str], max_workers: int = 5) -> str:
    """
    Выполняет проверку указанных филиалов и возвращает текстовую сводку.
    """
    MSK = pytz.timezone('Europe/Moscow')

    results_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_branch = {
            executor.submit(
                check_branch_cameras,
                branch_name,
                BRANCHES[branch_name],
                CAMERA_MATRIX[branch_name],
            ): branch_name
            for branch_name in selected_branches
        }

        for future, branch_name in list(future_to_branch.items()):
            camera_results = future.result()
            for camera_num, camera_info in camera_results.items():
                is_accessible = camera_info['accessible']
                status = 'Online' if is_accessible else 'Offline'
                results_rows.append({
                    'Branch': branch_name,
                    'Camera': camera_num,
                    'Status': status,
                    'Timestamp': datetime.now(MSK).strftime('%Y-%m-%d %H:%M:%S'),
                })

    df = pd.DataFrame(results_rows).sort_values(['Branch', 'Camera'])

    # Формируем сводку в том же формате, что и в основном скрипте
    summary_lines: list[str] = []
    branch_status_list: list[tuple] = []

    for branch in selected_branches:
        branch_results = df[df['Branch'] == branch]
        total = len(branch_results)
        online = len(branch_results[branch_results['Status'] == 'Online'])
        offline_cameras = branch_results[branch_results['Status'] == 'Offline']['Camera'].tolist()
        has_offline = len(offline_cameras) > 0
        branch_status_list.append((branch, online, total, has_offline, offline_cameras))

    # Сначала филиалы с проблемами (❌), затем остальные (✅)
    branch_status_list.sort(key=lambda x: (not x[3], x[0]))

    for branch, online, total, has_offline, offline_cameras in branch_status_list:
        if online == total:
            summary_lines.append(f"✅ {branch}: {online}/{total} камер работает")
        else:
            status_text = f" (камеры {', '.join(map(str, offline_cameras))})" if has_offline else ""
            summary_lines.append(f"❌ {branch}: {online}/{total} камер работает{status_text}")

    return '\n'.join(summary_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Тестовая проверка камер. Печатает результат сразу в консоль.'
    )
    parser.add_argument(
        '--branches',
        nargs='*',
        help='Список филиалов для проверки (по умолчанию: Подольск, Эталон).',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Проверить все филиалы из конфигурации.'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Количество потоков проверки (по умолчанию: 5).',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.all:
        selected = list(BRANCHES.keys())
    elif args.branches and len(args.branches) > 0:
        # Валидация имён филиалов
        unknown = [b for b in args.branches if b not in BRANCHES]
        if unknown:
            print('Неизвестные филиалы:', ', '.join(unknown))
            print('Доступные:', ', '.join(BRANCHES.keys()))
            raise SystemExit(2)
        selected = args.branches
    else:
        # Значения по умолчанию для быстрых тестов
        selected = ['Подольск', 'Эталон']

    summary = run_check(selected, max_workers=args.max_workers)
    print('\n' + '=' * 50)
    print(summary)
    print('=' * 50 + '\n')


