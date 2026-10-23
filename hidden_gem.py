#!/usr/bin/env python3
"""
Универсальный сборщик исходного кода из файлов .py, .yaml, .ipynb
с возможностью игнорирования файлов/папок по шаблонам.

Использование:
    python collect_code.py <папка_проекта> [выходной_файл.txt] [--ignore PATTERN ...]
Примеры:
    python collect_code.py ./project result.txt --ignore __pycache__ "*.egg-info"
"""

import os
import sys
import json
import argparse
import fnmatch
from abc import ABC, abstractmethod
from typing import Dict, List


# ----------------------------------------------------------------------
# Абстрактный интерфейс "читателя" файлов
# ----------------------------------------------------------------------
class BaseReader(ABC):
    """Базовый класс для чтения файлов определённого типа."""
    @abstractmethod
    def read(self, file_path: str) -> str:
        """
        Читает содержимое файла и возвращает его в виде строки.
        Должен быть переопределён для каждого расширения.
        """
        ...


# ----------------------------------------------------------------------
# Конкретные реализации "движков"
# ----------------------------------------------------------------------
class PlainTextReader(BaseReader):
    """Читатель для обычных текстовых файлов (.py, .yaml)."""
    def read(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


class IpynbReader(BaseReader):
    """
    Читатель для Jupyter-ноутбуков (.ipynb).
    Извлекает и объединяет исходный код всех ячеек типа 'code'.
    """
    def read(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Ошибка чтения ноутбука {file_path}: {e}", file=sys.stderr)
            return ""

        code_lines = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                # source – это список строк, сохраняем «как есть»
                code_lines.extend(cell.get('source', []))
        # Объединяем без дополнительных изменений (переносы строк уже внутри)
        return ''.join(code_lines)


# ----------------------------------------------------------------------
# Сопоставление расширений с читателями
# ----------------------------------------------------------------------
READERS: Dict[str, BaseReader] = {
    '.py': PlainTextReader(),
    '.yaml': PlainTextReader(),
    '.ipynb': IpynbReader(),
}


def should_ignore(name: str, patterns: List[str]) -> bool:
    """
    Проверяет, подпадает ли имя файла/папки под один из игнорируемых шаблонов.
    Шаблоны могут содержать подстановочные знаки *, ?, [...].
    """
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def collect_code(source_dir: str, output_file: str, ignore_patterns: List[str] = None) -> None:
    """
    Рекурсивно обходит source_dir, находит файлы с нужными расширениями,
    извлекает код и записывает всё в output_file.

    :param source_dir: корень проекта
    :param output_file: путь к результирующему txt-файлу
    :param ignore_patterns: список шаблонов имён файлов/папок, которые нужно пропускать
    """
    if ignore_patterns is None:
        ignore_patterns = []

    source_dir = os.path.abspath(source_dir)

    if not os.path.isdir(source_dir):
        print(f"Ошибка: директория '{source_dir}' не существует.", file=sys.stderr)
        sys.exit(1)

    files_to_read = []
    for root, dirs, files in os.walk(source_dir, topdown=True):
        # Исключаем игнорируемые папки
        dirs[:] = [d for d in dirs if not should_ignore(d, ignore_patterns)]

        for filename in files:
            # Пропускаем игнорируемые файлы (по базовому имени)
            if should_ignore(filename, ignore_patterns):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in READERS:
                full_path = os.path.join(root, filename)
                files_to_read.append(full_path)

    files_to_read.sort()

    with open(output_file, 'w', encoding='utf-8') as out:
        for file_path in files_to_read:
            ext = os.path.splitext(file_path)[1].lower()
            reader = READERS[ext]

            rel_path = os.path.relpath(file_path, source_dir)
            try:
                code = reader.read(file_path)
            except Exception as e:
                print(f"Пропущен файл {file_path}: {e}", file=sys.stderr)
                continue

            out.write(f"'{rel_path}':\n{code}\n\n")

    print(f"Готово! Код записан в {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Сборщик кода из .py, .yaml, .ipynb с возможностью игнорирования файлов/папок"
    )
    parser.add_argument(
        "source_dir",
        help="Папка проекта для сканирования"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="collected_code.txt",
        help="Имя выходного .txt файла (по умолчанию collected_code.txt)"
    )
    parser.add_argument(
        "-i", "--ignore",
        action="append",
        default=[],
        help="Шаблон имени файла или папки, который нужно игнорировать. "
             "Можно указывать несколько раз. Поддерживает *, ?, [...]"
    )

    args = parser.parse_args()
    collect_code(args.source_dir, args.output_file, args.ignore)


if __name__ == "__main__":
    main()
