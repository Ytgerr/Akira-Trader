#!/usr/bin/env python3
"""
Скрипт для валидации submission файла.

Использование:
    python3 validate_submission.py [OPTIONS]

Проверяет:
- Наличие файла data/processed/submission.csv (или указанного файла)
- Правильную структуру (uid;type;request)
- Соответствие количества строк с test.csv
- Наличие всех uid из test.csv
- Валидность HTTP методов в type
- Валидность API путей в request
- Отсутствие пустых значений
- Уникальность uid

Примеры:
    # Проверить data/processed/submission.csv (по умолчанию)
    python3 validate_submission.py

    # Проверить конкретный файл
    python3 validate_submission.py --file data/processed/sample_submission.csv

    # Проверить файл по абсолютному пути
    python3 validate_submission.py --file /path/to/submission.csv
"""

from typing import Optional

import click
from tests.test_submission_validator import SubmissionValidator


@click.command()
@click.option(
    "--file",
    "-f",
    "submission_file",
    type=click.Path(exists=True),
    help="Путь к файлу submission для проверки. По умолчанию: data/processed/submission.csv",
)
def main(submission_file: Optional[str]) -> int:
    """Валидировать submission файл для хакатона"""

    print("🚀 Запуск валидации submission файла...")

    if submission_file:
        print(f"📁 Проверяемый файл: {submission_file}")
    else:
        print("📁 Проверяемый файл: data/processed/submission.csv (по умолчанию)")

    print("=" * 50)

    try:
        validator = SubmissionValidator(submission_file)
        results = validator.run_all_validations()
    except FileNotFoundError as e:
        click.echo(f"❌ Ошибка: Файл test.csv не найден: {e}", err=True)
        return 1
    except Exception as e:
        click.echo(f"❌ Ошибка инициализации: {e}", err=True)
        return 1

    passed = 0
    failed = 0

    for name, success, error in results:
        status = "✅" if success else "❌"
        click.echo(f"{status} {name}")
        if not success:
            click.echo(f"   Ошибка: {error}")
            failed += 1
        else:
            passed += 1

    click.echo("=" * 50)
    click.echo(f"📊 Результаты: {passed} пройдено, {failed} провалено")

    if failed == 0:
        click.echo("🎉 Поздравляем! Submission файл полностью валиден.")
        return 0
    click.echo("⚠️  Найдены ошибки валидации. Исправьте их перед отправкой.")
    return 1


if __name__ == "__main__":
    exit(main())
