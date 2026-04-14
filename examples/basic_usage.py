"""
Pygmalion - Ejemplo de uso completo
====================================

Este script demuestra el flujo completo de Pygmalion:
1. Definir un spec JSON manualmente
2. Generar datos sintéticos
3. Exportar a CSV y JSON
4. Aprender un spec desde un CSV
5. Generar datos desde el spec aprendido
6. Comparar calidad entre datos originales y sintéticos
7. Usar stats_only para inspeccionar un spec sin generar
8. Usar template_from_data para obtener un esqueleto editable
"""

import json
from pathlib import Path

from pygmalion import (
    synthesize,
    learn_from_csv,
    template_from_data,
    stats_only,
    quality_report,
    to_csv,
    to_json,
)


def main():
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Definir un spec manualmente
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. Generación desde spec manual")
    print("=" * 60)

    spec = {
        "num_rows": 1000,
        "columns": [
            {
                "name": "nivel",
                "type": "categorical",
                "values": ["junior", "mid", "senior"],
                "weights": [0.5, 0.3, 0.2],
            },
            {
                "name": "edad",
                "type": "normal",
                "mean": 35,
                "std": 8,
                "min": 22,
                "max": 65,
            },
            {
                "name": "salario_base",
                "type": "mixture",
                "components": [
                    {"type": "normal", "mean": 30000, "std": 5000, "weight": 0.5},
                    {"type": "normal", "mean": 55000, "std": 8000, "weight": 0.3},
                    {"type": "normal", "mean": 90000, "std": 12000, "weight": 0.2},
                ],
            },
            {
                "name": "horas_semanales",
                "type": "uniform",
                "low": 20,
                "high": 50,
            },
            {
                "name": "ingreso_mensual",
                "type": "derived",
                "expr": "salario_base / 12",
                "dependencies": ["salario_base"],
            },
            {
                "name": "bono",
                "type": "conditional",
                "condition_column": "nivel",
                "cases": {
                    "junior": {"type": "normal", "mean": 1000, "std": 200},
                    "mid": {"type": "normal", "mean": 3000, "std": 500},
                    "senior": {"type": "normal", "mean": 8000, "std": 1500},
                },
            },
        ],
        "constraints": [
            "salario_base > 15000",
        ],
    }

    df = synthesize(spec)
    print(f"DataFrame generado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    print()
    print(df.head(10).to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 2. Exportar a CSV y JSON
    # ------------------------------------------------------------------
    print("=" * 60)
    print("2. Exportar datos")
    print("=" * 60)

    csv_path = output_dir / "empleados_sinteticos.csv"
    json_path = output_dir / "empleados_sinteticos.json"

    to_csv(df, csv_path)
    to_json(df, json_path)

    print(f"CSV guardado en: {csv_path}")
    print(f"JSON guardado en: {json_path}")
    print()

    # ------------------------------------------------------------------
    # 3. Stats only - inspeccionar sin generar
    # ------------------------------------------------------------------
    print("=" * 60)
    print("3. Stats only")
    print("=" * 60)

    stats = stats_only(spec)
    print(f"Filas configuradas: {stats['num_rows']}")
    print(f"Columnas: {stats['num_columns']}")
    for col_name, col_stats in stats["columns"].items():
        print(f"  {col_name}: {col_stats['type']}", end="")
        if "mean" in col_stats:
            print(f" (mean={col_stats['mean']})", end="")
        if "expected_mean" in col_stats:
            print(f" (expected_mean={col_stats['expected_mean']})", end="")
        print()
    print()

    # ------------------------------------------------------------------
    # 4. Learn from CSV
    # ------------------------------------------------------------------
    print("=" * 60)
    print("4. Learn from CSV")
    print("=" * 60)

    learned_spec = learn_from_csv(csv_path, num_rows=500)
    print("Spec aprendido:")
    print(json.dumps(learned_spec, indent=2))
    print()

    # ------------------------------------------------------------------
    # 5. Generar desde spec aprendido
    # ------------------------------------------------------------------
    print("=" * 60)
    print("5. Generar desde spec aprendido")
    print("=" * 60)

    df_learned = synthesize(learned_spec)
    print(f"DataFrame generado: {df_learned.shape[0]} filas")
    print()

    # ------------------------------------------------------------------
    # 6. Quality report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("6. Quality report")
    print("=" * 60)

    report = quality_report(df, df_learned)
    print(f"Score global: {report['overall_score']}")
    for col_name, col_report in report["columns"].items():
        print(f"  {col_name}: score={col_report['score']} ({col_report['type']})")
    print()

    # ------------------------------------------------------------------
    # 7. Template from data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("7. Template from data")
    print("=" * 60)

    template = template_from_data(csv_path)
    print("Template generado:")
    print(json.dumps(template, indent=2))
    print()

    # ------------------------------------------------------------------
    # 8. Polars output
    # ------------------------------------------------------------------
    print("=" * 60)
    print("8. Output en Polars")
    print("=" * 60)

    simple_spec = {
        "num_rows": 5,
        "columns": [
            {"name": "x", "type": "normal", "mean": 0, "std": 1},
            {"name": "y", "type": "uniform", "low": 0, "high": 100},
        ],
    }

    df_polars = synthesize(simple_spec, output_format="polars")
    print(f"Tipo: {type(df_polars)}")
    print(df_polars)
    print()

    print("=" * 60)
    print("Ejemplo completo terminado.")
    print("=" * 60)


if __name__ == "__main__":
    main()