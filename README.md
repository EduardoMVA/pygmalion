# Pygmalion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EduardoMVA/pygmalion/blob/main/notebooks/tutorial.ipynb)

Librería local de Python para generación de datos sintéticos tabulares single-table, guiada por JSON explícito.

## Principios

- El usuario controla la generación mediante un JSON spec explícito.
- La librería no infiere intenciones ambiguas.
- Claridad, control y extensibilidad sobre magia.

## Instalación

```bash
# Clonar el proyecto
git clone <tu-repositorio>
cd pygmalion-project

# Crear entorno virtual
python -m venv .venv

# Activar (Windows PowerShell)
.venv\Scripts\Activate

# Instalar en modo editable
pip install -e ".[dev]"
```

## Uso rápido

### Generar datos desde un spec manual

```python
from pygmalion import synthesize

spec = {
    "num_rows": 1000,
    "columns": [
        {"name": "edad", "type": "normal", "mean": 35, "std": 8, "min": 18, "max": 65},
        {"name": "salario", "type": "uniform", "low": 20000, "high": 80000},
        {"name": "departamento", "type": "categorical", "values": ["ventas", "tech", "rh"]},
    ]
}

df = synthesize(spec)
```

### Elegir formato de salida

```python
# Pandas (default)
df_pandas = synthesize(spec)

# Polars
df_polars = synthesize(spec, output_format="polars")
```

### Aprender desde un CSV

```python
from pygmalion import learn_from_csv, synthesize

spec = learn_from_csv("datos_reales.csv", num_rows=5000)
df = synthesize(spec)
```

### Template editable

```python
from pygmalion import template_from_data

template = template_from_data("datos_reales.csv")
# Editar el template manualmente, luego:
df = synthesize(template)
```

### Inspeccionar sin generar

```python
from pygmalion import stats_only

stats = stats_only(spec)
print(stats)
```

### Evaluar calidad

```python
from pygmalion import quality_report

report = quality_report(df_real, df_sintetico)
print(report["overall_score"])
```

### Exportar

```python
from pygmalion import to_csv, to_json

to_csv(df, "salida.csv")
to_json(df, "salida.json")
```

## Tipos de columna soportados

### normal

Distribución normal con truncamiento opcional.

```json
{"name": "edad", "type": "normal", "mean": 35, "std": 8, "min": 18, "max": 65}
```

### uniform

Distribución uniforme.

```json
{"name": "precio", "type": "uniform", "low": 10, "high": 100}
```

### categorical

Valores categóricos con pesos opcionales.

```json
{"name": "color", "type": "categorical", "values": ["rojo", "azul"], "weights": [0.7, 0.3]}
```

### mixture

Mezcla de distribuciones.

```json
{
    "name": "salario",
    "type": "mixture",
    "components": [
        {"type": "normal", "mean": 30000, "std": 5000, "weight": 0.7},
        {"type": "normal", "mean": 80000, "std": 15000, "weight": 0.3}
    ]
}
```

### derived

Columna calculada a partir de otras.

```json
{"name": "total", "type": "derived", "expr": "precio * cantidad", "dependencies": ["precio", "cantidad"]}
```

### conditional

Distribución que depende del valor de otra columna.

```json
{
    "name": "salario",
    "type": "conditional",
    "condition_column": "nivel",
    "cases": {
        "junior": {"type": "normal", "mean": 25000, "std": 3000},
        "senior": {"type": "normal", "mean": 60000, "std": 10000}
    }
}
```

## Constraints

Reglas que el DataFrame generado debe cumplir.

```json
{
    "num_rows": 1000,
    "columns": [...],
    "constraints": ["experiencia <= edad - 18", "precio_venta > precio_costo"]
}
```

## Arquitectura

pygmalion/
├── schema/       ← Validación del JSON spec (Pydantic)
├── generators/   ← Generadores por tipo de distribución
├── engine/       ← Orquestador que conecta schema + generators
├── io/           ← Lectura, escritura, stats, quality report
└── constraints/  ← Validación post-generación

Las dependencias van en una sola dirección: schema ← generators ← engine → io → constraints.

## Extender Pygmalion

Para agregar una nueva distribución:

1. Crear la clase spec en `pygmalion/schema/spec.py` con su `Literal` type.
2. Agregarla al `ColumnSpec` union.
3. Crear el generador en `pygmalion/generators/` heredando de `BaseGenerator`.
4. Registrarlo con `register("nombre", MiClase)`.
5. Importar el módulo en `pygmalion/generators/__init__.py`.
6. Agregar tests.

## Tests

```bash
pytest tests/ -v
```

## Ejemplo completo

Ver `examples/basic_usage.py` para un ejemplo end-to-end con todas las funcionalidades.