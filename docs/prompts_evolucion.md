# Evolución del prompt del agente PaperStats

Este documento registra, con trazabilidad de commits, la iteración del prompt
del subagente `.opencode/agents/paperstats.md`. Existe para cumplir el criterio
de **iteración documentada de prompts** sin alterar la arquitectura
determinista del motor (`src/` no consume LLM; ver
`docs/arquitectura_agente.md`).

## v1 — creación (commit `140fd8b`, 2026-08-22)

- Prompt inicial de 34 líneas: rol, reglas permanentes (guardarraíles),
  herramientas/skills y prohibiciones.
- Frontmatter con `model: Muse Spark 1.2 Free`.
- **Problemas detectados en auditoría (criterio 3.2):**
  1. Sin contrato de salida: no definía la estructura que debe tener una
     respuesta analítica (método → justificación → resultado → interpretación
     → limitaciones).
  2. Sin ejemplos entrada→salida (few-shot) que anclen el comportamiento.
  3. No distinguía explícitamente información de autores vs. interpretación
     vs. hipótesis del agente (regla anti-alucinaciones solo implícita).

## v1.1 — cambio de modelo (commit `3592aa9`, 2026-08-23)

- Se cambia `model` a `google/gemini-1.5-pro`. El commit estaba centrado en la
  gestión de credenciales (Criterio 3.5) y **no justificaba el cambio de
  modelo** en su mensaje — deficiencia de trazabilidad reconocida.
- Se introdujo accidentalmente un carácter TAB antes del `---` de apertura del
  frontmatter, que podía romper el parseo YAML en algunos parsers estrictos.

## v2 — actual (2026-08)

Cambios aplicados sobre v1.1, motivados por la auditoría del criterio 3.2:

1. **Corregido** el TAB espurio de la línea 1; frontmatter válido.
2. **Añadido CONTRATO DE SALIDA**: estructura obligatoria de las respuestas
   analíticas (Método / Justificación / Resultado / Interpretación /
   Limitaciones), reglas de incertidumbre (estadístico + valor p siempre) y
   marcado explícito `[AUTORES]` / `[INTERPRETACIÓN]` / `[HIPÓTESIS]`.
3. **Añadidos 2 ejemplos entrada→salida** (pregunta analítica completa y caso
   de dato ausente con respuesta "No se especifica en el artículo.").
4. **Marcador de versión** del prompt en comentario HTML (`PROMPT_VERSION:
   v2`) apuntando a este documento.
5. Se conserva íntegro el contenido semántico de v1 (rol, guardarraíles,
   skills, prohibiciones).

### Justificación

El contrato de salida codifica en el prompt las mismas reglas que ya gobiernan
al sistema determinista (AGENTS.md §4 y §5), de modo que el comportamiento del
subagente y el del motor sean consistentes. Los ejemplos few-shot reducen la
varianza de formato en respuestas posteriores.

## Estado de los prompts en la arquitectura

- El motor de análisis (`src/`) es determinista y **no consume prompts ni LLM**
  en tiempo de ejecución.
- Los únicos prompts activos del repositorio son este archivo de subagente y
  `AGENTS.md` (prompt de nivel sesión).
- La capa `src/llm/` (`BaseLLMClient` + `RuleBasedLLMClient`) es una
  abstracción preparatoria para Ollama; hasta su integración, cualquier
  mecanismo adicional de prompting sería código muerto y no se implementa.
