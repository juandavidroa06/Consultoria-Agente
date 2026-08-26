# Instrucciones de uso — Dashboard interactivo

Este panel permite **explorar los datos de forma visual y sencilla**, sin necesidad
de conocimientos técnicos ni de programación. Es solo una herramienta de consulta:
**los archivos originales nunca se modifican.**

---

## 1. Cómo abrir el dashboard

1. Abre una ventana de terminal en la carpeta del proyecto.
2. Escribe el siguiente comando y pulsa Enter:

   ```
   .venv\Scripts\python.exe run_dashboard.py
   ```

3. En pocos segundos se abrirá automáticamente una página en tu navegador
   (normalmente en la dirección `http://localhost:8501`).
4. Cuando termines, vuelve a la terminal y pulsa `Ctrl + C` para cerrarlo.

> Si el navegador no se abre solo, copia la dirección que aparece en la terminal
> (algo como `Local URL: http://localhost:8501`) y pégala en tu navegador.

---

## 2. Qué encontrarás en el dashboard

| Zona | Qué hace |
|------|----------|
| **Métricas superiores** | Muestran cuántos registros hay, cuántas variables, cuántos valores faltantes y si hay filas repetidas. |
| **Filtros (panel izquierdo)** | Permitir elegir categorías concretas (por ejemplo, ciertos países o niveles de ingresos). El resto del panel se actualiza al instante. |
| **Pestañas de gráficos** | *Histograma* (cómo se reparte una variable), *Boxplot* (mediana y valores atípicos), *Dispersión* (relación entre dos variables) y *Correlación* (mapa de calor de relaciones). |
| **Estadísticas descriptivas** | Tabla resumen: promedio, desviación, mínimos, máximos y cuartiles de cada variable numérica. |
| **Tabla "Datos"** | Los datos completos; puedes ordenar haciendo clic en los títulos de columna y buscar dentro de la tabla. |
| **Botón "Descargar CSV filtrado"** | Descarga un archivo de Excel/CSV con únicamente las filas visibles tras aplicar los filtros. |

---

## 3. Cómo usar tus propios datos

Por defecto el dashboard carga el primer archivo encontrado en la carpeta
`data/raw/`. Para analizar otro archivo:

1. Mira el **panel izquierdo**, sección *"O sube tu propio archivo"*.
2. Haz clic en **Browse files** y selecciona tu CSV o Excel.
3. El panel se recargará automáticamente con tus datos.

---

## 4. Preguntas frecuentes

- **¿Puedo romper algo?** No. Todo lo que hagas (filtros, descargas) ocurre solo
  en pantalla; el archivo original queda intacto.
- **¿Qué significa un valor rojo o extremo en el boxplot?** Son puntos que se
  alejan mucho del resto (posibles valores atípicos); conviene revisarlos.
- **En el mapa de correlación, ¿qué significan los colores?** Rojo fuerte =
  relación positiva (suben juntos), azul fuerte = relación negativa (uno sube
  cuando el otro baja), blanco = sin relación aparente.
- **El dashboard no abre / sale un error.** Verifica que escribiste bien el
  comando del paso 1 y que estás en la carpeta correcta del proyecto. Si el
  problema persiste, captura el mensaje de la terminal y consúltalo.

---

## 5. Requisitos previos (solo la primera vez)

Tener instalado el proyecto con su entorno virtual (carpeta `.venv`) y las
dependencias `streamlit` y `plotly`, ya incluidas y verificadas en este equipo.
