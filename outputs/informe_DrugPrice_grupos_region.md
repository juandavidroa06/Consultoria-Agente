# INFORME DE CONSULTORÍA ESTADÍSTICA DE DATASET — PAPERSTATS

**Archivo**: `DataFrame_en_memoria`  
**Dimensiones**: 73 filas, 8 columnas  

---

## 1. RESUMEN Y CALIDAD DE DATOS

- **Valores Faltantes Totales**: 159
- **Filas Duplicadas**: 0

---

## 2. EVALUACIÓN Y ESTADO DE SUPUESTOS ESTADÍSTICOS

- **Independencia de las observaciones**: `Supuesto no evaluado / Pendiente de verificación`
- **Normalidad de la variable 'Cannabis'**: `Evaluado mediante diagnóstico contextual (Se observan desviaciones de normalidad (Shapiro-Wilk p = 1.3074e-15) o presencia de atípicos (10 valores) con asimetría de 5.46.)`
- **Homocedasticidad entre grupos de 'Region'**: `Evaluado mediante prueba de Levene (p = 1.6635e-01)`

---

## 3. RECOMENDACIONES METODOLÓGICAS Y JUSTIFICACIÓN

### Kruskal-Wallis H (No Paramétrica)
- **Justificación Estadística**: Se comparan 5 grupos independientes. Al no cumplirse la normalidad o presentar atípicos severos, se recomienda la prueba no paramétrica de Kruskal-Wallis sobre los rangos de las observaciones.
- **Advertencia**: *Nota: La existencia de diferencias estadísticamente significativas entre grupos NO implica una relación de causalidad.*



---

## 4. EXPLICACIÓN PEDAGÓGICA Y RECOMENDACIONES DE ESTUDIO

### Explicación Metodológica para el Estudiante de Estadística

1. **Enfoque de Decisión Metodológica**:
   En consultoría estadística, las pruebas no se seleccionan mediante reglas rígidas de p-valor.
   Se evalúa en conjunto el tamaño muestral (n), la distribución de los datos, los valores atípicos y los supuestos teóricos.

2. **Teorema del Límite Central (TLC)**:
   Recordatorio fundamental: El TLC establece que para muestras de tamaño adecuado, la distribución de la media muestral
   tiende a ser normal. Esto otorga mayor robustez a pruebas paramétricas frente a desviaciones moderadas, pero NO implica
   que los datos originales sean normales ni justifica ignorar atípicos severos.

3. **Evaluación de Supuestos**:
   - **Independencia de las observaciones**: Supuesto no evaluado / Pendiente de verificación.
   - **Normalidad de la variable 'Cannabis'**: Evaluado mediante diagnóstico contextual (Se observan desviaciones de normalidad (Shapiro-Wilk p = 1.3074e-15) o presencia de atípicos (10 valores) con asimetría de 5.46.).
   - **Homocedasticidad entre grupos de 'Region'**: Evaluado mediante prueba de Levene (p = 1.6635e-01).

4. **Recomendaciones y Causalidad**:
   - **Kruskal-Wallis H (No Paramétrica)**: Se comparan 5 grupos independientes. Al no cumplirse la normalidad o presentar atípicos severos, se recomienda la prueba no paramétrica de Kruskal-Wallis sobre los rangos de las observaciones.
     *Nota: La existencia de diferencias estadísticamente significativas entre grupos NO implica una relación de causalidad.*


---

> [!IMPORTANT]
> *Los análisis cuantitativos indican patrones y diferencias estadísticamente significativas en los datos analizados. La significancia estadística no implica causalidad.*
