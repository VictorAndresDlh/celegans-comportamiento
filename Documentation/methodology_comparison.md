# Comparación Metodológica: Papers Originales vs. Implementación

## 1. Análisis de Vuelo de Lévy (Moy et al. 2015)

### 1.1 Qué hace el paper

Moy et al. (2015) desarrolla un método para distinguir entre estrategias de búsqueda Browniana y vuelo de Lévy en *C. elegans*:

1. **Detección de eventos de giro**: Identificar puntos donde el cambio en dirección excede 40°
2. **Cálculo de longitudes de paso**: Medir distancia euclidiana entre eventos de giro consecutivos
3. **Ajuste de distribución de ley de potencias**: Usar máxima verosimilitud para estimar el exponente α y el valor mínimo xmin
4. **Comparación de modelos**: Calcular el radio de log-verosimilitud (R) comparando power-law vs lognormal

**Criterio de clasificación estadística**:
- R > 0 y p < 0.05 indica patrón tipo Lévy
- **No menciona correcciones por comparaciones múltiples**

El exponente α caracteriza el tipo de búsqueda:
- 1 < α < 3: rango de vuelo de Lévy
- α ≈ 2: búsqueda óptima según teoría

### 1.2 Qué datos usa el paper

- Trayectorias de centroide de *C. elegans* individuales
- **Muestreo temporal**: 1 Hz (Δt = 1 segundo) - crítico para evitar sobresampling
- Duración mínima de 20 minutos de grabación
- Experimentos en presencia y ausencia de alimento
- Cepa N2 wild-type

### 1.3 Qué hace nuestra implementación

1. Calcula vectores de movimiento entre posiciones consecutivas
2. Calcula ángulos de dirección del movimiento
3. Detecta giros donde el cambio angular absoluto excede 40°
4. Calcula longitudes de paso como distancias euclidianas entre puntos de giro
5. Ajusta distribución power-law por máxima verosimilitud
6. Compara power-law vs lognormal usando radio de log-verosimilitud
7. Clasifica como "Lévy-like" si R > 0 y p < 0.05

**Datos utilizados**:
- Trayectorias de WMicrotracker SMART (muestreo nativo 1 Hz)
- 7 cepas de *C. elegans*
- Múltiples tratamientos cannabinoides

### 1.4 Diferencias principales

#### Alineamiento metodológico
- ✅ Umbral de 40° idéntico al paper
- ✅ Muestreo a 1 Hz coincide con especificación del paper
- ✅ Método de ajuste por máxima verosimilitud
- ✅ Comparación power-law vs lognormal con radio de log-verosimilitud
- ✅ Criterio de significancia R > 0, p < 0.05

#### Diferencias en aplicación
- **Población**: Paper usa solo N2; implementación analiza 7 cepas
- **Condiciones**: Paper compara presencia/ausencia de alimento; implementación evalúa cannabinoides
- **Escala**: Implementación procesa múltiples cepas × tratamientos (7 × ~8 = ~56 pruebas estadísticas)

#### Problema matemático identificado

**Normalización de ángulos**: La implementación normaliza diferencias angulares a rango [-π, π] antes de tomar valor absoluto. Esto puede generar errores en transiciones que cruzan el límite ±180°.

**Ejemplo del problema**: Un cambio de heading de 179° a -179° representa un cambio real de 2°, pero:
- Diferencia cruda: -179° - 179° = -358°
- Normalizada a [-π, π]: -358° → 2° (correcto)
- Pero si se normaliza DESPUÉS de otras operaciones, puede computarse como 358°

El paper no especifica explícitamente el procedimiento de normalización angular.

#### Validación estadística

El paper de Moy et al. **no menciona correcciones por comparaciones múltiples**. El análisis reportado compara 2-3 condiciones (food vs no food, diferentes genotipos).

Nuestra implementación tampoco aplica correcciones. Con ~56 pruebas estadísticas (7 cepas × 8 tratamientos) y α=0.05 sin corrección, se esperarían ~2.8 clasificaciones "Lévy-like" por azar puro bajo la hipótesis nula.

---

## 2. Análisis Topológico de Datos - TDA (Thomas et al. 2021)

### 2.1 Qué hace el paper

Thomas et al. (2021) aplica homología persistente para clasificar trayectorias conductuales:

1. **Embedding de ventana deslizante**: Convierte trayectoria temporal en nube de puntos en espacio de alta dimensión. Una ventana de L puntos en 2D se representa como punto en espacio 2L-dimensional
2. **Centrado de trayectoria**: Resta la posición media para remover deriva espacial
3. **Complejo de Vietoris-Rips**: Construcción topológica sobre la nube de puntos
4. **Homología persistente**: Calcula características topológicas que persisten a través de múltiples escalas:
   - H₀: componentes conectadas
   - H₁: bucles/ciclos
   - H₂: cavidades
5. **Paisajes de persistencia**: Convierte diagramas de persistencia en vectores numéricos para machine learning
6. **Clasificación**: Usa Support Vector Machines o Random Forest con los paisajes como features

**El paper prueba diferentes tamaños de ventana** (L = 10, 20, 50 puntos) y encuentra que el óptimo **depende del dataset**.

### 2.2 Qué datos usa el paper

- Tres datasets de validación:
  1. *C. elegans* N2 en medio líquido vs agar
  2. *C. elegans* mutante *tax-4* vs wild-type
  3. *Drosophila melanogaster* en diferentes condiciones
- Trayectorias de centroide
- Datos previamente publicados de cada sistema

### 2.3 Qué hace nuestra implementación

1. Centra cada trayectoria sustrayendo posición media
2. Aplica ventanas deslizantes de **tamaño fijo L=20 puntos**
3. Cada ventana se representa como punto en espacio 40-dimensional
4. Construye complejo de Vietoris-Rips hasta dimensión 2
5. Calcula homología persistente (H₀, H₁, H₂)
6. Extrae paisajes de persistencia, principalmente de H₁ (bucles)
7. Usa Support Vector Machines para clasificación binaria (Control vs cada tratamiento)

**Datos utilizados**:
- Trayectorias de 7 cepas × múltiples tratamientos cannabinoides
- Análisis por cepa: cada clasificador es Control vs un tratamiento específico

### 2.4 Diferencias principales

#### Alineamiento metodológico
- ✅ Pipeline topológico idéntico (embedding → Vietoris-Rips → persistencia → paisajes)
- ✅ Centrado de trayectorias coincide con el paper
- ✅ Uso de Support Vector Machines coincide con uno de los clasificadores del paper
- ✅ Dimensión del complejo (dim 2) coincide con el paper

#### Diferencias clave

**Tamaño de ventana**:
- **Paper**: Prueba L ∈ {10, 20, 50} y selecciona el mejor para cada dataset
- **Implementación**: Usa L=20 fijo para todos los casos

**Clasificadores**:
- **Paper**: Compara SVM vs Random Forest, reporta resultados de ambos
- **Implementación**: Solo usa SVM

#### Validación estadística

El paper de Thomas et al. reporta accuracy, precision, recall y F1-score. **No menciona**:
- Pruebas de permutación
- Intervalos de confianza
- Correcciones por comparaciones múltiples

Nuestra implementación reporta las mismas métricas (accuracy y F1), coincidiendo con el enfoque del paper.

---

## 3. Screening por Machine Learning (García-Garví et al. 2025)

### 3.1 Qué hace el paper

García-Garví et al. (2025) desarrolla un pipeline de screening toxicológico con **validación estadística rigurosa**:

#### Extracción de features

Usa sistema de tracking multi-esqueleto para obtener **256 features**:
- **90 features de cinemática del centroide**: velocidad, aceleración, cambios de dirección
- **166 features de morfología y esqueleto**: curvatura corporal, longitud, ángulos de segmentos
- **Features temporales**: frecuencias, duraciones, amplitudes de comportamientos específicos

#### Preprocesamiento estadístico (CRÍTICO)

1. **Pruebas de permutación por bloques**: 10,000 permutaciones por feature
   - Cada experimento es un "bloque" con control interno
   - Permuta etiquetas dentro de cada bloque
   - Calcula t-estadístico para cada permutación

2. **Corrección de Benjamini-Yekutieli**: Controla tasa de falso descubrimiento (FDR < 0.1)
   - Más conservadora que Benjamini-Hochberg
   - Asume dependencia arbitraria entre features

3. **Selección de features**: Solo usa features estadísticamente significativos para clasificación

#### Clasificación (Nivel 1)

- Reduce dimensionalidad aplicando PCA sobre features significativos
- Random Forest con validación cruzada
- Objetivo: Distinguir Tóxico vs No-tóxico

#### Clasificación (Nivel 2)

- Solo para compuestos clasificados como tóxicos
- Clasifica tipo de toxicidad (neurotóxica, oxidativa, etc.)

### 3.2 Qué datos usa el paper

- Tracking multi-esqueleto con análisis de contorno corporal completo
- **256 features** por gusano
- **Diseño experimental por bloques**: cada experimento incluye control negativo, control positivo y compuestos test
- Cepa N2 wild-type
- Panel de compuestos con toxicidad conocida para validación

### 3.3 Qué hace nuestra implementación

#### Extracción de features

Calcula **13 features cinemáticas** del centroide únicamente:
- **Velocidad (7 features)**: media, desviación estándar, mínimo, cuartiles 25%, 50%, 75%, máximo
- **Ángulos de giro (3 features)**: media del valor absoluto, desviación estándar, máximo absoluto
- **Trayectoria (3 features)**: longitud total, desplazamiento neto, ratio de confinamiento

#### Clasificación

1. Normaliza features estandarizando (media 0, varianza 1)
2. Random Forest para clasificación binaria
3. División 70% entrenamiento, 30% prueba
4. Reporta accuracy y F1-score por clase

#### No incluye

- Pruebas de permutación previas a clasificación
- Corrección de Benjamini-Yekutieli
- Selección de features significativos antes de clasificar
- Análisis de morfología de esqueleto

### 3.4 Diferencias principales

#### Diferencias cuantitativas críticas

**Número de features**:
- **Paper**: 256 features (90 cinemática + 166 morfología)
- **Implementación**: 13 features (solo cinemática básica)
- **Reducción**: 95% de features eliminadas

**Hallazgo crítico del paper**: García-Garví et al. reportan explícitamente que:
> "Intentamos usar solo features cinemáticos básicos del centroide y **no detectamos ningún compuesto tóxico** al aplicar corrección estadística rigurosa. Solo al expandir a 256 features incluyendo morfología del esqueleto logramos identificar compuestos tóxicos con significancia estadística."

Esto sugiere que nuestra implementación probablemente tiene **bajo poder estadístico** para detectar efectos sutiles.

#### Diferencias en validación estadística (CRÍTICAS)

**Pipeline del paper (orden estricto)**:
1. **Primero**: Pruebas de permutación por feature (10,000 permutaciones × N_features)
2. **Segundo**: Corrección de Benjamini-Yekutieli (FDR < 0.1)
3. **Tercero**: Clasificación SOLO con features significativos

**Pipeline de implementación**:
1. Clasificación directa con todas las 13 features
2. Sin pruebas de significancia estadística previas
3. Sin correcciones por comparaciones múltiples

#### Alineamiento metodológico

- ✅ Uso de Random Forest coincide con el paper
- ✅ Normalización de features (estandarización)
- ❌ **Ausencia TOTAL del pipeline de validación estadística**
- ❌ **Reducción masiva del espacio de features (256 → 13)**
- ❌ Sin análisis de morfología de esqueleto

#### Implicaciones

Sin validación estadística previa:
- Un accuracy del 70% podría deberse a overfitting, no a diferencias reales
- No sabemos si las features usadas son significativamente diferentes entre grupos
- Sin corrección múltiple, ~1/20 comparaciones será "significativa" por azar (p<0.05)

---

## 4. Descubrimiento de Estados Conductuales No Supervisado

### 4.1 Metodología implementada

Esta metodología **NO tiene un paper de referencia específico válido**.

**Aclaración importante sobre Koren et al. 2015**: El paper "Model-Independent Phenotyping of C. elegans Locomotion Using Scale-Invariant Feature Transform" por Koren et al. (2015) **NO usa el enfoque implementado**.

#### Qué hace realmente Koren et al. 2015

- Usa **SIFT (Scale-Invariant Feature Transform)** - descriptores de visión por computadora
- **NO extrae features cinemáticas** (velocidad, curvatura, etc.)
- Trabaja directamente con imágenes crudas
- Construye un "vocabulario visual" mediante k-means clustering de descriptores SIFT
- Representa cada video como histograma de "palabras visuales" (descriptores SIFT frecuentes)
- Compara videos por distancia euclidiana entre histogramas
- Es completamente "model-independent" - no requiere definir features específicas del gusano

#### Qué hace nuestra implementación (DIFERENTE a Koren 2015)

1. **Extracción de features cinemáticas**: Mismas 13 features que ML Screening
2. **Normalización**: Estandarización (media 0, varianza 1)
3. **Reducción de dimensionalidad**: PCA (retiene componentes que explican 95% de varianza)
4. **Clustering**: Modelos de Mezcla Gaussiana (GMM) con matriz de covarianza completa
5. **Selección de K**: Minimiza Criterio de Información Bayesiano (BIC), K entre 2 y 10
6. **Interpretación post-hoc de estados**:
   - **Estado de pausa**: Estado con menor velocidad promedio
   - **Estado de crucero/activo**: Estado con mayor desplazamiento neto
7. **Análisis de distribución**: Porcentaje de gusanos en cada estado por tratamiento

### 4.2 Análisis crítico sin paper de referencia

#### Naturaleza de la metodología

Esta es una **aplicación estándar de técnicas de clustering no supervisado** (GMM + PCA + BIC) comúnmente usadas en análisis de datos conductuales, pero:

- **No sigue ningún paper específico publicado** para *C. elegans*
- No está validada contra un gold standard
- Las decisiones metodológicas (K entre 2-10, BIC, 95% varianza en PCA) son arbitrarias

#### Ausencia de validación estadística

**No se reportan**:
- Pruebas de significancia para diferencias en ocupación de estados entre tratamientos
- Intervalos de confianza para porcentajes de ocupación
- Correcciones por comparaciones múltiples (múltiples estados × múltiples tratamientos)

**Ejemplo**: Si se reporta "Tratamiento X aumenta estado de pausa de 20% a 35%", no sabemos:
- Si 15% de diferencia es estadísticamente significativa
- Qué tan variable es esta medida (intervalo de confianza)
- Si sobrevive corrección por comparaciones múltiples

#### Limitaciones conceptuales

1. **BIC no garantiza validez biológica**: Selecciona modelo que balancea ajuste estadístico vs complejidad, pero no valida que:
   - Los estados sean reproducibles entre experimentos
   - Los estados tengan significado biológico distinto
   - K óptimo sea estable (bootstrap)

2. **Interpretación circular**: Llamar "pausa" al estado de baja velocidad es post-hoc. No valida que:
   - La pausa sea un estado discreto vs extremo de un continuo
   - Exista bimodalidad en distribución de velocidades
   - Las transiciones entre estados sean raras (evidencia de estados discretos)

3. **Dependencia de features**: Con solo 13 features y PCA al 95%, probablemente se usan ~5-8 componentes principales. Expandir a más features (como sugiere García-Garví) cambiaría completamente el clustering.

### 4.3 Comparación imposible

**Conclusión**: No se puede hacer una comparación metodológica formal contra un paper de referencia porque:
- Koren 2015 usa una metodología completamente diferente (SIFT, no features cinemáticas)
- No existe un paper publicado que haga exactamente GMM + PCA sobre features cinemáticas de trayectorias de centroide de *C. elegans*

La implementación es una **metodología ad-hoc** usando herramientas estándar de clustering, sin validación formal publicada.

---

## Resumen Comparativo Global

### Lévy Flight (Moy et al. 2015)
- **Fidelidad metodológica**: Alta (>95%)
- **Alineamiento algorítmico**: Casi perfecto
- **Problema identificado**: Posible error en normalización de ángulos al cruzar límite ±180°
- **Validación estadística**: Coincide con paper (usa R y p-value; no correcciones múltiples)
- **Escala**: Paper analiza 2-3 condiciones; implementación ~56 pruebas sin corrección

### TDA (Thomas et al. 2021)
- **Fidelidad metodológica**: Alta (>90%)
- **Alineamiento algorítmico**: Excelente
- **Diferencia principal**: Tamaño de ventana fijo (L=20) vs optimización por dataset
- **Validación estadística**: Coincide con paper (accuracy, precision, recall, F1)
- **Limitación**: No explora hiperparámetro crítico (tamaño de ventana)

### ML Screening (García-Garví et al. 2025)
- **Fidelidad metodológica**: Muy baja (~20%)
- **Diferencias críticas**:
  - **Features**: 256 → 13 (95% reducción)
  - **Validación estadística**: Completa (permutación + BH/BY) → Ninguna (0%)
  - **Morfología**: Esqueleto completo → Solo centroide
- **Hallazgo del paper**: Features reducidas similares a las nuestras = cero detecciones
- **Implicación**: Implementación probablemente tiene bajo poder estadístico

### Estados No Supervisados (Sin paper de referencia)
- **Fidelidad metodológica**: No aplica
- **Paper citado incorrectamente**: Koren 2015 usa SIFT (visión por computadora), no GMM sobre features cinemáticas
- **Naturaleza**: Metodología ad-hoc usando técnicas estándar (GMM + PCA + BIC)
- **Validación estadística**: Ausente (no pruebas de significancia, no correcciones múltiples)
- **Limitaciones**: Interpretación post-hoc, sin validación de reproducibilidad, sin gold standard

---

## Conclusiones Generales

### Metodologías bien implementadas
1. **Lévy Flight**: Fidelidad alta, salvo posible bug de normalización angular
2. **TDA**: Fidelidad alta, implementación sólida del pipeline topológico

### Metodologías con limitaciones severas
3. **ML Screening**: Reducción masiva de features (256→13) y ausencia total de validación estadística
4. **Estados No Supervisados**: Sin paper de referencia válido, metodología ad-hoc sin validación

### Hallazgos críticos transversales

1. **Ausencia sistemática de validación estadística rigurosa**:
   - Ninguna metodología aplica correcciones por comparaciones múltiples
   - Solo García-Garví 2025 menciona pruebas de permutación + corrección (no implementado)
   - No se reportan intervalos de confianza ni pruebas de significancia

2. **Problema de features reducidas**:
   - García-Garví 2025 reporta explícitamente que features cinemáticas básicas (como las nuestras) no detectan efectos tras corrección estadística
   - Solo al expandir a 256 features (morfología de esqueleto) logran detecciones significativas

3. **Escalamiento sin ajuste estadístico**:
   - Moy 2015: 2-3 comparaciones → Implementación: ~56 pruebas (sin corrección)
   - Múltiples metodologías × múltiples cepas × múltiples tratamientos = cientos de pruebas estadísticas implícitas
