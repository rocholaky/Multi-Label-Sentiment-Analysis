<body>
  <h1>Multi-Label Sentiment Analysis</h1>
  <p>Este repositorio contiene el código y los recursos necesarios para implementar un modelo de clasificación de sentimientos en comentarios extraídos de diversas redes sociales.</p>
    <h2>Descripción del proyecto</h2>
  <p>En la era digital actual, las plataformas de redes sociales se han convertido en fuentes vitales de información y sentimiento del cliente. Sin embargo, clasificar y analizar manualmente las emociones expresadas en estos comentarios es un proceso laborioso y propenso a errores. Este repositorio presenta una solución para automatizar el análisis de sentimientos y la etiquetación multiple de emociones en comentarios de redes sociales. Al aprovechar técnicas como TF-IDF como codificador de palabras y utilizar clasificadores como XGBoost, Naive Bayes, Regresión Logística y Random Forest, ofrecemos un enfoque eficiente y no computacionalmente costoso para solucionar esta tarea.</p>
  <h2>Contenido del repositorio</h2>
  <ul>
    <li><code>train.csv</code>: archivo que contiene los datos de entrenamiento.</li>
    <li><code>test.csv</code>: archivo que contiene los datos de prueba.</li>
    <li><code>emotions.txt</code>: archivo que contiene la lista de emociones.</li>
    <li><code>SentimentAnalysis.ipynb</code>: cuaderno de Jupyter con el código para construir y entrenar el modelo de multi etiqueta, donde se realiza el EDA además de la explicación de las métricas requeridas.</li>
  </ul>

  <h2>Contenido del cuaderno de Jupyter</h2>
  <p>El cuaderno de Jupyter <code>model.ipynb</code> contiene los siguientes pasos:</p>
  <ol>
    <li>Análisis y preprocesamiento de los datos</li>
    <li>Construcción del modelo de clasificación</li>
    <li>Evaluación del modelo utilizando diferentes métricas (accuracy, F1-score, etc.)</li>
  </ol>

  <h2>Indicador/KPI de satisfacción global</h2>
  <p>Se proporciona el archivo <code>satisfaction_indicator.py</code> que contiene el código para construir un indicador/KPI para medir el nivel de satisfacción global basado en la clasificación de los comentarios.</p>
</body>

