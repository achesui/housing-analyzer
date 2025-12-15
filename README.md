# Analziador de viviendas

una implementacion simple de una red neuronal en Rust para predecir precios de casas.
construida desde cero (sin librerias) utilizando regresión lineal y descenso del gradiente, el modelo toma metros cuadrados y cantidad de habitaciones para hacer las predicciones

## TODO

- cargar datos desde un archivo CSV para soportar datasets mas grandes
- implementar capas ocultas para convertirlo en un perceptron multicapa capaz de aprender patrones no lineales (como clasificacion, modelado de curvas de precios complejas o entender interacciones entre variables)
- guardar y cargar los pesos del modelo utilizando serializacion para evitar re entrenar cada vez
- crear una interfaz de línea de comandos interactiva para probar predicciones manualmente
- implementar un archivo wasm para que pueda ser utilizado en el navegador web utilizando la CPU/GPU
