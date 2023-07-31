# QR Code AI Art Generator
Por [José Gabriel Navarro Comabella](https://github.com/TTigre)
# Natasquad Hackathon
Este repositorio ha sido realizado para presentarse al hackathon de Natasquad. Para más información dirigirse al [sitio del hackathon](https://hackathon.natasquad.com/)

# Notas importantes

## GPU

El repositorio que se le presenta fue realizado y probado utilizando un entorno de ejecución de Google Colab con GPU T4. Verifique que su entorno de ejecución tiene una GPU disponible.

## Dependiente del prompt

Los modelos generativos tienen sus propias limitaciones. El resultado es fuertemente dependiente del prompt y el elemento artístico de la generación puede no ser satisfactorio en todos los casos. Se recomienda hacer un poco de Prompt Engineering para mejores resultados. Usualmente funciona mejor con elementos que puedan tener forma de código QR, los que tienen formas muy específicas suelen presentar peores resultados. Los prompts de *birdview* suelen tener buenos resultados

## Tiempo de ejecución 

Se espera que la generación se demore aproximadamente 5 minutos. Ante más pasos se demorará más, pero debido a que por lo menos genera 5 imágenes, la demora mínima sería aproximadamente 2 minutos.

## Idioma

El texto en este notebook se encuentra en español por la naturaleza del hackaton, sin embargo los nombres de variables, comentarios, etc se encuentran en inglés por convenio.

# Descripción del problema 6: QR Code Design

## Introducción

Las IA de generación de imágenes se refieren a una clase de algoritmos de inteligencia artificial conocidos como modelos generativos. Estos modelos se entrenan con grandes cantidades de datos visuales y aprenden a generar nuevas imágenes que imitan la distribución de los datos de entrenamiento. Modelos recientes como Midjourney y Stable Difussion han demostrado una impresionante habilidad para generar imágenes de alta calidad y detalladas, abriendo nuevas posibilidades en una variedad de campos. En este problema los participantes deberán desarrollar un sistema utilizando Inteligencia Artificial generativa para transformar códigos QR estándar en imágenes más estéticamente agradables, preservando al mismo tiempo su funcionalidad.

## Importancia

La generación de imágenes mediante IA puede ser de gran importancia para las empresas en varios sectores. Para los diseñadores, puede servir como una valiosa herramienta de asistencia creativa, generando nuevos conceptos y diseños basados en parámetros y estilos definidos. En el campo del marketing, puede utilizarse para crear contenido visual para campañas publicitarias o para generar variaciones de imágenes de productos para pruebas A/B. Además, la generación de imágenes mediante IA también puede ser útil para la creación de contenido en medios de comunicación y entretenimiento, como videojuegos y películas. La capacidad de generar rápidamente una gran cantidad de imágenes detalladas y realistas puede ahorrar tiempo y recursos, y permitir un grado de personalización y variación que sería difícil de lograr de otra manera.

## Desarrollo del problema técnico

Se quiere desarrollar un sistema utilizando Inteligencia Artificial generativa para transformar códigos QR estándar en imágenes más estéticamente agradables, preservando al mismo tiempo su funcionalidad.

Visión general: Los códigos QR están en todas partes en el entorno digital de hoy, ofreciendo una solución simple de escaneo y uso para acceder a contenido digital. Sin embargo, su diseño carece de atractivo estético y generalmente no se alinea con la marca o el lenguaje de diseño de los materiales en los que se colocan. Este proyecto desafía el desarrollo de un sistema que emplea Inteligencia Artificial generativa para rediseñar códigos QR en imágenes visualmente atractivas que sean congruentes con la marca asociada o el tema de diseño, todo mientras mantiene la funcionalidad del código QR.

El sistema debe generar imágenes de alta calidad que incorporen el código QR de una manera visualmente atractiva, pero que aún permita que el código sea escaneado de manera efectiva.

El sistema debe ofrecer un grado de personalización, permitiendo a los usuarios especificar ciertos aspectos estéticos, como esquemas de colores, elementos gráficos o estilos que se alineen con su marca o el contexto específico en el que se utilizará el código QR.

El sistema debe asegurarse de que los códigos QR resultantes sigan siendo funcionales y compatibles con los lectores de códigos QR estándar.

El sistema debe ser escalable para soportar altos volúmenes de transformaciones de códigos QR y lo suficientemente adaptable para ser integrado en diversas plataformas y aplicaciones digitales.

# Proceso

## Conocimiento previo

La primera vez que escuché del uso de la IA generativa como la solicitada fue a través del canal DotCSV, que siempre resumen lo más actual e interesante de las nuevas tecnologías (Ver [CONTROLAR a Stable Diffusion, más fácil que nunca con ControlNet! (+Tutorial)](https://youtu.be/owHKDZMoWIM))

## Primer acercamiento

Inicialmente mientras esperaba por la API key de [stability.ai](https://stability.ai/) intenté realizar una implementación guíandome por este post que encontré: [How to generate a QR code with Stable Diffusion](https://stable-diffusion-art.com/qr-code/). En general en mi búsqueda encontré varias publicaciones similares, pero esta parecía la mejor.

Se logró alcanzar resultados, pero no eran muy satisfactorios o fáciles de escanear. Obtener algo satisfactorio requería mucho juego con los parámetros y se hacía necesario realizar demasiado cherry picking.

## Con la API Key

Al obtener la API Key se abrieron nuevas posibilidades y probablemente con mejores y mayores recursos que los que ya estaba utilizando.

Sin embargo, aunque la API abrió nuevas puertas y fue bastante fácil implementar una solución, esta tendía a ser demasiado simple para poder ser escaneable, muy poco artística. Esto se debía principalmente a no poder utilizar un modelo de ControNet pre-entrenado.

## La vía final

Paralelamente a las vías anteriores estuve utilizando una IA con LLM para buscar la web y obtener explicaciones (Lo cuál se puede ver en [perplexity.ai](https://www.perplexity.ai/search/8275e2c3-f1fa-4aa7-afa7-bba52d0d74de)). Aunque el código que proveía probaba ser insuficiente o incorrecto, sirvió como guía para entender mejor algunos elementos. Además de eso Perplexity incluye los resultados de las búsquedas que realiza y entre ellos encontré [QR Code AI Art Generator](https://huggingface.co/spaces/huggingface-projects/QR-code-AI-art-generator). Este repositorio sí estaba bastante bien hecho, así que lo cloné y manos a la obra.

Lo mejor que tiene esta nueva vía es que tiene bien planteados los requerimientos, así que se sabe exactamente que instalar para poder usarlo, punto en el que fallaban las vías anteriores. Además de eso referencia un modelo de ControlNet ya pre-entrenado para esta tarea.

Sin embargo tiene algunos elementos faltantes. Tiene demasiados hiperparámetros que cambiar en búsqueda de un mejor resultado. No da un resultado necesariamente escaneable. Para solucionar estos problemas se investigaron los hiperparámetros y se escogió el que más parece afectar el equilibrio entre artístico y escaneable, y utilizándolo para buscar la imagen más artística posible que sea escaneable. Así que esto funciona out of the box.

Sin embargo esta solución tuvo también su problemática: Funcionaba demasiado bien; las imágenes eran muy artísticas y eran escaneables pero en muchos casos con elevada dificultad, dependiendo del lector utilizado. Así que se decidió devolver la imagen más artística junto con algunas que prioricen prioricen más la legibilidad del código QR, para que el usuario escoja la que considere mejor.

Es una solución además modificable. Se le puede cambiar fácilmente el modelo de Stable Diffusion a utilizar, así como el de ControlNet. Sin embargo funciona bastante bien sin mucha complicación, pudiera decirse que es apta para el uso por usuarios.

# Uso del sistema

El proyecto cuenta con un [Notebook](QR_Natasquad.ipynb) y un [Archivo de python](app.py) que sirven para ejectar el sistema. Sin embargo la forma recomendada de probarlo rápidamente y tener acceso al hardware necesario es utilizando el [notebook que se utilizó durante el desarrollo](https://colab.research.google.com/drive/1t1b0FL27WFGQ4aJCNoSnvp9iOoXChyFh?usp=sharing)

# Posibles mejoras

## Paralelización

La computación del modelo pudiera paralelizarse o utilizarse mejor hardware para acelerar su rendimiento. Un caso susceptible de paralelización sería la generación de las imágenes más probables de escanear en el último paso.

## Otros Modelos

Otra opción sería utilizar otros modelos de Stable Diffusion y de ControlNet, incluso hacer fine-tunning para ajustar los resultados a lo deseado.

## Aprovechar prompts pasados

También sería posible guardar los resultados de la búsqueda binaria relacionado a los prompts anteriores o un vector que lo represente para tal vez saltarse este paso por completo en caso que se haya utilizado un prompt parecido en el pasado.

## Diferentes bibliotecas de escaneo de código QR

Utilizar conjuntamente varias bibliotecas diferentes para escanear el código QR, para así garantizar que el resultado sea escaneable con menor dificultad