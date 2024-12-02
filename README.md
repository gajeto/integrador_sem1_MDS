# Proyecto Integrador MCDA - EAFIT Semestre 2024-2

Modelo analítico predictivo para estimar los dias de estancia hospitalaria como herramienta para optimizar recursos en el Hospital San Vicente Fundación

## Presentado por:
- Gustavo Andrés Rubio Castillo
- Juan Pablo Bertel Morales
- Gustavo Adolfo Jerez Tous

# CONSIDERACIONES

1. Por limitaciones en el tamaño de archivo permitido por Github, se carga el notebook ejeuctado en PDF, y el notebook propio de desarrollo .ipynb en limpio para su ejecución. El tamaño elevado del notebook ejeuctado se debe a la dimensionalidad de la data procesada y a los gráficos interactivos implementados que facilitaron el análisis durante el proceso de modelación.
2. Para un adecuado funcionamiento del notebook, se sugiere ejecutar el código en un ambiente de desarrollo con buena capacidad de memoria RAM (más de 20GB). En el proceso, se inició el desarrollo en local con Jupyter pero no fue posible continuar porque la cardinalidad de las variables categóricas obligaba a probar codificaciones que aumentaban demasiado la dimensionalidad del dataset y consumían todos los recursos del computador. El desarrollo final fue implementado en Google Colab PRO donde se contaba con más de 50GB de memoria RAM.
3. No se consideró una implementación de la solución modelada en AWS dado que el modelo ajustado es una semilla para un desarrollo a futuro más robusto en el Hospital San Vicente Fundación. La infraestructura tecnólogica del hospital, está soportada por bases de datos relaciones y ARP como SAP por lo que se entrega el objeto modelo PKL para que pueda ser utilizado en el ambiente productivo del hospital. 


