Implementación header only de framework de merge basado en Samplesort y Mergesort como estratégia de paralelización genérica aprovechando instrucciones vectorizadas de highway para fusionar eficientemente.

Memoria: O(n)
Complejidad: O(n/p log n/p)

Resultados actuales (Ryzen 5 5600X, 16gb ddr4 3600mhz):

```r
03:16:10 root@DESKTOP-O8GELGC build → ./main
OK: correctitud

Timing N=200000013
std::sort (par-unseq):   2035.123 ms
SampleMerge(1): 1146.119 ms | sort=1146.119 pivot=0.000 class=0.000 bucket=0.000 merge=0.000 copyback=0.000 ms
SampleMerge(2): 1605.523 ms | sort=687.679 pivot=0.000 class=0.000 bucket=0.000 merge=870.352 copyback=47.398 ms
SampleMerge(3): 1451.562 ms | sort=505.260 pivot=0.015 class=0.007 bucket=0.075 merge=908.262 copyback=37.830 ms
SampleMerge(4): 1152.593 ms | sort=431.321 pivot=0.003 class=0.021 bucket=0.041 merge=680.114 copyback=40.966 ms
SampleMerge(6): 1108.764 ms | sort=514.848 pivot=0.030 class=0.023 bucket=0.065 merge=553.358 copyback=40.288 ms
SampleMerge(7): 924.321 ms | sort=448.478 pivot=0.050 class=0.026 bucket=0.060 merge=433.652 copyback=41.866 ms
SampleMerge(8): 840.635 ms | sort=407.734 pivot=0.049 class=0.033 bucket=0.066 merge=389.941 copyback=42.683 ms
SampleMerge(9): 857.371 ms | sort=417.266 pivot=0.006 class=0.043 bucket=0.070 merge=397.754 copyback=42.116 ms
SampleMerge(10): 792.589 ms | sort=411.738 pivot=0.050 class=0.092 bucket=0.125 merge=338.795 copyback=41.558 ms
SampleMerge(11): 885.226 ms | sort=446.479 pivot=0.186 class=0.082 bucket=0.074 merge=395.048 copyback=43.232 ms
SampleMerge(12): 871.494 ms | sort=442.060 pivot=0.196 class=0.234 bucket=0.080 merge=385.871 copyback=42.956 ms
SampleMerge(16): 801.393 ms | sort=396.945 pivot=0.278 class=0.314 bucket=0.187 merge=360.813 copyback=42.769 ms
SampleMerge(32): 938.760 ms | sort=378.101 pivot=0.644 class=0.937 bucket=0.235 merge=481.394 copyback=77.333 ms
SampleMerge(36): 852.321 ms | sort=364.421 pivot=0.632 class=0.884 bucket=0.176 merge=403.923 copyback=82.227 ms
SampleMerge(48): 766.357 ms | sort=314.676 pivot=0.592 class=1.469 bucket=0.192 merge=369.095 copyback=80.224 ms
SampleMerge(64): 855.074 ms | sort=301.640 pivot=1.038 class=1.823 bucket=0.257 merge=467.721 copyback=82.456 ms
```

Resumen de estratégia actual:
1. Ordena segmentos de tamaño ~n/THREADS con un algoritmo lineal optimo.
2. Toma los elementos de las posiciones n/THREADS * segment_id + n/(THREADS^2) * CANDIDATE_ID como candidatos aprovechando el gran orden como heuristica.
3. Selecciona la mediana para cada candidato, quedando THREADS-1 Pivotes.
4. Busqueda binaria de cuantos elementos corresponderian a cada bucket dentro de cada segmento y creacion de los buckets de manera alineada en memoria. Actualmente se crea un arreglo grande con padding de infinitos de |LANES_SIMD| + E, donde E es lo necesario para alinear el siguiente bucket. 
5. Guardar los punteros del inicio y el largo de cada run clasificada a un bucket determinado y guardarlo en un arreglo.
6. Fusionar las runs en el bucket correspondiente. La estratégia actual es hacer un minimo vertical por la cantidad de registros SIMD necesarias para almacenar todas las cabezas de las runs junto con una mascara para determinar cual registro y en que posicion fue, reduciendo con un minimo horizontal para obtener el elemento menor en cada iteración y copiandolo a un buffer alineado. Al llenarse el buffer se hace streaming para no contaminar la cache. Esta estratégia permite escalar a un numero arbitrario de runs.
7. Copiar los buckets de vuelta al arreglo original y lanzar un thread para destruir el arreglo qeu contiene los buckets para limpiar.

Para compilar se necesita cmake, build-essential y make.
Los pasos para la primera compilación son:
mkdir build
cd build
cmake ..
make

Y luego la ejecución del entorno del test es ./main

Tomar en cuenta que compilar VQSort toma algunos minutos