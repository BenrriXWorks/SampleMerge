### Introducción

Implementación header only de framework de merge basado en Samplesort y Mergesort como estratégia de paralelización genérica aprovechando instrucciones vectorizadas de highway para fusionar eficientemente.

Memoria: O(n)
Complejidad: O(n/p log n/p)

### Benchmark

Benchmark para AMD Ryzen™ Threadripper™ PRO 5975WX 32-Cores, AVX512, DDR4 128GB 3200mhz:

```r
Timing N=200000013
std::sort (par_unseq):   699.389 ms
SampleMerge(1): 1085.054 ms | sort=1085.054 pivot=0.000 class=0.000 bucket=0.000 merge=0.000 copyback=0.000 ms
SampleMerge(2): 1007.928 ms | sort=582.736 pivot=0.000 class=0.000 bucket=0.000 merge=380.301 copyback=44.842 ms
SampleMerge(3): 1226.763 ms | sort=423.667 pivot=0.011 class=0.003 bucket=0.257 merge=760.237 copyback=42.530 ms
SampleMerge(4): 946.869 ms | sort=327.278 pivot=0.009 class=0.006 bucket=0.366 merge=576.421 copyback=42.726 ms
SampleMerge(6): 697.714 ms | sort=264.750 pivot=0.015 class=0.014 bucket=0.569 merge=389.720 copyback=42.566 ms
SampleMerge(7): 644.638 ms | sort=262.878 pivot=0.461 class=0.019 bucket=0.696 merge=338.013 copyback=42.491 ms
SampleMerge(8): 611.535 ms | sort=267.279 pivot=1.470 class=0.025 bucket=0.774 merge=299.404 copyback=42.507 ms
SampleMerge(9): 597.219 ms | sort=228.481 pivot=0.028 class=0.033 bucket=0.902 merge=329.987 copyback=37.715 ms
SampleMerge(10): 542.625 ms | sort=204.387 pivot=3.473 class=0.035 bucket=0.999 merge=299.522 copyback=34.133 ms
SampleMerge(11): 496.274 ms | sort=185.749 pivot=2.036 class=0.041 bucket=1.098 merge=276.284 copyback=30.979 ms
SampleMerge(12): 470.916 ms | sort=157.210 pivot=5.628 class=4.946 bucket=1.232 merge=269.759 copyback=31.961 ms
SampleMerge(16): 345.356 ms | sort=121.099 pivot=1.439 class=0.063 bucket=1.728 merge=194.799 copyback=26.117 ms
SampleMerge(32): 258.465 ms | sort=74.182 pivot=6.999 class=6.024 bucket=3.185 merge=140.187 copyback=27.801 ms
SampleMerge(36): 261.276 ms | sort=77.566 pivot=8.808 class=7.384 bucket=3.123 merge=138.202 copyback=26.124 ms
SampleMerge(48): 234.888 ms | sort=62.736 pivot=0.252 class=0.660 bucket=4.801 merge=144.934 copyback=21.411 ms
SampleMerge(64): 225.888 ms | sort=55.334 pivot=0.367 class=0.826 bucket=6.672 merge=141.162 copyback=21.413 ms
SampleMerge(128): 267.779 ms | sort=45.463 pivot=0.485 class=1.666 bucket=12.736 merge=186.170 copyback=21.167 ms
```

Benchmark para Ryzen 5 5600X 6-cores, AVX2, 16gb ddr4 3600mhz:

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

Benchmark para Intel Core i5-1135G7 4 cores, AVX512, 8gb ddr4.
```r
11:27:18 root@NB-BenjaminParra build ±|main ✗|→ ./main
OK: correctitud

Timing N=200000013
std::sort (par_unseq):   8216.909 ms
SampleMerge(1): 1614.995 ms | sort=1614.968 pivot=0.000 class=0.000 bucket=0.000 merge=0.000 copyback=0.000 ms
SampleMerge(2): 1837.199 ms | sort=615.438 pivot=0.000 class=0.000 bucket=0.000 merge=1174.155 copyback=46.785 ms
SampleMerge(3): 1646.742 ms | sort=583.026 pivot=0.136 class=0.015 bucket=0.050 merge=1021.319 copyback=41.531 ms
SampleMerge(4): 1244.400 ms | sort=487.919 pivot=0.021 class=0.012 bucket=0.203 merge=717.524 copyback=34.776 ms
SampleMerge(6): 1022.424 ms | sort=434.165 pivot=0.009 class=0.029 bucket=0.140 merge=551.028 copyback=36.135 ms
SampleMerge(7): 961.126 ms | sort=366.798 pivot=1.013 class=0.027 bucket=0.083 merge=558.208 copyback=34.325 ms
SampleMerge(8): 979.159 ms | sort=415.990 pivot=0.131 class=0.025 bucket=0.102 merge=519.305 copyback=43.296 ms
SampleMerge(9): 1248.458 ms | sort=591.750 pivot=0.257 class=0.333 bucket=0.089 merge=618.087 copyback=37.238 ms
SampleMerge(10): 1167.842 ms | sort=372.150 pivot=0.398 class=0.458 bucket=0.096 merge=758.774 copyback=34.615 ms
SampleMerge(11): 1136.181 ms | sort=378.266 pivot=0.170 class=0.475 bucket=0.166 merge=715.999 copyback=40.744 ms
SampleMerge(12): 1195.716 ms | sort=370.931 pivot=0.419 class=0.414 bucket=0.067 merge=778.327 copyback=41.430 ms
SampleMerge(16): 1202.238 ms | sort=385.608 pivot=0.683 class=0.805 bucket=0.105 merge=780.842 copyback=33.538 ms
SampleMerge(32): 936.392 ms | sort=280.279 pivot=0.650 class=0.618 bucket=0.207 merge=621.280 copyback=33.149 ms
SampleMerge(36): 1164.664 ms | sort=326.609 pivot=0.916 class=1.257 bucket=0.352 merge=792.035 copyback=43.252 ms
SampleMerge(48): 1206.880 ms | sort=327.679 pivot=0.563 class=1.273 bucket=0.378 merge=842.905 copyback=33.743 ms
SampleMerge(64): 1155.816 ms | sort=292.576 pivot=2.044 class=2.696 bucket=3.105 merge=812.776 copyback=38.672 ms
```

### Estratégia

Resumen de estratégia actual:
1. Ordena segmentos de tamaño ~n/THREADS con un algoritmo lineal optimo. En este caso usa VQSort.
2. Toma los elementos de las posiciones n/THREADS * segment_id + n/(THREADS^2) * CANDIDATE_ID como candidatos aprovechando el gran orden como heuristica.
3. Selecciona la mediana para cada candidato, quedando THREADS-1 Pivotes.
4. Busqueda binaria de cuantos elementos corresponderian a cada bucket dentro de cada segmento y creacion de los buckets de manera alineada en memoria. Actualmente se crea un arreglo grande con padding de infinitos de |LANES_SIMD| + E, donde E es lo necesario para alinear el siguiente bucket. 
5. Guardar los punteros del inicio y el largo de cada run clasificada a un bucket determinado y guardarlo en un arreglo.
6. Fusionar las runs en el bucket correspondiente. La estratégia actual es hacer un minimo vertical por la cantidad de registros SIMD necesarias para almacenar todas las cabezas de las runs junto con una mascara para determinar cual registro y en que posicion fue, reduciendo con un minimo horizontal para obtener el elemento menor en cada iteración y copiandolo a un buffer alineado. Al llenarse el buffer se hace streaming para no contaminar la cache. Esta estratégia permite escalar a un numero arbitrario de runs.
7. Copiar los buckets de vuelta al arreglo original y lanzar un thread para destruir el arreglo qeu contiene los buckets para limpiar.

### Compilación

Para compilar se necesita cmake, build-essential y make.
Los pasos para la primera compilación son:
mkdir build
cd build
cmake ..
make

Tomar en cuenta que compilar VQSort toma algunos minutos

### Ejecución del benchmark


Y luego la ejecución del entorno del test es ./main
