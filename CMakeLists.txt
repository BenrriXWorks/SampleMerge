cmake_minimum_required(VERSION 3.14) # Recomendado poner una versión mínima
project(TesisSimd CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Highway ----
include(FetchContent)

FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        1.2.0
)

set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(TLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)
add_subdirectory(extern/tlx)

# ---- Dependencias ----
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

add_executable(main main.cpp)

# ---- Librerías ----
target_link_libraries(main PRIVATE
   hwy
   hwy_contrib
   OpenMP::OpenMP_CXX
   TBB::tbb
)

target_include_directories(main PRIVATE extern/tlx)

# ---- Flags de Compilación (Compile Options) ----
target_compile_options(main PRIVATE
	-O3
  #-O1
  -g
  #-gcodeview
  -gcolumn-info
  -march=native
#  -mavx2
  -fopenmp # <-- OpenMP::OpenMP_CXX ya se encarga de esto, pero dejarlo explicito no daña
  -D_GLIBCXX_PARALLEL 
  -static
#    -fsanitize=address        # <--- Instrumenta el código
#    -fno-omit-frame-pointer   # <--- NECESARIO para ver stack traces útiles
)

# ---- Flags de Enlazado (Link Options) - EL FIX ----
target_link_options(main PRIVATE
#    -fsanitize=address        # <--- Enlaza la librería libasan
)
