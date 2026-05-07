cmake_minimum_required(VERSION 3.18)
project(hurricane_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# Eigen3 — prefer conda-installed location if not found automatically
find_package(Eigen3 REQUIRED NO_MODULE
    PATHS "C:/Users/DELL/miniconda3/Library/share/eigen3/cmake"
    NO_DEFAULT_PATH)

# OpenBLAS — no cmake config from conda, link directly
set(OPENBLAS_INCLUDE_DIR "C:/Users/DELL/miniconda3/Library/include" CACHE PATH "OpenBLAS include dir")
set(OPENBLAS_LIB         "C:/Users/DELL/miniconda3/Library/lib/openblas.lib" CACHE FILEPATH "OpenBLAS import lib")
set(OPENBLAS_DLL         "C:/Users/DELL/miniconda3/Library/bin/openblas.dll"  CACHE FILEPATH "OpenBLAS DLL")

if(NOT EXISTS "${OPENBLAS_LIB}")
    message(FATAL_ERROR "OpenBLAS lib not found at ${OPENBLAS_LIB}. "
        "Run: conda install openblas")
endif()

# Tell Eigen to route large matmul/QR calls through BLAS
add_compile_definitions(EIGEN_USE_BLAS)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W3 /EHsc /MP)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(SOURCES
    src/rsvd.cpp
    src/warm_rsvd.cpp
    src/metrics.cpp
    src/data_loader.cpp
    src/csv_writer.cpp
    src/hurricane_experiment.cpp
)

add_executable(hurricane_bench ${SOURCES})

target_include_directories(hurricane_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENBLAS_INCLUDE_DIR}
)

target_link_libraries(hurricane_bench PRIVATE
    Eigen3::Eigen
    "${OPENBLAS_LIB}"
)

# On Windows, copy OpenBLAS DLL next to the executable so it can be run in-place
if(WIN32 AND EXISTS "${OPENBLAS_DLL}")
    add_custom_command(TARGET hurricane_bench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENBLAS_DLL}"
            "$<TARGET_FILE_DIR:hurricane_bench>"
        COMMENT "Copying openblas.dll to output directory"
    )
endif()

# ---------------------------------------------------------------------------
# Dumb warm-start benchmark (cold vs dumb_warm_rsvd)
# ---------------------------------------------------------------------------
set(DUMB_SOURCES
    src/rsvd.cpp
    src/dumb_warm_rsvd.cpp
    src/metrics.cpp
    src/data_loader.cpp
    src/csv_writer.cpp
    src/hurricane_dumb_experiment.cpp
)

add_executable(hurricane_dumb_bench ${DUMB_SOURCES})

target_include_directories(hurricane_dumb_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENBLAS_INCLUDE_DIR}
)

target_link_libraries(hurricane_dumb_bench PRIVATE
    Eigen3::Eigen
    "${OPENBLAS_LIB}"
)

if(WIN32 AND EXISTS "${OPENBLAS_DLL}")
    add_custom_command(TARGET hurricane_dumb_bench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENBLAS_DLL}"
            "$<TARGET_FILE_DIR:hurricane_dumb_bench>"
        COMMENT "Copying openblas.dll to hurricane_dumb_bench output"
    )
endif()
