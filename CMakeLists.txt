# ============================================================================
# latent-oracle - root build configuration
# C++26 with transparent degradation to C++23/C++20 (plan of record, section 1)
# ============================================================================

cmake_minimum_required(VERSION 3.28...4.0)
project(latent-oracle
    VERSION 0.1.0
    DESCRIPTION "Searchless neural chess research engine"
    LANGUAGES C CXX)

# ----------------------------------------------------------------------------
# Options (sanitizer structure adapted from WyattAu/OmniCPP-template)
# ----------------------------------------------------------------------------
option(LO_BUILD_TESTS "Build and register tests" ON)
option(LO_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(LO_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(LO_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(LO_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

if(LO_ENABLE_ASAN AND LO_ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer are mutually exclusive")
endif()

# ----------------------------------------------------------------------------
# C++ standard with transparent degradation
# ----------------------------------------------------------------------------
include(CheckCXXCompilerFlag)
set(LO_STD 26)
foreach(candidate 26 23 20)
    check_cxx_compiler_flag("-std=c++${candidate}" LO_HAS_CXX_${candidate})
    if(LO_HAS_CXX_${candidate})
        set(LO_STD ${candidate})
        break()
    endif()
endforeach()
message(STATUS "latent-oracle: compiling with -std=c++${LO_STD}")

set(CMAKE_CXX_STANDARD ${LO_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# ----------------------------------------------------------------------------
# Fathom (Syzygy probing) - vendored, MIT (third_party/fathom, see PROVENANCE)
# ----------------------------------------------------------------------------
add_library(fathom STATIC third_party/fathom/tbprobe.c)
target_include_directories(fathom PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fathom)
target_compile_options(fathom PRIVATE -w)

# ----------------------------------------------------------------------------
# Sanitizers. Must precede all target definitions: add_compile_options only
# affects targets created afterwards, and a partially instrumented build
# (runtime linked, compile flags missing) silently disables atomic tracking
# and produces false-positive races from intercepted calls alone.
# ----------------------------------------------------------------------------
if(LO_ENABLE_ASAN)
    add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
    add_link_options(-fsanitize=address)
endif()
if(LO_ENABLE_UBSAN)
    add_compile_options(-fno-omit-frame-pointer -fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()
if(LO_ENABLE_TSAN)
    add_compile_options(-fno-omit-frame-pointer -fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

# ----------------------------------------------------------------------------
# Core library
# ----------------------------------------------------------------------------
add_library(lo_core STATIC
    src/movegen/attack.cpp
    src/movegen/movegen.cpp
    src/position.cpp
    src/nn/net.cpp
    src/nn/netq.cpp
    src/tb/tb.cpp
    src/engine_oracle/engine.cpp
    src/uci/uci.cpp)
target_link_libraries(lo_core PUBLIC fathom)

target_include_directories(lo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lo_core PUBLIC Threads::Threads)

target_compile_options(lo_core PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<BOOL:${LO_WARNINGS_AS_ERRORS}>:-Werror>)

# ----------------------------------------------------------------------------
# Engine binary
# ----------------------------------------------------------------------------
add_executable(latent-oracle src/main.cpp)
target_link_libraries(latent-oracle PRIVATE lo_core)
target_compile_options(latent-oracle PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<BOOL:${LO_WARNINGS_AS_ERRORS}>:-Werror>)

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
include(CTest)
if(LO_BUILD_TESTS AND BUILD_TESTING)
    add_test(NAME perft_quick COMMAND latent-oracle perftsuite --quick)
    add_test(NAME perft_smoke COMMAND latent-oracle perft 4
             --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    set_tests_properties(perft_smoke PROPERTIES PASS_REGULAR_EXPRESSION "nodes: 4085603")
endif()
