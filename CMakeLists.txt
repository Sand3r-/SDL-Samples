cmake_minimum_required (VERSION 3.21)

# Define an executable target
project(SDL-Samples)

# Add cmake/ includes
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

# Fix problems with relative paths
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set-up SDL build
# Define static build-only
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)

# Remove unnecessary dependencies
set(DISABLED_SDL_SUBSYSTEMS SDL_ATOMIC SDL_HAPTIC SDL_POWER SDL_THREADS 
    SDL_TIMERS SDL_TIMERS SDL_CPUINFO SDL_SENSOR SDL_LOCALE SDL_MISC)
foreach(SDL_SUBSYSTEM ${DISABLED_SDL_SUBSYSTEMS})
    set(${SDL_SUBSYSTEM} OFF CACHE BOOL "" FORCE)
endforeach(SDL_SUBSYSTEM DISABLED_SDL_SUBSYSTEMS)

add_subdirectory(external/SDL)

# Add samples
add_subdirectory(01-Window)
add_subdirectory(02-OGL-Triangle)
add_subdirectory(03-D3D12-Triangle)
# add_subdirectory(04-VulkanTriangle)
# add_subdirectory(05-2D-Drawing)
# add_subdirectory(06-ControllerInput)
# add_subdirectory(07-Networking)
# add_subdirectory(08-Audio)