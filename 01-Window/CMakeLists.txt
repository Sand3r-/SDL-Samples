cmake_minimum_required (VERSION 3.21)

# Create target
set(TARGET 01-Window)
add_executable(${TARGET})

# Specify the required C standard
set_target_properties(${TARGET} PROPERTIES C_STANDARD 17)
set_target_properties(${TARGET} PROPERTIES CMAKE_C_STANDARD_REQUIRED True)

# Specify sources for the target
target_sources(${TARGET} PRIVATE main.c)

# Specify libraries for the target
list(APPEND LIBRARIES SDL2-static)
list(APPEND LIBRARIES SDL2main)
target_link_libraries(${TARGET} ${LIBRARIES})