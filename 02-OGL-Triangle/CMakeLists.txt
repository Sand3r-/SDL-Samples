cmake_minimum_required (VERSION 3.21)

# Create target
set(TARGET 02-OGL-Triangle)
add_executable(${TARGET})

# Set debugger working directory
set_property(TARGET ${TARGET} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Specify the required C standard
set_target_properties(${TARGET} PROPERTIES C_STANDARD 17)
set_target_properties(${TARGET} PROPERTIES CMAKE_C_STANDARD_REQUIRED True)

# Specify sources for the target
target_sources(${TARGET} PRIVATE main.c gl.c)
configure_file(shader.vert ${CMAKE_BINARY_DIR}/bin COPYONLY)
configure_file(shader.frag ${CMAKE_BINARY_DIR}/bin COPYONLY)

# Specify libraries for the target
list(APPEND LIBRARIES SDL2-static)
list(APPEND LIBRARIES SDL2main)
target_link_libraries(${TARGET} ${LIBRARIES})
