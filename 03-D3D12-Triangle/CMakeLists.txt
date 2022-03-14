cmake_minimum_required (VERSION 3.21)

# Create target
set(TARGET 03-D3D12-Triangle)
add_executable(${TARGET})

# Specify the required C standard
set_target_properties(${TARGET} PROPERTIES C_STANDARD 17)
set_target_properties(${TARGET} PROPERTIES CMAKE_C_STANDARD_REQUIRED True)

# Set debugger working directory
set_property(TARGET ${TARGET} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Specify sources for the target
target_sources(${TARGET} PRIVATE main.c)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/shaders)
configure_file(shaders/vertex.hlsl ${CMAKE_BINARY_DIR}/bin/shaders COPYONLY)
configure_file(shaders/pixel.hlsl ${CMAKE_BINARY_DIR}/bin/shaders COPYONLY)

# Specify libraries for the target
list(APPEND LIBRARIES d3d12.lib)
list(APPEND LIBRARIES dxgi.lib)
list(APPEND LIBRARIES dxguid.lib)
list(APPEND LIBRARIES d3dcompiler.lib)
list(APPEND LIBRARIES SDL2-static)
list(APPEND LIBRARIES SDL2main)
target_link_libraries(${TARGET} ${LIBRARIES})