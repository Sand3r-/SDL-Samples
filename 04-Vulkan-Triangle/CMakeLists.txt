cmake_minimum_required (VERSION 3.21)

# Create target
set(TARGET 04-Vulkan-Triangle)
add_executable(${TARGET})

# Find Vulkan
find_package(Vulkan REQUIRED FATAL_ERROR)

# Set debugger working directory
set_property(TARGET ${TARGET} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Specify the required C standard
set_target_properties(${TARGET} PROPERTIES C_STANDARD 17)
set_target_properties(${TARGET} PROPERTIES CMAKE_C_STANDARD_REQUIRED True)

# Specify sources for the target
target_sources(${TARGET} PRIVATE main.c)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/shaders)
configure_file(shaders/shader.vert.spv ${CMAKE_BINARY_DIR}/bin/shaders COPYONLY)
configure_file(shaders/shader.frag.spv ${CMAKE_BINARY_DIR}/bin/shaders COPYONLY)

# Specify libraries for the target
list(APPEND LIBRARIES Vulkan::Vulkan)
list(APPEND LIBRARIES SDL2-static)
list(APPEND LIBRARIES SDL2main)
target_link_libraries(${TARGET} ${LIBRARIES})
