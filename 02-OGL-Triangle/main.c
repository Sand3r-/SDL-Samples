#include "glad/gl.h"
#include <SDL.h>
#include <SDL_opengl.h>

char* File2Str(const char* filename) {
    SDL_RWops *file_rwops = SDL_RWFromFile(filename, "rb");
    if (file_rwops == NULL) return NULL;

    Sint64 file_size = SDL_RWsize(file_rwops);
    char* file_contents = (char*)SDL_malloc(file_size + 1);

    Sint64 total_objects_read = 0, objects_read = 1;
    char* buffer = file_contents;
    while (total_objects_read < file_size && objects_read != 0) {
        objects_read = SDL_RWread(file_rwops, buffer, 1, (file_size - total_objects_read));
        total_objects_read += objects_read;
        buffer += objects_read;
    }
    SDL_RWclose(file_rwops);
    if (total_objects_read != file_size) {
        SDL_free(file_contents);
        return NULL;
    }

    file_contents[total_objects_read] = '\0';
    return file_contents;
}

int main()
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("OpenGL - Hello Trinagle", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 640, 480, 
        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

    // Check that the window was successfully created
    if (window == NULL) {
        // In the case that the window could not be made...
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Could not create window: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetSwapInterval(0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_GLContext context = SDL_GL_CreateContext(window);
    gladLoadGL((GLADloadfunc) SDL_GL_GetProcAddress);

    // Create vao
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create & pload vertex data
    GLuint vbo;
    glGenBuffers(1, &vbo);

    GLfloat vertices[] = {0.0f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f};

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);


    // Load and compile vertex and fragment shaders
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    char* shader_source = File2Str("shader.vert");
    glShaderSource(vertex_shader, 1, &shader_source, NULL);
    glCompileShader(vertex_shader);
    SDL_free(shader_source);

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    shader_source = File2Str("shader.frag");
    glShaderSource(fragment_shader, 1, &shader_source, NULL);
    glCompileShader(fragment_shader);
    SDL_free(shader_source);

    // Link the shader program
    GLuint shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);
    glUseProgram(shader_program);

    GLint pos_attrib = glGetAttribLocation(shader_program, "position");
    glEnableVertexAttribArray(pos_attrib);
    glVertexAttribPointer(pos_attrib, 2, GL_FLOAT, GL_FALSE, 0, 0);

    while (1)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e) > 0)
        {
            if (e.type == SDL_QUIT)
                goto quit;
        }

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawArrays(GL_TRIANGLES, 0, 3);

        SDL_GL_SwapWindow(window);
    }

    quit:
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}