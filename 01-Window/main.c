#include <SDL.h>

int main()
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window= SDL_CreateWindow("Hello SDL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 640, 480, 0);

    // Check that the window was successfully created
    if (window == NULL) {
        // In the case that the window could not be made...
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Could not create window: %s\n", SDL_GetError());
        return 1;
    }

    while (1)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e) > 0)
        {
            if (e.type == SDL_QUIT)
                goto quit;
        }
    }

    quit:
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}