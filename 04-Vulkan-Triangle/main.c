#include <SDL.h>
#include <SDL_opengl.h>
#include <SDL_vulkan.h>
#include <vulkan/vulkan.h>

#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(x, y) (((x) > (y)) ? (x) : (y))

#define _countof(array) (sizeof(array) / sizeof(array[0]))
#define LOG_ERROR(...) SDL_LogError(SDL_LOG_CATEGORY_ERROR, __VA_ARGS__);
#define LOG_INFO(...) SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, __VA_ARGS__);

static int MAX_FRAMES_IN_FLIGHT = 2;
static uint32_t WIDTH;
static uint32_t HEIGHT;

struct Context
{
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    VkImage* swapChainImages;
    uint32_t swapChainImagesNum;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkImageView* swapChainImageViews;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkFramebuffer* swapChainFramebuffers;
    VkCommandPool commandPool;
    VkCommandBuffer* commandBuffers;
    VkSemaphore* imageAvailableSemaphores;
    VkSemaphore* renderFinishedSemaphores;
    VkFence* inFlightFences;
    VkFence* imagesInFlight;
    size_t currentFrame;

    const char* validationLayers[1];
    const char* deviceExtensions[1];

    VkBool32 enableValidationLayers;

    SDL_Window* window;
} VkContext;

static const int INVALID_QUEUE_FAMILY_INDEX = -1;

#define QUEUES_NUM 2
typedef struct QueueFamilyIndices
{
    union
    {
        struct
        {
            int graphicsFamily;
            int presentFamily;
        };
        int indices[QUEUES_NUM];
    };
} QueueFamilyIndices;

typedef struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilites;
    VkSurfaceFormatKHR* formats;
    int formatsNum;
    VkPresentModeKHR* presentModes;
    int presentModesNum;
} SwapChainSupportDetails;

typedef struct Buffer
{
    void* ptr;
    size_t length;
} Buffer;

Buffer AllocBuffer(size_t size)
{
    Buffer buffer = {
        .ptr = SDL_malloc(size),
        .length = size
    };
    return buffer;
}

Buffer FreeBuffer(Buffer buffer)
{
    SDL_free(buffer.ptr);
    buffer.ptr = NULL;
    buffer.length = 0;
    return buffer;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != NULL)
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    else
        return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void InitVkContext()
{
    VkContext.physicalDevice = VK_NULL_HANDLE;
    VkContext.currentFrame = 0;
    VkContext.validationLayers[0] = "VK_LAYER_KHRONOS_validation";
    VkContext.deviceExtensions[0] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;

    #ifdef NDEBUG
    VkContext.enableValidationLayers = 0;
    #else
    VkContext.enableValidationLayers = 1;
    #endif
}

void InitQueueFamilyIndices(QueueFamilyIndices* indices)
{
    indices->graphicsFamily = INVALID_QUEUE_FAMILY_INDEX;
    indices->presentFamily = INVALID_QUEUE_FAMILY_INDEX;
}

VkBool32 AreQueueFamilyIndicesComplete(QueueFamilyIndices* indices)
{
    return indices->graphicsFamily != INVALID_QUEUE_FAMILY_INDEX &&
           indices->presentFamily != INVALID_QUEUE_FAMILY_INDEX;
}

void FreeSwapChainSupportDetails(SwapChainSupportDetails* details)
{
    if (details->formats != NULL) SDL_free(details->formats);
    if (details->presentModes != NULL) SDL_free(details->presentModes);
}

VkBool32 CheckValidationLayerSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, NULL);
    VkLayerProperties* availableLayers = SDL_malloc(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);

    for (int i = 0; i < _countof(VkContext.validationLayers); i++)
    {
        VkBool32 layerFound = VK_FALSE;
        const char* layerName = VkContext.validationLayers[i];

        for (int j = 0; j < layerCount; j++)
        {
            const VkLayerProperties* layerProperties = &availableLayers[j];
            if (strcmp(layerName, layerProperties->layerName) == 0)
            {
                layerFound = VK_TRUE;
                break;
            }
        }

        if (!layerFound)
        {
            SDL_free(availableLayers);
            return VK_FALSE;
        }
    }
    SDL_free(availableLayers);

    return VK_TRUE;
}

char** GetRequiredExtensions(uint32_t* extensionCount)
{
    SDL_Vulkan_GetInstanceExtensions(VkContext.window, extensionCount, NULL);
    size_t allocSize =
        (*extensionCount + VkContext.enableValidationLayers) * sizeof(char*);
    char** extensions = SDL_malloc(allocSize);
    // Please note that the call to SDL_Vulkan_GetInstanceExtensions
    // resets the extensionCount to it's original value.
    SDL_Vulkan_GetInstanceExtensions(VkContext.window, extensionCount, extensions);

    if (VkContext.enableValidationLayers)
        extensions[(*extensionCount)++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

    return extensions;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {

        SDL_Log("Vulkan: %s", pCallbackData->pMessage);

        return VK_FALSE;
    }

void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT* createInfo)
{
    VkDebugUtilsMessengerCreateInfoEXT info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = DebugCallback,
        .pUserData = NULL
    };
    *createInfo = info;
}

void PrintExtensions()
{
    uint32_t num_extensions;
    vkEnumerateInstanceExtensionProperties(NULL, &num_extensions, NULL);
    VkExtensionProperties* extensions = SDL_malloc(num_extensions * sizeof(VkExtensionProperties));
    vkEnumerateInstanceExtensionProperties(NULL, &num_extensions, extensions);


    SDL_Log("Available extensions:");
    for (int i = 0; i < num_extensions; i++)
    {
        SDL_Log("%s in version of %u", extensions[i].extensionName, extensions[i].specVersion);
    }

    SDL_free(extensions);
}

void CreateInstance()
{
    if (VkContext.enableValidationLayers && !CheckValidationLayerSupport())
        LOG_ERROR("The validation layers couldn't have been enabled");

    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "-",
        .engineVersion = VK_MAKE_VERSION(0, 0, 1),
        .apiVersion = VK_API_VERSION_1_2,
    };

    VkInstanceCreateInfo createInfo = {
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        createInfo.pApplicationInfo = &appInfo,
    };

    uint32_t extensionsCount;
    char** extensions = GetRequiredExtensions(&extensionsCount);
    createInfo.enabledExtensionCount = extensionsCount;
    createInfo.ppEnabledExtensionNames = extensions;
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    if (VkContext.enableValidationLayers)
    {
        createInfo.enabledLayerCount = _countof(VkContext.validationLayers);
        createInfo.ppEnabledLayerNames = VkContext.validationLayers;

        PopulateDebugMessengerCreateInfo(&debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else
    {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateInstance(&createInfo, NULL, &VkContext.instance) != VK_SUCCESS)
        LOG_ERROR("Vulkan Instance creation failure");

    SDL_free(extensions);

    PrintExtensions();
}

void SetupDebugMessenger()
{
    if (!VkContext.enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo = {0};
    PopulateDebugMessengerCreateInfo(&createInfo);

    if (CreateDebugUtilsMessengerEXT(
        VkContext.instance, &createInfo, NULL, &VkContext.debugMessenger) != VK_SUCCESS)
        LOG_ERROR("The debug messenger couldn't have been created.");
}

void CreateSurface()
{
    if (SDL_Vulkan_CreateSurface(VkContext.window, VkContext.instance, &VkContext.surface)
        != SDL_TRUE)
        LOG_ERROR("Vulkan surface creation error.");
}

QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    InitQueueFamilyIndices(&indices);

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, NULL);

    VkQueueFamilyProperties* queueFamilies = SDL_malloc(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies);

    
    for (int i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, VkContext.surface, &presentSupport);

        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (AreQueueFamilyIndicesComplete(&indices)) {
            break;
        }
    }
    
    SDL_free(queueFamilies);

    return indices;
}

VkBool32 CheckDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, NULL);

    VkExtensionProperties* availableExtensions = SDL_malloc(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, availableExtensions);

    int matchedExtensionsNum = 0;

    for (int i = 0; i < extensionCount; i++)
    {
        for (int j = 0; j < _countof(VkContext.deviceExtensions); j++)
        {
            if (!SDL_strcmp(availableExtensions[i].extensionName, VkContext.deviceExtensions[j]))
                matchedExtensionsNum++;
        }
    }

    SDL_free(availableExtensions);

    return matchedExtensionsNum == _countof(VkContext.deviceExtensions);
}

// It's callers responsible to free memory with FreeSwapChainSupportDetails
SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, VkContext.surface, &details.capabilites);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, VkContext.surface, &formatCount, NULL);
        if (formatCount > 0) {
            details.formats = SDL_malloc(formatCount * sizeof(VkSurfaceFormatKHR));
            details.formatsNum = formatCount;
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, VkContext.surface, &formatCount, details.formats);
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, VkContext.surface, &presentModeCount, NULL);
        if (presentModeCount > 0) {
            details.presentModes = SDL_malloc(presentModeCount * sizeof(VkPresentModeKHR));
            details.presentModesNum = presentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, VkContext.surface, &presentModeCount, details.presentModes);
        }

        return details;
    }

VkBool32 IsDeviceSuitable(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    VkBool32 suitable = deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    VkBool32 tex_compression = deviceFeatures.textureCompressionETC2;

    QueueFamilyIndices indices = FindQueueFamilies(device);

    LOG_INFO("Selecting device %s", deviceProperties.deviceName);
    LOG_INFO("Having API version of %u", deviceProperties.apiVersion);
    LOG_INFO("And type of %d", deviceProperties.deviceType);

    VkBool32 extensionsSupported = CheckDeviceExtensionSupport(device);

    VkBool32 swapChainAdequate = VK_FALSE;
    if (extensionsSupported) {
        SwapChainSupportDetails details = QuerySwapChainSupport(device);
        swapChainAdequate = details.formatsNum && details.presentModesNum;
        FreeSwapChainSupportDetails(&details);
    }

    return AreQueueFamilyIndicesComplete(&indices) && extensionsSupported && swapChainAdequate;
}

void PickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(VkContext.instance, &deviceCount, NULL);

    if (deviceCount == 0)
        LOG_ERROR("No GPUs found on the machine.");

    VkPhysicalDevice* devices = SDL_malloc(deviceCount);
    vkEnumeratePhysicalDevices(VkContext.instance, &deviceCount, devices);


    for (int i = 0; i < deviceCount; i++)
    {
        if (IsDeviceSuitable(devices[i]))
        {
            VkContext.physicalDevice = devices[i];
            break;
        }
    }

    if (VkContext.physicalDevice == VK_NULL_HANDLE)
        LOG_ERROR("No suitable GPU found.");

    SDL_free(devices);
}

void CreateLogicalDevice()
{
    QueueFamilyIndices indices = FindQueueFamilies(VkContext.physicalDevice);

    VkDeviceQueueCreateInfo queueCreateInfos[QUEUES_NUM];

    int uniqueQueuesNum = indices.graphicsFamily != indices.presentFamily ?
                          QUEUES_NUM : 1;
    float priority = 1.0f;
    for (int i = 0; i < uniqueQueuesNum; i++)
    {
        VkDeviceQueueCreateInfo queueCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = indices.indices[i],
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
        queueCreateInfos[i] = queueCreateInfo;
    }

    VkPhysicalDeviceFeatures deviceFeatures = {0};

    VkDeviceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = (uint32_t)uniqueQueuesNum,
        .pQueueCreateInfos = queueCreateInfos,
        .pEnabledFeatures = &deviceFeatures,
        .enabledExtensionCount = _countof(VkContext.deviceExtensions),
        .ppEnabledExtensionNames = VkContext.deviceExtensions
    };

    if (VkContext.enableValidationLayers)
    {
        createInfo.enabledLayerCount = _countof(VkContext.validationLayers);
        createInfo.ppEnabledLayerNames = VkContext.validationLayers;
    }
    else
    {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(VkContext.physicalDevice, &createInfo, NULL, &VkContext.device) != VK_SUCCESS)
        LOG_ERROR("Logical device creation error.");

    vkGetDeviceQueue(VkContext.device, indices.graphicsFamily, 0, &VkContext.graphicsQueue);
    vkGetDeviceQueue(VkContext.device, indices.presentFamily, 0, &VkContext.presentQueue);
}

VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const VkSurfaceFormatKHR* availableFormats, uint32_t formatsNum)
{
    for (int i = 0; i < formatsNum; i++)
    {
        if (availableFormats[i].format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return availableFormats[i];
    }

    return availableFormats[0]; // Fallback to any surface format
}

VkPresentModeKHR ChooseSwapPresentMode(const VkPresentModeKHR* modes, uint32_t presentModesNum)
{
    for (int i = 0; i < presentModesNum; i++)
    {
        if (modes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
            return modes[i];
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR* capabilities)
{
    if (capabilities->currentExtent.width != UINT32_MAX)
    {
        return capabilities->currentExtent;
    }
    else
    {
        VkExtent2D actualExtent;
        SDL_Vulkan_GetDrawableSize(VkContext.window, &actualExtent.width, &actualExtent.height);
        VkExtent2D minExtent = capabilities->minImageExtent;
        VkExtent2D maxExtent = capabilities->maxImageExtent;
        actualExtent.width = max(minExtent.width, min(maxExtent.width, actualExtent.width));
        actualExtent.height = max(minExtent.height, min(maxExtent.height, actualExtent.width));

        return actualExtent;
    }
}

void CreateSwapChain()
{
    SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(VkContext.physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats, swapChainSupport.formatsNum);
    VkPresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.presentModes, swapChainSupport.presentModesNum);
    VkExtent2D extent = ChooseSwapExtent(&swapChainSupport.capabilites);

    uint32_t imageCount = swapChainSupport.capabilites.minImageCount + 1;

    if (swapChainSupport.capabilites.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilites.maxImageCount)
        imageCount = swapChainSupport.capabilites.maxImageCount;

    VkSwapchainCreateInfoKHR createInfo = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = VkContext.surface,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    };

    QueueFamilyIndices indices = FindQueueFamilies(VkContext.physicalDevice);

    if(indices.graphicsFamily != indices.presentFamily)
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = indices.indices;
    }
    else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = NULL;
    }

    createInfo.preTransform = swapChainSupport.capabilites.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(VkContext.device, &createInfo, NULL, &VkContext.swapChain) != VK_SUCCESS)
    {
        LOG_ERROR("SwapChain creation failure.");
    }

    LOG_INFO("Swapchain created.");

    vkGetSwapchainImagesKHR(VkContext.device, VkContext.swapChain, &imageCount, NULL);
    VkContext.swapChainImages = SDL_malloc(imageCount * sizeof(VkImage));
    VkContext.swapChainImagesNum = imageCount;
    vkGetSwapchainImagesKHR(VkContext.device, VkContext.swapChain, &imageCount, VkContext.swapChainImages);

    VkContext.swapChainImageFormat = surfaceFormat.format;
    VkContext.swapChainExtent = extent;
}

void CreateImageViews()
{
    VkContext.swapChainImageViews = SDL_malloc(VkContext.swapChainImagesNum * sizeof(VkImageView));
    VkImageViewCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .format = VkContext.swapChainImageFormat,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .subresourceRange.baseMipLevel = 0,
        .subresourceRange.levelCount = 1,
        .subresourceRange.baseArrayLayer = 0,
        .subresourceRange.layerCount = 1
    };

    for (size_t i = 0; i < VkContext.swapChainImagesNum; i++) {
        createInfo.image = VkContext.swapChainImages[i];
        if (vkCreateImageView(VkContext.device, &createInfo, NULL, &VkContext.swapChainImageViews[i])
            != VK_SUCCESS) {
                LOG_ERROR("Image view creation failure");
            }
    }

    LOG_INFO("SwapChain image views have been created.");
}

void CreateRenderPass()
{
    VkAttachmentDescription colorAttachment = {
        .format = VkContext.swapChainImageFormat,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    VkAttachmentReference colorAttachmentRef = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef,
    };

    VkSubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    VkRenderPassCreateInfo renderPassInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    if (vkCreateRenderPass(VkContext.device, &renderPassInfo, NULL, &VkContext.renderPass)
        != VK_SUCCESS)
        LOG_ERROR("Render pass creation failure.");

    LOG_INFO("Render pass was created.");
}

static Buffer File2Buffer(const char* filename)
{
    Buffer result = {0};
    SDL_RWops *fileRWops = SDL_RWFromFile(filename, "rb");
    if (fileRWops == NULL)
    {
        LOG_ERROR("File %s not found.", filename);
        return result;
    }

    Sint64 fileSize = SDL_RWsize(fileRWops);
    result = AllocBuffer(fileSize);
    char* fileContents = (char*)result.ptr;

    Sint64 totalObjectsRead = 0, objectsRead = 1;
    char* buffer = fileContents;
    while (totalObjectsRead < fileSize && objectsRead != 0)
    {
        objectsRead = SDL_RWread(fileRWops, buffer, 1, (fileSize - totalObjectsRead));
        totalObjectsRead += objectsRead;
        buffer += objectsRead;
    }
    SDL_RWclose(fileRWops);
    if (totalObjectsRead != fileSize)
    {
        result = FreeBuffer(result);
        return result;
    }

    return result;
}

VkShaderModule CreateShaderModule(Buffer code)
{
    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.length,
        .pCode = (uint32_t*)code.ptr
    };

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(VkContext.device, &createInfo, NULL, &shaderModule)
        != VK_SUCCESS)
            LOG_ERROR("Shader module creation failure.");

    return shaderModule;
}

void CreateGraphicsPipeline()
{
    Buffer vertShaderCode = File2Buffer("shaders/shader.vert.spv");
    Buffer fragShaderCode = File2Buffer("shaders/shader.frag.spv");

    VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

    FreeBuffer(vertShaderCode);
    FreeBuffer(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertexShaderStageInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertShaderModule,
        .pName = "main",
    };

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragShaderModule,
        .pName = "main",
    };

    // Note: pSpecializationInfo can be used to set constants in shaders
    // which can be used to optimize out if statements and unroll for loops
    // pretty neat for optimization

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        vertexShaderStageInfo, fragShaderStageInfo
    };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .pVertexAttributeDescriptions = NULL,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = NULL,
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = (float) VkContext.swapChainExtent.width,
        .height = (float) VkContext.swapChainExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    VkRect2D scissor = {
        .offset  = {0, 0},
        .extent = VkContext.swapChainExtent,
    };

    VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .lineWidth = 1.0f,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
    };

    // Requires enabling GPU Feature.
    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = VK_FALSE,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .minSampleShading = 1.0f,
        .pSampleMask = NULL,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    VkPipelineColorBlendAttachmentState colorBlendAttachement = {
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
    };

    VkPipelineColorBlendStateCreateInfo colorBlending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachement,
        .blendConstants[0] = 0.0f,
        .blendConstants[1] = 0.0f,
        .blendConstants[2] = 0.0f,
        .blendConstants[3] = 0.0f,
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0,
        .pSetLayouts = NULL,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = NULL,
    };

    if (vkCreatePipelineLayout(VkContext.device, &pipelineLayoutInfo, NULL,
        &VkContext.pipelineLayout) != VK_SUCCESS)
        LOG_ERROR("PipelineLayout creation failure.");

    VkGraphicsPipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = NULL,
        .pColorBlendState = &colorBlending,
        .pDynamicState = NULL,
        .layout = VkContext.pipelineLayout,
        .renderPass = VkContext.renderPass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    if (vkCreateGraphicsPipelines(VkContext.device, VK_NULL_HANDLE, 1, &pipelineInfo,
        NULL, &VkContext.graphicsPipeline) != VK_SUCCESS)
        LOG_ERROR("Graphics Pipelines creation failure");

    LOG_INFO("Graphics Pipeline has been created.");

    vkDestroyShaderModule(VkContext.device, fragShaderModule, NULL);
    vkDestroyShaderModule(VkContext.device, vertShaderModule, NULL);
}

void CreateFramebuffers()
{
    VkContext.swapChainFramebuffers = 
        SDL_malloc(VkContext.swapChainImagesNum * sizeof(VkFramebuffer));

    for (size_t i = 0; i < VkContext.swapChainImagesNum; i++)
    {
        VkImageView attachments[] = {
            VkContext.swapChainImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = VkContext.renderPass,
            .attachmentCount = 1,
            .pAttachments = attachments,
            .width = VkContext.swapChainExtent.width,
            .height = VkContext.swapChainExtent.height,
            .layers = 1,
        };

        if (vkCreateFramebuffer(VkContext.device, &framebufferInfo, NULL,
            &VkContext.swapChainFramebuffers[i]) != VK_SUCCESS)
            LOG_ERROR("Framebuffer creation failure");
    }

    LOG_INFO("Framebuffers created");
}

void CreateCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = FindQueueFamilies(VkContext.physicalDevice);

    VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily,
        .flags = 0,
    };

    if (vkCreateCommandPool(VkContext.device, &poolInfo, NULL, &VkContext.commandPool)
        != VK_SUCCESS)
        LOG_ERROR("Command pool creation failure");
}

void CreateCommandBuffers()
{
    VkContext.commandBuffers = SDL_malloc(VkContext.swapChainImagesNum);

    VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = VkContext.commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = (uint32_t)VkContext.swapChainImagesNum,
    };

    if (vkAllocateCommandBuffers(VkContext.device, &allocInfo, VkContext.commandBuffers)
        != VK_SUCCESS)
        LOG_ERROR("Command buffer allocation failure");

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
        .pInheritanceInfo = NULL,
    };

    VkRenderPassBeginInfo renderPassInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = VkContext.renderPass,
        .renderArea.offset = {0, 0},
        .renderArea.extent = VkContext.swapChainExtent,
    };

    VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    for (size_t i = 0; i < VkContext.swapChainImagesNum; i++) {
        VkCommandBuffer cmdBuffer = VkContext.commandBuffers[i];
        if (vkBeginCommandBuffer(cmdBuffer, &beginInfo)
            != VK_SUCCESS)
            LOG_ERROR("BeginCommandBuffer failure");

        renderPassInfo.framebuffer = VkContext.swapChainFramebuffers[i];

        vkCmdBeginRenderPass(
            cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(
            cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, VkContext.graphicsPipeline);

        vkCmdDraw(cmdBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(cmdBuffer);

        if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS)
            LOG_ERROR("EndCommandBuffer failure");
    }

    LOG_INFO("CommandBuffers recorded");
}

void CreateSyncObjects()
{
    VkContext.imageAvailableSemaphores = SDL_malloc(sizeof(VkSemaphore) * MAX_FRAMES_IN_FLIGHT);
    VkContext.renderFinishedSemaphores = SDL_malloc(sizeof(VkSemaphore) * MAX_FRAMES_IN_FLIGHT);
    VkContext.inFlightFences = SDL_malloc(sizeof(VkFence) * MAX_FRAMES_IN_FLIGHT);
    VkContext.imagesInFlight = SDL_malloc(sizeof(VkFence) * VkContext.swapChainImagesNum);
    SDL_memset(VkContext.imagesInFlight, 0, sizeof(VkFence) * VkContext.swapChainImagesNum);

    VkSemaphoreCreateInfo semaphoreInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
    };

    VkFenceCreateInfo fenceInfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(VkContext.device, &semaphoreInfo, NULL, &VkContext.imageAvailableSemaphores[i])
            != VK_SUCCESS ||
            vkCreateSemaphore(VkContext.device, &semaphoreInfo, NULL, &VkContext.renderFinishedSemaphores[i])
            != VK_SUCCESS ||
            vkCreateFence(VkContext.device, &fenceInfo, NULL, &VkContext.inFlightFences[i])
            != VK_SUCCESS)
            LOG_ERROR("Semaphore/Fence creation failure");
    }

    LOG_INFO("Synchronisation objects created.")
}

void InitVk()
{
    InitVkContext();
    CreateInstance();
    SetupDebugMessenger();
    CreateSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapChain();
    CreateImageViews();
    CreateRenderPass();
    CreateGraphicsPipeline();
    CreateFramebuffers();
    CreateCommandPool();
    CreateCommandBuffers();
    CreateSyncObjects();
}

void DrawFrame()
{
    vkWaitForFences(VkContext.device, 1, &VkContext.inFlightFences[VkContext.currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    vkAcquireNextImageKHR(VkContext.device, VkContext.swapChain, UINT64_MAX,
        VkContext.imageAvailableSemaphores[VkContext.currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (VkContext.imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        vkWaitForFences(VkContext.device, 1, &VkContext.imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO
    };

    VkSemaphore waitSemaphores[] = { 
        VkContext.imageAvailableSemaphores[VkContext.currentFrame]
    };
    VkPipelineStageFlags waitStages[] =
        { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &VkContext.commandBuffers[imageIndex];

    VkSemaphore signalSemaphores[] = { VkContext.renderFinishedSemaphores[VkContext.currentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkResetFences(VkContext.device, 1, &VkContext.inFlightFences[VkContext.currentFrame]);
    if (vkQueueSubmit(VkContext.graphicsQueue, 1, &submitInfo, VkContext.inFlightFences[VkContext.currentFrame])
        != VK_SUCCESS)
        LOG_ERROR("Queue submit failure");

    VkSwapchainKHR swapChains[] = {VkContext.swapChain};
    VkPresentInfoKHR presentInfo = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = signalSemaphores,
        .swapchainCount = 1,
        .pSwapchains = swapChains,
        .pImageIndices = &imageIndex,
        .pResults = NULL,
    };

    vkQueuePresentKHR(VkContext.presentQueue, &presentInfo);

    VkContext.currentFrame = (VkContext.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void DestroyVk()
{
    SDL_free(VkContext.swapChainImages);
    SDL_free(VkContext.swapChainImageViews);
    SDL_free(VkContext.swapChainFramebuffers);
    SDL_free(VkContext.commandBuffers);
}

int main()
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_LogSetPriority(SDL_LOG_CATEGORY_ERROR, SDL_LOG_PRIORITY_ERROR);

    VkContext.window = SDL_CreateWindow("Vulkan - Hello Triangle", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 640, 480, 
        SDL_WINDOW_VULKAN  | SDL_WINDOW_SHOWN);

    // Check that the window was successfully created
    if (VkContext.window == NULL) {
        // In the case that the window could not be made...
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Could not create window: %s\n", SDL_GetError());
        return 1;
    }

    InitVk();

    while (1)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e) > 0)
        {
            if (e.type == SDL_QUIT)
                goto quit;
        }
        DrawFrame();
    }

    quit:
    SDL_DestroyWindow(VkContext.window);
    SDL_Quit();
    DestroyVk();

    return 0;
}