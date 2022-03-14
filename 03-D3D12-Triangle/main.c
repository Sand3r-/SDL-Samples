#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <assert.h>
#include <time.h>
#include <signal.h>
#include <SDL.h>
#include <SDL_syswm.h>

#define COBJMACROS
    #pragma warning(push)
    #pragma warning(disable:4115) // named type definition in parentheses
    #include <d3d12.h>
    #include <d3dcompiler.h>
    #include <dxgi1_6.h>
    #include <dxgidebug.h>
    #include <d3d12sdklayers.h>
    #pragma warning(pop)
#undef COBJMACROS

#define HD_EXIT_FAILURE -1
#define HD_EXIT_SUCCESS 0

// Number of render targets
#define FRAMES_NUM 3

#define ID3DBlob_GetBufferPointer(self) ID3D10Blob_GetBufferPointer(self)
#define ID3DBlob_Release(self) ID3D10Blob_Release(self)
#define ID3DBlob_GetBufferSize(self) ID3D10Blob_GetBufferSize(self)

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define ExitOnFailure(expression) if (FAILED(expression)) raise(SIGINT);
#define S(x) #x
#define S_(x) S(x)
#define S__LINE__ S_(__LINE__)
#define GEN_REPORT_STRING() "Reporting Live objects at " __FUNCTION__ ", L:" S__LINE__ "\n"
#define REPORT_LIVE_OBJ() ReportLiveObjects(GEN_REPORT_STRING())

typedef struct ResizeData
{
    D3D12_VIEWPORT* viewport;
    ID3D12Device2* device;
    ID3D12Resource** depthBuffer;
    float fov;
} ResizeData;

ID3D12Device2* g_Device;
IDXGISwapChain4* g_SwapChain;
ID3D12CommandQueue* g_CommandQueue;
ID3D12GraphicsCommandList* g_CommandList;
ID3D12PipelineState* g_PipelineState;
ID3D12RootSignature* g_RootSignature;
D3D12_VIEWPORT g_Viewport;
D3D12_RECT g_ScissorRect;
IDXGIDebug1* g_Debug;
ID3D12Resource* g_BackBuffers[FRAMES_NUM];
ID3D12Resource* g_DepthBuffer;
ID3D12CommandAllocator* g_CommandAllocators[FRAMES_NUM];
uint64_t g_FrameFenceValues[FRAMES_NUM];
uint64_t g_FenceValue = 0;
UINT g_CurrentBackBufferIndex;
UINT g_RTVDescriptorSize;
UINT g_DSVDescriptorSize;
ID3D12DescriptorHeap* g_RTVDescriptorHeap;
ID3D12DescriptorHeap* g_DSVDescriptorHeap;
ID3D12Fence* g_Fence;
HANDLE g_FenceEvent;
ResizeData g_ResizeData;

void EnableDebuggingLayer()
{
    ID3D12Debug* debugInterface;
    ExitOnFailure(D3D12GetDebugInterface(&IID_ID3D12Debug, &debugInterface));
    ID3D12Debug_EnableDebugLayer(debugInterface);
    ID3D12Debug_Release(debugInterface);

    ExitOnFailure(DXGIGetDebugInterface1(0, &IID_IDXGIDebug1, &g_Debug));
}

void ReportLiveObjects(const char* report)
{
    OutputDebugString("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    OutputDebugString(report);
    OutputDebugString("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    IDXGIDebug1_ReportLiveObjects(g_Debug, DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_ALL);
}

IDXGIAdapter4* GetAdapter()
{
    IDXGIFactory4* dxgiFactory;
    UINT createFactoryFlags = 0;
#if defined(_DEBUG)
    createFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

    ExitOnFailure(CreateDXGIFactory2(createFactoryFlags, &IID_IDXGIFactory4, &dxgiFactory));

    IDXGIAdapter1* dxgiAdapter1;
    IDXGIAdapter4* dxgiAdapter4;

    SIZE_T maxDedicatedVideoMemory = 0;
    for (UINT i = 0; IDXGIFactory1_EnumAdapters1(dxgiFactory, i, &dxgiAdapter1) != DXGI_ERROR_NOT_FOUND; ++i)
    {
        DXGI_ADAPTER_DESC1 dxgiAdapterDesc1;
        IDXGIAdapter1_GetDesc1(dxgiAdapter1, &dxgiAdapterDesc1);

        if ((dxgiAdapterDesc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
            SUCCEEDED(D3D12CreateDevice((IUnknown*)dxgiAdapter1,
                                        D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, NULL)) &&
            dxgiAdapterDesc1.DedicatedVideoMemory > maxDedicatedVideoMemory )
        {
            maxDedicatedVideoMemory = dxgiAdapterDesc1.DedicatedVideoMemory;

            ExitOnFailure(IDXGIAdapter1_QueryInterface(dxgiAdapter1, &IID_IDXGIAdapter4, &dxgiAdapter4));
        }
        IDXGIAdapter1_Release(dxgiAdapter1);
    }

    IDXGIFactory4_Release(dxgiFactory);

    return dxgiAdapter4;
}

ID3D12Device2* CreateDevice(IDXGIAdapter4* adapter)
{
    ID3D12Device2* d3d12Device2;
    ExitOnFailure(D3D12CreateDevice((IUnknown*)adapter, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device2, &d3d12Device2));

    // Enable debug messages in debug mode.
#if defined(_DEBUG)
    ID3D12InfoQueue* pInfoQueue;

    if (SUCCEEDED(ID3D12Device2_QueryInterface(d3d12Device2, &IID_ID3D12InfoQueue, &pInfoQueue)))
    {
        ID3D12InfoQueue_SetBreakOnSeverity(pInfoQueue, D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        ID3D12InfoQueue_SetBreakOnSeverity(pInfoQueue, D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
        ID3D12InfoQueue_SetBreakOnSeverity(pInfoQueue, D3D12_MESSAGE_SEVERITY_WARNING, FALSE);

        // Suppress messages based on their severity level
        D3D12_MESSAGE_SEVERITY Severities[] =
        {
            D3D12_MESSAGE_SEVERITY_INFO
        };

        // Suppress individual messages by their ID
        D3D12_MESSAGE_ID DenyIds[] = {
            D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
            D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,
            D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,
        };

        D3D12_INFO_QUEUE_FILTER NewFilter = {0};
        NewFilter.DenyList.NumSeverities = _countof(Severities);
        NewFilter.DenyList.pSeverityList = Severities;
        NewFilter.DenyList.NumIDs = _countof(DenyIds);
        NewFilter.DenyList.pIDList = DenyIds;

        ExitOnFailure(ID3D12InfoQueue_PushStorageFilter(pInfoQueue, &NewFilter));
    }
#endif

    ID3D12Object_SetName(d3d12Device2, L"Device2");
    return d3d12Device2;
}

ID3D12CommandQueue* CreateCommandQueue(ID3D12Device2* device, D3D12_COMMAND_LIST_TYPE type)
{
    ID3D12CommandQueue* d3d12CommandQueue;

    D3D12_COMMAND_QUEUE_DESC desc = {0};
    desc.Type =     type;
    desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    desc.Flags =    D3D12_COMMAND_QUEUE_FLAG_NONE;
    desc.NodeMask = 0;

    ExitOnFailure(ID3D12Device2_CreateCommandQueue(device, &desc, &IID_ID3D12CommandQueue, &d3d12CommandQueue));

    ID3D12Object_SetName(d3d12CommandQueue, L"CommandQueue");
    return d3d12CommandQueue;
}

IDXGISwapChain4* CreateSwapChain(HWND hWnd,
                                 ID3D12CommandQueue* commandQueue,
                                 uint32_t width, uint32_t height, uint32_t bufferCount )
{
    IDXGISwapChain4* dxgiSwapChain4;
    IDXGIFactory4* dxgiFactory4;
    UINT createFactoryFlags = 0;
#if defined(_DEBUG)
    createFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

    ExitOnFailure(CreateDXGIFactory2(createFactoryFlags, &IID_IDXGIFactory4, &dxgiFactory4));

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {
            .Width = width,
            .Height = height,
            .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
            .Stereo = FALSE,
            .SampleDesc = {
                .Count = 1,
                .Quality = 0,
            },
            .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
            .BufferCount = bufferCount,
            .Scaling = DXGI_SCALING_STRETCH,
            .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
            .AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED,
            .Flags = 0
        };


    IDXGISwapChain1* swapChain1;
    ExitOnFailure(IDXGIFactory4_CreateSwapChainForHwnd(dxgiFactory4,
                                                       (IUnknown*)commandQueue,
                                                       hWnd,
                                                       &swapChainDesc,
                                                       NULL,
                                                       NULL,
                                                       &swapChain1));

    ExitOnFailure(IDXGIFactory4_MakeWindowAssociation(dxgiFactory4, hWnd, DXGI_MWA_NO_ALT_ENTER));

    ExitOnFailure(IDXGISwapChain1_QueryInterface(swapChain1, &IID_IDXGISwapChain4, &dxgiSwapChain4));

    IDXGIFactory4_Release(dxgiFactory4);

    return dxgiSwapChain4;
}

ID3D12DescriptorHeap* CreateDescriptorHeap(ID3D12Device2* device,
                                           D3D12_DESCRIPTOR_HEAP_TYPE type,
                                           uint32_t numDescriptors)
{
    ID3D12DescriptorHeap* descriptorHeap;

    D3D12_DESCRIPTOR_HEAP_DESC desc = {0};
    desc.NumDescriptors = numDescriptors;
    desc.Type = type;

    ExitOnFailure(ID3D12Device2_CreateDescriptorHeap(device, &desc, &IID_ID3D12DescriptorHeap, &descriptorHeap));

    ID3D12Object_SetName(descriptorHeap, L"DescriptorHeap");
    return descriptorHeap;
}

void UpdateRenderTargetViews(ID3D12Device2* device,
                             IDXGISwapChain4* swapChain, ID3D12DescriptorHeap* descriptorHeap)
{
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(descriptorHeap, &rtvHandle);

    for (int i = 0; i < FRAMES_NUM; ++i)
    {
        ID3D12Resource* backBuffer;
        ExitOnFailure(IDXGISwapChain4_GetBuffer(swapChain, i, &IID_ID3D12Resource, &backBuffer));

        ID3D12Device2_CreateRenderTargetView(device, backBuffer, NULL, rtvHandle);

        g_BackBuffers[i] = backBuffer;

        rtvHandle.ptr += g_RTVDescriptorSize;
    }
}

ID3D12CommandAllocator* CreateCommandAllocator(ID3D12Device2* device,
                                               D3D12_COMMAND_LIST_TYPE type)
{
    ID3D12CommandAllocator* commandAllocator;
    ExitOnFailure(ID3D12Device2_CreateCommandAllocator(device, type,
                                                       &IID_ID3D12CommandAllocator,
                                                       &commandAllocator));
    ID3D12Object_SetName(commandAllocator, L"Command Allocator");
    return commandAllocator;
}

ID3D12GraphicsCommandList* CreateCommandList(ID3D12Device2* device,
                                             ID3D12CommandAllocator* commandAllocator,
                                             D3D12_COMMAND_LIST_TYPE type)
{
    ID3D12GraphicsCommandList* commandList;
    ExitOnFailure(ID3D12Device2_CreateCommandList(device, 0, type, commandAllocator, NULL, &IID_ID3D12CommandList, &commandList));

    ExitOnFailure(ID3D12GraphicsCommandList_Close(commandList));

    ID3D12Object_SetName(commandList, L"CommandList");
    return commandList;
}

ID3D12Fence* CreateFence(ID3D12Device2* device)
{
    // Reset fence values
    memset(g_FrameFenceValues, 0, FRAMES_NUM * sizeof(uint64_t));

    ID3D12Fence* fence;
    ExitOnFailure(ID3D12Device2_CreateFence(device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, &fence));

    ID3D12Object_SetName(fence, L"Fence");
    return fence;
}

D3D12_RESOURCE_BARRIER D3D12_RESOURCE_BARRIER_Transition(
        ID3D12Resource* pResource,
        D3D12_RESOURCE_STATES stateBefore,
        D3D12_RESOURCE_STATES stateAfter,
        UINT subresource, // = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        D3D12_RESOURCE_BARRIER_FLAGS flags) // = D3D12_RESOURCE_BARRIER_FLAG_NONE
{
    D3D12_RESOURCE_BARRIER barrier;
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = flags;
    barrier.Transition.pResource = pResource;
    barrier.Transition.StateBefore = stateBefore;
    barrier.Transition.StateAfter = stateAfter;
    barrier.Transition.Subresource = subresource;
    return barrier;
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12_CPU_DESCRIPTOR_HANDLE_Offset(
    D3D12_CPU_DESCRIPTOR_HANDLE handle,
    INT offsetInDescriptors,
    UINT descriptorIncrementSize)
{
    handle.ptr += offsetInDescriptors * descriptorIncrementSize;
    return handle;
}

uint64_t Signal(ID3D12CommandQueue* commandQueue, ID3D12Fence* fence,
                uint64_t* fenceValue)
{
    (*fenceValue)++;
    ExitOnFailure(ID3D12CommandQueue_Signal(commandQueue, fence, *fenceValue));

    return *fenceValue;
}

HANDLE CreateEventHandle()
{
    HANDLE fenceEvent;

    fenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    assert(fenceEvent && "Failed to create fence event.");

    return fenceEvent;
}

void WaitForFenceValue(ID3D12Fence* fence, uint64_t fenceValue, HANDLE fenceEvent, DWORD duration)
{
    if (ID3D12Fence_GetCompletedValue(fence) < fenceValue)
    {
        ExitOnFailure(ID3D12Fence_SetEventOnCompletion(fence, fenceValue, fenceEvent));
        WaitForSingleObject(fenceEvent, duration ? duration : INFINITE);
    }
}

ID3DBlob* LoadShader(LPCWSTR path, LPCSTR target)
{
    ID3DBlob* shaderBlob = NULL;
    ID3DBlob* errorBlob = NULL;
    UINT flags = D3DCOMPILE_DEBUG | D3DCOMPILE_DEBUG_NAME_FOR_SOURCE | D3DCOMPILE_PARTIAL_PRECISION | D3DCOMPILE_OPTIMIZATION_LEVEL3;
    HRESULT hr = D3DCompileFromFile(path, NULL,
        D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", target, flags, 0, &shaderBlob, &errorBlob);

    if (FAILED(hr))
    {
        if (errorBlob != NULL)
        {
            char* data = ID3DBlob_GetBufferPointer(errorBlob);
            OutputDebugString(data);
            ID3DBlob_Release(errorBlob);
            exit(HD_EXIT_FAILURE);
        }
        else
        {
            OutputDebugString("Invalid shader path");
        }
    }

    return shaderBlob;
}

// From d3dx12.h, converted to C
//------------------------------------------------------------------------------------------------
// D3D12 exports a new method for serializing root signatures in the Windows 10 Anniversary Update.
// To help enable root signature 1.1 features when they are available and not require maintaining
// two code paths for building root signatures, this helper method reconstructs a 1.0 signature when
// 1.1 is not supported.
inline HRESULT D3DX12SerializeVersionedRootSignature(
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignatureDesc,
    D3D_ROOT_SIGNATURE_VERSION MaxVersion,
    ID3DBlob** ppBlob,
    ID3DBlob** ppErrorBlob)
{
    if (ppErrorBlob != NULL)
    {
        *ppErrorBlob = NULL;
    }

    switch (MaxVersion)
    {
        case D3D_ROOT_SIGNATURE_VERSION_1_0:
            switch (pRootSignatureDesc->Version)
            {
                case D3D_ROOT_SIGNATURE_VERSION_1_0:
                    return D3D12SerializeRootSignature(&pRootSignatureDesc->Desc_1_0, D3D_ROOT_SIGNATURE_VERSION_1, ppBlob, ppErrorBlob);

                case D3D_ROOT_SIGNATURE_VERSION_1_1:
                {
                    HRESULT hr = S_OK;
                    const D3D12_ROOT_SIGNATURE_DESC1* desc_1_1 = &pRootSignatureDesc->Desc_1_1;

                    const SIZE_T ParametersSize = sizeof(D3D12_ROOT_PARAMETER) * desc_1_1->NumParameters;
                    void* pParameters = (ParametersSize > 0) ? HeapAlloc(GetProcessHeap(), 0, ParametersSize) : NULL;
                    if (ParametersSize > 0 && pParameters == NULL)
                    {
                        hr = E_OUTOFMEMORY;
                    }
                    D3D12_ROOT_PARAMETER* pParameters_1_0 = (D3D12_ROOT_PARAMETER*)pParameters;

                    if (SUCCEEDED(hr))
                    {
                        for (UINT n = 0; n < desc_1_1->NumParameters; n++)
                        {
                            pParameters_1_0[n].ParameterType = desc_1_1->pParameters[n].ParameterType;
                            pParameters_1_0[n].ShaderVisibility = desc_1_1->pParameters[n].ShaderVisibility;

                            switch (desc_1_1->pParameters[n].ParameterType)
                            {
                            case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
                                pParameters_1_0[n].Constants.Num32BitValues = desc_1_1->pParameters[n].Constants.Num32BitValues;
                                pParameters_1_0[n].Constants.RegisterSpace = desc_1_1->pParameters[n].Constants.RegisterSpace;
                                pParameters_1_0[n].Constants.ShaderRegister = desc_1_1->pParameters[n].Constants.ShaderRegister;
                                break;

                            case D3D12_ROOT_PARAMETER_TYPE_CBV:
                            case D3D12_ROOT_PARAMETER_TYPE_SRV:
                            case D3D12_ROOT_PARAMETER_TYPE_UAV:
                                pParameters_1_0[n].Descriptor.RegisterSpace = desc_1_1->pParameters[n].Descriptor.RegisterSpace;
                                pParameters_1_0[n].Descriptor.ShaderRegister = desc_1_1->pParameters[n].Descriptor.ShaderRegister;
                                break;

                            case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE:
                                const D3D12_ROOT_DESCRIPTOR_TABLE1* table_1_1 = &desc_1_1->pParameters[n].DescriptorTable;

                                const SIZE_T DescriptorRangesSize = sizeof(D3D12_DESCRIPTOR_RANGE) * table_1_1->NumDescriptorRanges;
                                void* pDescriptorRanges = (DescriptorRangesSize > 0 && SUCCEEDED(hr)) ? HeapAlloc(GetProcessHeap(), 0, DescriptorRangesSize) : NULL;
                                if (DescriptorRangesSize > 0 && pDescriptorRanges == NULL)
                                {
                                    hr = E_OUTOFMEMORY;
                                }
                                D3D12_DESCRIPTOR_RANGE* pDescriptorRanges_1_0 = (D3D12_DESCRIPTOR_RANGE*)pDescriptorRanges;

                                if (SUCCEEDED(hr))
                                {
                                    for (UINT x = 0; x < table_1_1->NumDescriptorRanges; x++)
                                    {
                                        pDescriptorRanges_1_0[x].BaseShaderRegister = table_1_1->pDescriptorRanges[x].BaseShaderRegister;
                                        pDescriptorRanges_1_0[x].NumDescriptors = table_1_1->pDescriptorRanges[x].NumDescriptors;
                                        pDescriptorRanges_1_0[x].OffsetInDescriptorsFromTableStart = table_1_1->pDescriptorRanges[x].OffsetInDescriptorsFromTableStart;
                                        pDescriptorRanges_1_0[x].RangeType = table_1_1->pDescriptorRanges[x].RangeType;
                                        pDescriptorRanges_1_0[x].RegisterSpace = table_1_1->pDescriptorRanges[x].RegisterSpace;
                                    }
                                }

                                D3D12_ROOT_DESCRIPTOR_TABLE* table_1_0 = &pParameters_1_0[n].DescriptorTable;
                                table_1_0->NumDescriptorRanges = table_1_1->NumDescriptorRanges;
                                table_1_0->pDescriptorRanges = pDescriptorRanges_1_0;
                            }
                        }
                    }

                    if (SUCCEEDED(hr))
                    {
                        D3D12_ROOT_SIGNATURE_DESC desc_1_0;
                        desc_1_0.NumParameters = desc_1_1->NumParameters;
                        desc_1_0.pParameters = pParameters_1_0;
                        desc_1_0.NumStaticSamplers = desc_1_1->NumStaticSamplers;
                        desc_1_0.pStaticSamplers = desc_1_1->pStaticSamplers;
                        desc_1_0.Flags = desc_1_1->Flags;
                        hr = D3D12SerializeRootSignature(&desc_1_0, D3D_ROOT_SIGNATURE_VERSION_1, ppBlob, ppErrorBlob);
                    }

                    if (pParameters)
                    {
                        for (UINT n = 0; n < desc_1_1->NumParameters; n++)
                        {
                            if (desc_1_1->pParameters[n].ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
                            {
                                HeapFree(GetProcessHeap(), 0, (void*)((D3D12_DESCRIPTOR_RANGE*)(pParameters_1_0[n].DescriptorTable.pDescriptorRanges)));
                            }
                        }
                        HeapFree(GetProcessHeap(), 0, pParameters);
                    }
                    return hr;
                }
            }
            break;

        case D3D_ROOT_SIGNATURE_VERSION_1_1:
            return D3D12SerializeVersionedRootSignature(pRootSignatureDesc, ppBlob, ppErrorBlob);
    }

    return E_INVALIDARG;
}

ID3D12RootSignature* CreateRootSignature(ID3D12Device2* device)
{
    // Create a root signature.
    D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {0};
    featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
    if (FAILED(ID3D12Device2_CheckFeatureSupport(device,
        D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData))))
    {
        featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    // Allow input layout and deny unnecessary access to certain pipeline stages.
    D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDescription;
    rootSignatureDescription.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rootSignatureDescription.Desc_1_1.NumParameters = 0;
    rootSignatureDescription.Desc_1_1.pParameters = NULL;
    rootSignatureDescription.Desc_1_1.NumStaticSamplers = 0;
    rootSignatureDescription.Desc_1_1.pStaticSamplers = NULL;
    rootSignatureDescription.Desc_1_1.Flags = rootSignatureFlags;

    // Serialize the root signature.
    ID3DBlob* rootSignatureBlob;
    ID3DBlob* errorBlob;
    ExitOnFailure(D3DX12SerializeVersionedRootSignature(&rootSignatureDescription,
        featureData.HighestVersion, &rootSignatureBlob, &errorBlob));
    // Create the root signature.
    ID3D12RootSignature* rootSignature;
    ExitOnFailure(ID3D12Device2_CreateRootSignature(device, 0, ID3DBlob_GetBufferPointer(rootSignatureBlob),
        ID3DBlob_GetBufferSize(rootSignatureBlob), &IID_ID3D12RootSignature, &rootSignature));

    return rootSignature;
}

D3D12_SHADER_BYTECODE D3D12_SHADER_BYTECODE_Init(ID3DBlob* blob)
{
    D3D12_SHADER_BYTECODE bytecode = {
        .BytecodeLength = ID3DBlob_GetBufferSize(blob),
        .pShaderBytecode = ID3DBlob_GetBufferPointer(blob)
    };

    return bytecode;
}

ID3D12PipelineState* CreatePipelineState(ID3D12Device2* device,
    ID3D12RootSignature* rootSignature, ID3DBlob* vertexShaderBlob, ID3DBlob* pixelShaderBlob)
{
    D3D12_SHADER_BYTECODE vertexShaderBytecode = D3D12_SHADER_BYTECODE_Init(vertexShaderBlob);
    D3D12_SHADER_BYTECODE pixelShaderBytecode = D3D12_SHADER_BYTECODE_Init(pixelShaderBlob);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipelineStateDesc = {
        .pRootSignature = rootSignature,
        .InputLayout = { NULL, 0 },
        .PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        .RasterizerState = {
            .DepthClipEnable = TRUE,
            .FillMode = D3D12_FILL_MODE_SOLID,
            .CullMode = D3D12_CULL_MODE_BACK
        },
        .VS = vertexShaderBytecode,
        .PS = pixelShaderBytecode,
        .DSVFormat = DXGI_FORMAT_D32_FLOAT,
        .RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM,
        .NumRenderTargets = 1,
        .SampleDesc = {
            .Count = 1,
            .Quality = 0
        },
        .DepthStencilState = {
            .DepthEnable = TRUE,
            .DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL,
            .DepthFunc = D3D12_COMPARISON_FUNC_LESS,
            .StencilEnable = FALSE
        },
        .SampleMask = UINT_MAX,
        .BlendState = {
            .RenderTarget[0] = {
                .RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL
            }
        }
    };

    ID3D12PipelineState* pipelineState;
    ID3D12Device2_CreateGraphicsPipelineState(device, &pipelineStateDesc, &IID_ID3D12PipelineState, &pipelineState);

    return pipelineState;
}

void ResizeDepthBuffer(ID3D12Device2* device, int width, int height, ID3D12Resource** depthBuffer)
{
    WaitForFenceValue(g_Fence, g_FenceValue, g_FenceEvent, 0);

    width = MAX(1, width);
    height = MAX(1, height);

    // Resize screen dependent resources.
    // Create a depth buffer.
    D3D12_CLEAR_VALUE optimizedClearValue = {
        .Color = { 0.0f, 0.0f, 0.0f, 0.0f },
        .Format = DXGI_FORMAT_D32_FLOAT,
        .DepthStencil = {
            .Depth = 1.0f,
            .Stencil = 0.0f
        }
    };

    D3D12_HEAP_PROPERTIES heapProperties = {
        .Type = D3D12_HEAP_TYPE_DEFAULT,
        .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
        .CreationNodeMask = 1,
        .VisibleNodeMask = 1
    };

    D3D12_RESOURCE_DESC resourceDesc = {
        .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        .Alignment = 0,
        .Width = width,
        .Height = height,
        .DepthOrArraySize = 1,
        .MipLevels = 0,
        .Format = DXGI_FORMAT_D32_FLOAT,
        .SampleDesc = {
            .Count = 1,
            .Quality = 0
        },
        .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        .Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
    };

    if (*depthBuffer != NULL) ID3D12Resource_Release(*depthBuffer);
    ExitOnFailure(ID3D12Device2_CreateCommittedResource(
        device,
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &optimizedClearValue,
        &IID_ID3D12Resource,
        depthBuffer
    ));

    // Update the depth-stencil view.
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {
        .Format = DXGI_FORMAT_D32_FLOAT,
        .ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D,
        .Texture2D.MipSlice = 0,
        .Flags = D3D12_DSV_FLAG_NONE
    };

    D3D12_CPU_DESCRIPTOR_HANDLE descHandle = {0};
    ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(g_DSVDescriptorHeap, &descHandle);
    ID3D12Device2_CreateDepthStencilView(device, *depthBuffer, &dsv, descHandle);
}

void Render(IDXGISwapChain4* swapChain, ID3D12CommandQueue* g_CommandQueue,
            ID3D12GraphicsCommandList* commandList, ID3D12PipelineState* pipelineState,
            ID3D12RootSignature* rootSignature, D3D12_VIEWPORT* viewport,
            D3D12_RECT* scisssorRect)
{
    ID3D12CommandAllocator* commandAllocator = g_CommandAllocators[g_CurrentBackBufferIndex];
    ID3D12Resource* backBuffer = g_BackBuffers[g_CurrentBackBufferIndex];

    ID3D12CommandAllocator_Reset(commandAllocator);
    ID3D12GraphicsCommandList_Reset(commandList, commandAllocator, NULL);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv;
    ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(g_DSVDescriptorHeap, &dsv);
    // Clear the render target.
    {
        D3D12_RESOURCE_BARRIER barrier = D3D12_RESOURCE_BARRIER_Transition(backBuffer,
                                                    D3D12_RESOURCE_STATE_PRESENT,
                                                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                                    D3D12_RESOURCE_BARRIER_FLAG_NONE);
        ID3D12GraphicsCommandList_ResourceBarrier(commandList, 1, &barrier);

        FLOAT clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };

        ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(g_RTVDescriptorHeap, &rtv);
        rtv = D3D12_CPU_DESCRIPTOR_HANDLE_Offset(rtv, g_CurrentBackBufferIndex, g_RTVDescriptorSize);

        ID3D12GraphicsCommandList_ClearRenderTargetView(commandList, rtv, clearColor, 0, NULL);
        ID3D12GraphicsCommandList_ClearDepthStencilView(commandList, dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, NULL);
    }

    ID3D12GraphicsCommandList_SetPipelineState(commandList, pipelineState);
    ID3D12GraphicsCommandList_SetGraphicsRootSignature(commandList, rootSignature);

    ID3D12GraphicsCommandList_IASetPrimitiveTopology(commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    ID3D12GraphicsCommandList_RSSetViewports(commandList, 1, viewport);
    ID3D12GraphicsCommandList_RSSetScissorRects(commandList, 1, scisssorRect);

    ID3D12GraphicsCommandList_OMSetRenderTargets(commandList, 1, &rtv, FALSE, &dsv);
    ID3D12GraphicsCommandList_DrawInstanced(commandList, 3, 1, 0, 0);

    // Present
    {
        D3D12_RESOURCE_BARRIER barrier = D3D12_RESOURCE_BARRIER_Transition(backBuffer,
                                                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_STATE_PRESENT,
                                                    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                                    D3D12_RESOURCE_BARRIER_FLAG_NONE);
        ID3D12GraphicsCommandList_ResourceBarrier(commandList, 1, &barrier);

        ExitOnFailure(ID3D12GraphicsCommandList_Close(commandList));

        ID3D12CommandList* const commandLists[] = { (ID3D12CommandList* const)commandList };
        ID3D12CommandQueue_ExecuteCommandLists(g_CommandQueue, _countof(commandLists), commandLists);

        g_FrameFenceValues[g_CurrentBackBufferIndex] = Signal(g_CommandQueue, g_Fence, &g_FenceValue);

        UINT syncInterval = 1;
        UINT presentFlags = 0;
        ExitOnFailure(IDXGISwapChain4_Present(swapChain, syncInterval, presentFlags));

        g_CurrentBackBufferIndex = IDXGISwapChain4_GetCurrentBackBufferIndex(swapChain);

        WaitForFenceValue(g_Fence, g_FrameFenceValues[g_CurrentBackBufferIndex], g_FenceEvent, 0);
    }
}

void Flush(ID3D12CommandQueue* commandQueue, ID3D12Fence* fence,
    uint64_t* fenceValue, HANDLE fenceEvent)
{
    uint64_t fenceValueForSignal = Signal(commandQueue, fence, fenceValue);
    WaitForFenceValue(fence, fenceValueForSignal, fenceEvent, 0);
}

void ResizeSwapchain(int width, int height)
{
    width = MAX(1, width);
    height = MAX(1, height);

    Flush(g_CommandQueue, g_Fence, &g_FenceValue, g_FenceEvent);

    for (int i = 0; i < FRAMES_NUM; ++i)
    {
        ID3D12Resource_Release(g_BackBuffers[i]);
    }

    DXGI_SWAP_CHAIN_DESC swapChainDesc = {0};
    ExitOnFailure(IDXGISwapChain4_GetDesc(g_SwapChain, &swapChainDesc));
    ExitOnFailure(IDXGISwapChain4_ResizeBuffers(g_SwapChain, FRAMES_NUM, width, height, swapChainDesc.BufferDesc.Format, swapChainDesc.Flags));
    
    g_CurrentBackBufferIndex = IDXGISwapChain4_GetCurrentBackBufferIndex(g_SwapChain);

    UpdateRenderTargetViews(g_Device, g_SwapChain, g_RTVDescriptorHeap);
}

static int Resize(void* data, SDL_Event* event)
{
    if (event->type != SDL_WINDOWEVENT ||
        event->window.event != SDL_WINDOWEVENT_RESIZED) 
        return 1;
    
    SDL_Window* window = SDL_GetWindowFromID(event->window.windowID);
    if (window != (SDL_Window*)data)
        return 2;

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    ResizeData* resizeData = SDL_GetWindowData(window, "resize");
    resizeData->viewport->Width = (float)width;
    resizeData->viewport->Height = (float)height;

    ResizeSwapchain(width, height);
    ResizeDepthBuffer(resizeData->device, width, height, resizeData->depthBuffer);
    return 0;
}

int InitRenderer(SDL_Window* window)
{
#ifdef _DEBUG
    EnableDebuggingLayer();
#endif

    g_ResizeData;
    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    SDL_SetWindowData(window, "resize", (void*)&g_ResizeData);
    SDL_AddEventWatch(Resize, window);

    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);
    HWND hwnd = wmInfo.info.win.window;

    IDXGIAdapter4* dxgiAdapter4 = GetAdapter();
    g_Device = CreateDevice(dxgiAdapter4);
    g_CommandQueue = CreateCommandQueue(g_Device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    g_SwapChain = CreateSwapChain(hwnd,
                                                 g_CommandQueue, width, height,
                                                 FRAMES_NUM);

    g_CurrentBackBufferIndex = IDXGISwapChain4_GetCurrentBackBufferIndex(g_SwapChain);

    g_RTVDescriptorHeap = CreateDescriptorHeap(g_Device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, FRAMES_NUM);
    g_DSVDescriptorHeap = CreateDescriptorHeap(g_Device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1);
    g_RTVDescriptorSize = ID3D12Device2_GetDescriptorHandleIncrementSize(g_Device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    UpdateRenderTargetViews(g_Device, g_SwapChain, g_RTVDescriptorHeap);

    for (int i = 0; i < FRAMES_NUM; ++i)
    {
        g_CommandAllocators[i] = CreateCommandAllocator(g_Device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    }

    g_CommandList = CreateCommandList(g_Device,
        g_CommandAllocators[g_CurrentBackBufferIndex], D3D12_COMMAND_LIST_TYPE_DIRECT);

    g_Fence = CreateFence(g_Device);
    g_FenceEvent = CreateEventHandle();

    // Load the vertex shader.
    ID3DBlob* vertexShaderBlob = LoadShader(L"shaders/vertex.hlsl", "vs_5_1");
    // Load the pixel shader.
    ID3DBlob* pixelShaderBlob = LoadShader(L"shaders/pixel.hlsl", "ps_5_1");

    // Create depth buffer
    g_DepthBuffer = NULL;
    ResizeDepthBuffer(g_Device, width, height, &g_DepthBuffer);

    // Root signature
    g_RootSignature = CreateRootSignature(g_Device);

    // Pipeline state object.
    g_PipelineState = CreatePipelineState(g_Device, g_RootSignature, vertexShaderBlob, pixelShaderBlob);

    D3D12_VIEWPORT viewport = { 0.0f, 0.0f, (float)width, (float)height, D3D12_MIN_DEPTH, D3D12_MAX_DEPTH };
    g_Viewport = viewport;
    D3D12_RECT scissors = { 0, 0, LONG_MAX, LONG_MAX };
    g_ScissorRect = scissors;

    g_ResizeData.device = g_Device;
    g_ResizeData.depthBuffer = &g_DepthBuffer;
    g_ResizeData.viewport = &g_Viewport;
    g_ResizeData.fov = 45.0f;

    IDXGIAdapter4_Release(dxgiAdapter4);
    ID3DBlob_Release(vertexShaderBlob);
    ID3DBlob_Release(pixelShaderBlob);
}

static void RendererUpdate()
{
    Render(g_SwapChain, g_CommandQueue, g_CommandList, g_PipelineState,
               g_RootSignature, &g_Viewport, &g_ScissorRect);
}

static void RendererClose()
{
     // Make sure the command queue has finished all commands before closing.
     Flush(g_CommandQueue, g_Fence, &g_FenceValue, g_FenceEvent);

     CloseHandle(g_FenceEvent);

    ID3D12Resource_Release(g_DepthBuffer);
    ID3D12PipelineState_Release(g_PipelineState);
    ID3D12RootSignature_Release(g_RootSignature);
    ID3D12Fence_Release(g_Fence);
    ID3D12GraphicsCommandList_Release(g_CommandList);
    for (int i = 0; i < FRAMES_NUM; ++i)
    {
        ID3D12CommandAllocator_Release(g_CommandAllocators[i]);
    }
    ID3D12DescriptorHeap_Release(g_DSVDescriptorHeap);
    ID3D12DescriptorHeap_Release(g_RTVDescriptorHeap);
    IDXGISwapChain4_Release(g_SwapChain);
    IDXGISwapChain4_Release(g_SwapChain);
    IDXGISwapChain4_Release(g_SwapChain);
    ID3D12CommandQueue_Release(g_CommandQueue);
    ID3D12Device2_Release(g_Device);
    ID3D12Device2_Release(g_Device);

 #ifdef _DEBUG
    REPORT_LIVE_OBJ();
 #endif
}

int main()
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("D3D12 - Hello Triangle", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 640, 480, SDL_WINDOW_RESIZABLE);

    // Check that the window was successfully created
    if (window == NULL) {
        // In the case that the window could not be made...
        SDL_LogCritical(SDL_LOG_CATEGORY_ERROR, "Could not create window: %s\n", SDL_GetError());
        return 1;
    }

    InitRenderer(window);

    while (1)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e) > 0)
        {
            if (e.type == SDL_QUIT)
                goto quit;
        }

        RendererUpdate();
    }

    quit:
    RendererClose();
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}