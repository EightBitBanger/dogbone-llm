#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cstddef>

#define GLEW_STATIC
#include <GL/glew.h>

class ShaderTensor {
public:
    // Construct with no program bound.
    ShaderTensor();

    // Release GL objects (buffers/programs) and unmap mapped buffers.
    ~ShaderTensor();

    // Build a compute shader program from a GLSL source string.
    bool buildComputeFromSource(const std::string &src, const std::string &debugName = "");

    // Build a compute shader program by loading a file from disk.
    bool buildComputeFromFile(const std::string &path, const std::string &debugName = "");

    // Bind the internal program for subsequent uniform sets/dispatch.
    void use() const;

    // Dispatch the compute shader with given workgroup grid.
    void dispatch(unsigned gx, unsigned gy=1, unsigned gz=1) const;

    // Insert a memory barrier so SSBO writes are visible to subsequent passes.
    void barrierStorage() const;

    // Insert a conservative barrier covering all write/read hazards.
    void barrierAll() const;

    // Set an int uniform by name (cached lookup).
    void setInt(const std::string &name, int v);

    // Set an unsigned int uniform by name (cached lookup).
    void setUInt(const std::string &name, unsigned v);

    // Set a float uniform by name (cached lookup).
    void setFloat(const std::string &name, float v);

    // Set an ivec2 uniform by name.
    void setIVec2(const std::string &name, int x,int y);

    // Set an ivec3 uniform by name.
    void setIVec3(const std::string &name, int x,int y,int z);

    // Set a vec2 uniform by name.
    void setVec2(const std::string &name, float x,float y);

    // Set a vec3 uniform by name.
    void setVec3(const std::string &name, float x,float y,float z);

    // Set a mat4 uniform (16 floats) by name.
    void setMat4(const std::string &name, const float *m16, bool transpose=false);

    // Bind a named uniform block in the program to a binding index.
    void bindUniformBlock(const std::string &blockName, unsigned bindingIndex);

    // Create an SSBO (for tensors), optionally persistently mapped for fast uploads.
    unsigned createSSBO(const std::string &name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                        unsigned mapFlags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
                        const void* initialData = nullptr);

    // Adopt an existing SSBO id and track/bind it under a friendly name.
    void adoptSSBO(const std::string &name, unsigned existing, std::ptrdiff_t sizeBytes, unsigned bindingIndex);

    // Upload raw bytes into a named SSBO (memcpy if persistently mapped).
    void upload(const std::string &name, const void* src, std::ptrdiff_t bytes, std::ptrdiff_t offset=0);

    // Read back bytes synchronously from a named SSBO (may stall the GPU).
    void downloadSync(const std::string &name, void* dst, std::ptrdiff_t bytes, std::ptrdiff_t offset=0);

    // Rebind a named SSBO to its recorded binding index.
    void rebindSSBO(const std::string &name);

    // Create a UBO for small metadata structs (std140), optionally persistently mapped.
    unsigned createUBO(const std::string &name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                       unsigned mapFlags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
                       const void* initialData=nullptr);

    // Update bytes in a named UBO (memcpy if persistently mapped).
    void updateUBO(const std::string &name, const void* src, std::ptrdiff_t bytes, std::ptrdiff_t offset=0);

    // Begin an async readback by copying an SSBO region into a staging buffer and fencing it.
    struct ReadbackHandle { std::uint64_t id=0; };
    ReadbackHandle startReadback(const std::string &srcName, std::ptrdiff_t bytes, std::ptrdiff_t srcOffset=0);

    // Check if the async readback has completed without blocking.
    bool pollReadback(ReadbackHandle h);

    // Block (with timeout ns) until the async readback completes.
    void waitReadback(ReadbackHandle h, std::uint64_t timeoutNS = 1000000000ULL);

    // Get the CPU pointer to the staging buffer for a completed readback.
    const void* mapReadback(ReadbackHandle h);

    // Release resources associated with a readback handle.
    void finishReadback(ReadbackHandle h);

    // Attach human-readable labels for debugging in GL debuggers.
    void labelBuffer(const std::string &name, const char* label);

    // Label the GL program object.
    void labelProgram(const char* label);

    // Fetch the underlying GL program object id.
    unsigned program() const;

private:
    // Query/cached lookup of a uniform location by name.
    int loc(const std::string &name);

    // Check a shader's compile status and print its info log.
    bool checkShader(unsigned sh, const char* dbg);

    // Check a program's link status and print its info log.
    bool checkProgram(unsigned prog, const char* dbg);

    // Staging object for async readback bookkeeping.
    struct RB { unsigned staging=0; void* ptr=nullptr; std::ptrdiff_t size=0; std::ptrdiff_t offset=0; unsigned flags=0; void* fence=0; };

    // Destroy a readback staging buffer and its sync fence.
    void destroyReadback(RB &rb);

    // Internal record for SSBOs.
    struct Buffer { unsigned id=0; std::ptrdiff_t size=0; unsigned binding=0; void* map=nullptr; unsigned flags=0; };

    // Internal record for UBOs.
    struct UBO { unsigned id=0; std::ptrdiff_t size=0; unsigned binding=0; void* map=nullptr; unsigned flags=0; };

private:
    unsigned program_ = 0;
    std::string name_;
    std::unordered_map<std::string, int> uniformCache_;
    std::unordered_map<std::string, Buffer> buffers_;
    std::unordered_map<std::string, UBO>    ubos_;
    std::uint64_t nextRbId_ = 0;
    std::unordered_map<std::uint64_t, RB> readbacks_;
};
