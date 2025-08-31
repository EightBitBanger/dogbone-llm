#ifndef SHADER_TENSOR_H
#define SHADER_TENSOR_H

#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cstddef>

#define GLEW_STATIC
#include <GL/glew.h>

class ShaderTensor {
public:
    ShaderTensor();
    // Upload into an existing SSBO by GL id
    void uploadRawSSBO(unsigned id, std::ptrdiff_t offset, const void* data, std::ptrdiff_t size);
    ~ShaderTensor();

    bool buildComputeFromSource(const char* src, const char* debugLabel=nullptr);
    void use() const;
    void dispatch(unsigned gx, unsigned gy, unsigned gz) const;

    void setInt(const std::string& name, int v);
    void setUInt(const std::string& name, unsigned v);
    void setFloat(const std::string& name, float v);
    void setIVec2(const std::string& name, int x, int y);

    unsigned createSSBO(const std::string& name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                        unsigned mapFlags = 0, const void* initialData=nullptr);
    // Ensure an SSBO exists with given size and binding; reuse if possible
    unsigned ensureSSBO(const std::string& name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                        unsigned mapFlags = 0, const void* initialData=nullptr);
    void adoptSSBO(const std::string& name, unsigned id, std::ptrdiff_t sizeBytes, unsigned bindingIndex);
    void upload(const std::string& name, const void* src, std::ptrdiff_t bytes, std::ptrdiff_t offset=0);
    void downloadSync(const std::string& name, void* dst, std::ptrdiff_t bytes, std::ptrdiff_t offset=0);
    void rebindSSBO(const std::string& name) const;

    unsigned createUBO(const std::string& name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                       unsigned mapFlags = GL_MAP_WRITE_BIT, const void* initialData=nullptr);
    unsigned ensureUBO(const std::string& name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                                 unsigned mapFlags, const void* initialData);
    void updateUBO(const std::string& name, const void* src, std::ptrdiff_t bytes, std::ptrdiff_t offset=0);

    struct ReadbackHandle { std::uint64_t id=0; };
    ReadbackHandle startReadback(const std::string& srcName, std::ptrdiff_t bytes, std::ptrdiff_t srcOffset=0);
    bool pollReadback(ReadbackHandle h);
    bool waitReadback(ReadbackHandle h, std::uint64_t timeoutNanoseconds);
    const void* mapReadback(ReadbackHandle h);
    void finishReadback(ReadbackHandle h);

    void sync();

private:
    struct Buffer { unsigned id=0; std::ptrdiff_t size=0; unsigned binding=0; void* map=nullptr; unsigned flags=0; };
    struct UBO    { unsigned id=0; std::ptrdiff_t size=0; unsigned binding=0; void* map=nullptr; unsigned flags=0; };
    struct RB     { unsigned staging=0; void* ptr=nullptr; std::ptrdiff_t size=0; std::ptrdiff_t offset=0; unsigned flags=0; GLsync fence=0; };

    unsigned program_ = 0;
    mutable std::unordered_map<std::string, int> uniformCache_;
    std::unordered_map<std::string, Buffer> buffers_;
    std::unordered_map<std::string, UBO>    ubos_;
    std::uint64_t nextRbId_ = 0;
    std::unordered_map<std::uint64_t, RB> readbacks_;
};

#endif // SHADER_TENSOR_H
