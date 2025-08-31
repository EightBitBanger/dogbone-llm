#include "ShaderTensor.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

ShaderTensor::ShaderTensor() = default;

ShaderTensor::~ShaderTensor() {
    for (std::pair<const std::string, Buffer>& kv : buffers_) {
        if (kv.second.map) glUnmapNamedBuffer(kv.second.id);
        glDeleteBuffers(1, &kv.second.id);
    }
    for (std::pair<const std::string, UBO>& kv : ubos_) {
        if (kv.second.map) glUnmapNamedBuffer(kv.second.id);
        glDeleteBuffers(1, &kv.second.id);
    }
    for (auto &rb : readbacks_) destroyReadback(rb.second);
    if (program_) glDeleteProgram(program_);
}

bool ShaderTensor::buildComputeFromSource(const std::string &src, const std::string &debugName) {
    unsigned cs = glCreateShader(GL_COMPUTE_SHADER);
    const char* csrc = src.c_str();
    glShaderSource(cs, 1, &csrc, nullptr);
    glCompileShader(cs);
    if (!checkShader(cs, debugName.c_str())) { glDeleteShader(cs); return false; }

    program_ = glCreateProgram();
    glAttachShader(program_, cs);
    glLinkProgram(program_);
    glDeleteShader(cs);
    if (!checkProgram(program_, debugName.c_str())) { glDeleteProgram(program_); program_ = 0; return false; }

    uniformCache_.clear();
    name_ = debugName;
    return true;
}

// Build a compute shader program by loading a file from disk.
bool ShaderTensor::buildComputeFromFile(const std::string &path, const std::string &debugName) {
    std::ifstream f(path);
    if (!f) { std::cerr << "[ShaderTensor] failed to open file: " << path << "\n"; return false; }
    std::stringstream ss; ss << f.rdbuf();
    return buildComputeFromSource(ss.str(), debugName.empty()? path : debugName);
}

// Bind the internal program for subsequent uniform sets/dispatch.
void ShaderTensor::use() const { glUseProgram(program_); }

// Dispatch the compute shader with given workgroup grid.
void ShaderTensor::dispatch(unsigned gx, unsigned gy, unsigned gz) const {
    glUseProgram(program_);
    glDispatchCompute(gx, gy, gz);
}

// Insert a memory barrier so SSBO writes are visible to subsequent passes.
void ShaderTensor::barrierStorage() const {
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

// Insert a conservative barrier covering all write/read hazards.
void ShaderTensor::barrierAll() const {
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

// Set an int uniform by name (cached lookup).
void ShaderTensor::setInt(const std::string &name, int v) { glUniform1i(loc(name), v); }

// Set an unsigned int uniform by name (cached lookup).
void ShaderTensor::setUInt(const std::string &name, unsigned v) { glUniform1ui(loc(name), v); }

// Set a float uniform by name (cached lookup).
void ShaderTensor::setFloat(const std::string &name, float v) { glUniform1f(loc(name), v); }

// Set an ivec2 uniform by name.
void ShaderTensor::setIVec2(const std::string &name, int x,int y) { glUniform2i(loc(name), x,y); }

// Set an ivec3 uniform by name.
void ShaderTensor::setIVec3(const std::string &name, int x,int y,int z) { glUniform3i(loc(name), x,y,z); }

// Set a vec2 uniform by name.
void ShaderTensor::setVec2(const std::string &name, float x,float y) { glUniform2f(loc(name), x,y); }

// Set a vec3 uniform by name.
void ShaderTensor::setVec3(const std::string &name, float x,float y,float z) { glUniform3f(loc(name), x,y,z); }

// Set a mat4 uniform (16 floats) by name.
void ShaderTensor::setMat4(const std::string &name, const float *m16, bool transpose) {
    glUniformMatrix4fv(loc(name), 1, transpose? GL_TRUE:GL_FALSE, m16);
}

// Bind a named uniform block in the program to a binding index.
void ShaderTensor::bindUniformBlock(const std::string &blockName, unsigned bindingIndex) {
    unsigned idx = glGetUniformBlockIndex(program_, blockName.c_str());
    if (idx == GL_INVALID_INDEX) {
        std::cerr << "[ShaderTensor] uniform block not found: " << blockName << "\n";
        return;
    }
    glUniformBlockBinding(program_, idx, bindingIndex);
}

// Create an SSBO (for tensors), optionally persistently mapped for fast uploads.
unsigned ShaderTensor::createSSBO(const std::string &name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                                  unsigned mapFlags, const void* initialData) {
    Buffer buf{}; buf.binding = bindingIndex; buf.size = sizeBytes; buf.flags = mapFlags;
    glCreateBuffers(1, &buf.id);
    glNamedBufferStorage(buf.id, sizeBytes, initialData, mapFlags);
    if (mapFlags) buf.map = glMapNamedBufferRange(buf.id, 0, sizeBytes, mapFlags);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingIndex, buf.id);
    buffers_[name] = buf;
    return buf.id;
}

// Adopt an existing SSBO id and track/bind it under a friendly name.
void ShaderTensor::adoptSSBO(const std::string &name, unsigned existing, std::ptrdiff_t sizeBytes, unsigned bindingIndex) {
    Buffer buf{}; buf.id = existing; buf.size = sizeBytes; buf.binding = bindingIndex; buf.flags = 0; buf.map = nullptr;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingIndex, buf.id);
    buffers_[name] = buf;
}

// Upload raw bytes into a named SSBO (memcpy if persistently mapped).
void ShaderTensor::upload(const std::string &name, const void* src, std::ptrdiff_t bytes, std::ptrdiff_t offset) {
    std::unordered_map<std::string, Buffer>::iterator it = buffers_.find(name); if (it == buffers_.end()) { std::cerr << "[ShaderTensor] SSBO not found: " << name << "\n"; return; }
    Buffer &b = it->second; if (offset + bytes > b.size) { std::cerr << "[ShaderTensor] upload out of range on " << name << "\n"; return; }
    if (b.map) std::memcpy(static_cast<char*>(b.map) + offset, src, (size_t)bytes);
    else glNamedBufferSubData(b.id, offset, bytes, src);
}

// Read back bytes synchronously from a named SSBO (may stall the GPU).
void ShaderTensor::downloadSync(const std::string &name, void* dst, std::ptrdiff_t bytes, std::ptrdiff_t offset) {
    std::unordered_map<std::string, Buffer>::iterator it = buffers_.find(name); if (it == buffers_.end()) { std::cerr << "[ShaderTensor] SSBO not found: " << name << "\n"; return; }
    Buffer &b = it->second; if (offset + bytes > b.size) { std::cerr << "[ShaderTensor] download out of range on " << name << "\n"; return; }
    glGetNamedBufferSubData(b.id, offset, bytes, dst);
}

// Rebind a named SSBO to its recorded binding index.
void ShaderTensor::rebindSSBO(const std::string &name) {
    std::unordered_map<std::string, Buffer>::iterator it = buffers_.find(name); if (it == buffers_.end()) { std::cerr << "[ShaderTensor] SSBO not found: " << name << "\n"; return; }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, it->second.binding, it->second.id);
}

// Create a UBO for small metadata structs (std140), optionally persistently mapped.
unsigned ShaderTensor::createUBO(const std::string &name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                                 unsigned mapFlags, const void* initialData) {
    UBO u{}; u.binding = bindingIndex; u.size = sizeBytes; u.flags = mapFlags;
    glCreateBuffers(1, &u.id);
    glNamedBufferStorage(u.id, sizeBytes, initialData, mapFlags);
    if (mapFlags) u.map = glMapNamedBufferRange(u.id, 0, sizeBytes, mapFlags);
    glBindBufferBase(GL_UNIFORM_BUFFER, bindingIndex, u.id);
    ubos_[name] = u;
    return u.id;
}

// Update bytes in a named UBO (memcpy if persistently mapped).
void ShaderTensor::updateUBO(const std::string &name, const void* src, std::ptrdiff_t bytes, std::ptrdiff_t offset) {
    std::unordered_map<std::string, UBO>::iterator it = ubos_.find(name); if (it == ubos_.end()) { std::cerr << "[ShaderTensor] UBO not found: " << name << "\n"; return; }
    UBO &u = it->second; if (offset + bytes > u.size) { std::cerr << "[ShaderTensor] updateUBO out of range on " << name << "\n"; return; }
    if (u.map) std::memcpy(static_cast<char*>(u.map) + offset, src, (size_t)bytes);
    else glNamedBufferSubData(u.id, offset, bytes, src);
}

// Begin an async readback by copying an SSBO region into a staging buffer and fencing it.
ShaderTensor::ReadbackHandle ShaderTensor::startReadback(const std::string &srcName, std::ptrdiff_t bytes, std::ptrdiff_t srcOffset) {
    auto it = buffers_.find(srcName); if (it == buffers_.end()) { std::cerr << "[ShaderTensor] SSBO not found: " << srcName << "\n"; return {}; }
    Buffer &src = it->second; if (srcOffset + bytes > src.size) { std::cerr << "[ShaderTensor] startReadback out of range on " << srcName << "\n"; return {}; }

    RB rb{}; rb.size = bytes; rb.offset = srcOffset; rb.flags = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    glCreateBuffers(1, &rb.staging);
    glNamedBufferStorage(rb.staging, bytes, nullptr, rb.flags);
    rb.ptr = glMapNamedBufferRange(rb.staging, 0, bytes, rb.flags);

    glCopyNamedBufferSubData(src.id, rb.staging, srcOffset, 0, bytes);
    rb.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

    std::uint64_t key = ++nextRbId_;
    readbacks_[key] = rb;
    ShaderTensor::ReadbackHandle h; h.id = key; return h;
}

// Check if the async readback has completed without blocking.
bool ShaderTensor::pollReadback(ReadbackHandle h) {
    std::unordered_map<std::uint64_t, RB>::iterator it = readbacks_.find(h.id); if (it == readbacks_.end()) return false;
    RB &rb = it->second; if (!rb.fence) return false;
    GLenum r = glClientWaitSync((GLsync)rb.fence, 0, 0);
    return (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED);
}

// Block (with timeout ns) until the async readback completes.
void ShaderTensor::waitReadback(ReadbackHandle h, std::uint64_t timeoutNS) {
    std::unordered_map<std::uint64_t, RB>::iterator it = readbacks_.find(h.id); if (it == readbacks_.end()) return;
    RB &rb = it->second; if (!rb.fence) return;
    glClientWaitSync((GLsync)rb.fence, GL_SYNC_FLUSH_COMMANDS_BIT, timeoutNS);
}

// Get the CPU pointer to the staging buffer for a completed readback.
const void* ShaderTensor::mapReadback(ReadbackHandle h) {
    std::unordered_map<std::uint64_t, RB>::iterator it = readbacks_.find(h.id); if (it == readbacks_.end()) return nullptr;
    return it->second.ptr;
}

// Release resources associated with a readback handle.
void ShaderTensor::finishReadback(ReadbackHandle h) {
    std::unordered_map<std::uint64_t, RB>::iterator it = readbacks_.find(h.id); if (it == readbacks_.end()) return;
    destroyReadback(it->second);
    readbacks_.erase(it);
}

// Attach human-readable labels for debugging in GL debuggers.
void ShaderTensor::labelBuffer(const std::string &name, const char* label) {
#ifdef GL_KHR_debug
    auto itB = buffers_.find(name);
    if (itB != buffers_.end()) glObjectLabel(GL_BUFFER, itB->second.id, -1, label);
    auto itU = ubos_.find(name);
    if (itU != ubos_.end()) glObjectLabel(GL_BUFFER, itU->second.id, -1, label);
#else
    (void)name; (void)label;
#endif
}

// Label the GL program object.
void ShaderTensor::labelProgram(const char* label) {
#ifdef GL_KHR_debug
    if (program_) glObjectLabel(GL_PROGRAM, program_, -1, label);
#else
    (void)label;
#endif
}

// Fetch the underlying GL program object id.
unsigned ShaderTensor::program() const { return program_; }

// Query/cached lookup of a uniform location by name.
int ShaderTensor::loc(const std::string &name) {
    auto it = uniformCache_.find(name);
    if (it != uniformCache_.end()) return it->second;
    int l = glGetUniformLocation(program_, name.c_str());
    if (l == -1) {
        // not fatal: uniform may be optimized out or misspelled
        // std::cerr << "[ShaderTensor] uniform not found: " << name << "\n";
    }
    uniformCache_[name] = l;
    return l;
}

// Check a shader's compile status and print its info log.
bool ShaderTensor::checkShader(unsigned sh, const char* dbg) {
    int ok = GL_FALSE; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (ok == GL_TRUE) return true;
    int len = 0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
    std::vector<char> log((size_t)len);
    glGetShaderInfoLog(sh, len, nullptr, log.data());
    std::cerr << "[ShaderTensor] compute shader compile failed (" << (dbg?dbg:"") << ")\n" << log.data();
    return false;
}

// Check a program's link status and print its info log.
bool ShaderTensor::checkProgram(unsigned prog, const char* dbg) {
    int ok = GL_FALSE; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (ok == GL_TRUE) return true;
    int len = 0; glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
    std::vector<char> log((size_t)len);
    glGetProgramInfoLog(prog, len, nullptr, log.data());
    std::cerr << "[ShaderTensor] program link failed (" << (dbg?dbg:"") << ")\n" << log.data();
    return false;
}

// Destroy a readback staging buffer and its sync fence.
void ShaderTensor::destroyReadback(RB &rb) {
    if (rb.fence) { glDeleteSync((GLsync)rb.fence); rb.fence = nullptr; }
    if (rb.staging) {
        if (rb.ptr) { glUnmapNamedBuffer(rb.staging); rb.ptr = nullptr; }
        glDeleteBuffers(1, &rb.staging); rb.staging = 0;
    }
}
