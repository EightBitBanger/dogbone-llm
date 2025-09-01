#include "ShaderTensor.h"
#include "Tensor2D.h"
#include <iostream>
#include <cstring>

#ifdef GL_KHR_debug
static void LabelProgram(GLuint prog, const char* label) {
    if (label && *label) glObjectLabel(GL_PROGRAM, prog, -1, label);
}
#else
static void LabelProgram(GLuint, const char*) {}
#endif

ShaderTensor::ShaderTensor() {}
ShaderTensor::~ShaderTensor() {
    for (auto& kv : buffers_) {
        if (kv.second.map) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, kv.second.id);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
        glDeleteBuffers(1, &kv.second.id);
    }
    for (auto& kv : ubos_) {
        if (kv.second.map) {
            glBindBuffer(GL_UNIFORM_BUFFER, kv.second.id);
            glUnmapBuffer(GL_UNIFORM_BUFFER);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
        }
        glDeleteBuffers(1, &kv.second.id);
    }
    if (program_) glDeleteProgram(program_);
    for (auto& kv : programs_) glDeleteProgram(kv.second);
}

static GLuint BuildCompute(const char* src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = GL_FALSE; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len=0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        std::string log; log.resize((size_t)std::max(0, len));
        if (len>0) { GLsizei out=0; glGetShaderInfoLog(sh, len, &out, &log[0]); std::cerr << "[ShaderTensor] CS compile error:\n" << log << "\n"; }
        glDeleteShader(sh);
        return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    glDeleteShader(sh);
    GLint linked = GL_FALSE; glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint len=0; glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log; log.resize((size_t)std::max(0, len));
        if (len>0) { GLsizei out=0; glGetProgramInfoLog(prog, len, &out, &log[0]); std::cerr << "[ShaderTensor] Program link error:\n" << log << "\n"; }
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

bool ShaderTensor::buildComputeFromSource(const char* src, const char* debugLabel) {
    GLuint prog = BuildCompute(src);
    if (!prog) return false;
    if (program_) glDeleteProgram(program_);
    program_ = prog;
    LabelProgram(program_, debugLabel);
    uniformCache_.clear();
    return true;
}

bool ShaderTensor::buildComputeFromSourceNamed(const std::string& key, const char* src, const char* debugLabel) {
    GLuint prog = BuildCompute(src);
    if (!prog) return false;
    auto it = programs_.find(key);
    if (it != programs_.end()) {
        glDeleteProgram(it->second);
        it->second = prog;
    } else {
        programs_[key] = prog;
    }
    LabelProgram(prog, debugLabel);
    // Keep program_ pointing at the last used one for compatibility
    program_ = prog;
    uniformCache_.clear();
    return true;
}

void ShaderTensor::use() const { glUseProgram(program_); }
void ShaderTensor::useNamed(const std::string& key) const {
    auto it = programs_.find(key);
    if (it != programs_.end()) {
        glUseProgram(it->second);
    } else {
        // fall back to last program_ if key missing
        glUseProgram(program_);
    }
}

void ShaderTensor::dispatch(unsigned gx, unsigned gy, unsigned gz) const {
    glDispatchCompute(gx, gy, gz);
}
static GLint Loc(GLuint prog, std::unordered_map<std::string,int>& cache, const std::string& name) {
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    GLint loc = glGetUniformLocation(prog, name.c_str());
    cache[name] = loc;
    return loc;
}
void ShaderTensor::setInt(const std::string& name, int v)   { GLint loc = Loc(program_, uniformCache_, name); if (loc>=0) glUniform1i(loc, v); }
void ShaderTensor::setUInt(const std::string& name, unsigned v){ GLint loc = Loc(program_, uniformCache_, name); if (loc>=0) glUniform1ui(loc, v); }
void ShaderTensor::setFloat(const std::string& name, float v){ GLint loc = Loc(program_, uniformCache_, name); if (loc>=0) glUniform1f(loc, v); }
void ShaderTensor::setIVec2(const std::string& name, int x, int y){ GLint loc = Loc(program_, uniformCache_, name); if (loc>=0) glUniform2i(loc, x, y); }


void ShaderTensor::uploadRawSSBO(unsigned id, std::ptrdiff_t offset, const void* data, std::ptrdiff_t size) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, size, data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}
unsigned ShaderTensor::createSSBO(const std::string& name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                                  unsigned mapFlags, const void* initialData) {
    // Clean up existing buffer with the same name to avoid leaks
    auto it = buffers_.find(name);
    if (it != buffers_.end()) {
        if (it->second.map) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, it->second.id);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
        glDeleteBuffers(1, &it->second.id);
        buffers_.erase(it);
    }
    Buffer b; b.binding = bindingIndex; b.size = sizeBytes; b.flags = mapFlags;
    glGenBuffers(1, &b.id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeBytes, initialData, GL_DYNAMIC_DRAW);
    if (mapFlags & GL_MAP_WRITE_BIT) {
        GLbitfield acc = mapFlags & ~(GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
        b.map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeBytes, acc);
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingIndex, b.id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    buffers_[name] = b;
    return b.id;
}

unsigned ShaderTensor::ensureSSBO(const std::string& name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                                  unsigned mapFlags, const void* initialData) {
    auto it = buffers_.find(name);
    if (it != buffers_.end()) {
        Buffer& b = it->second;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, b.id);
        if (b.map) {
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            b.map = nullptr;
        }
        if (b.size != sizeBytes) {
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeBytes, initialData, GL_DYNAMIC_DRAW);
            b.size = sizeBytes;
        } else if (initialData) {
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeBytes, initialData);
        }
        b.binding = bindingIndex;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, b.binding, b.id);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return b.id;
    }
    return createSSBO(name, sizeBytes, bindingIndex, mapFlags, initialData);
}

void ShaderTensor::adoptSSBO(const std::string& name, unsigned id, std::ptrdiff_t sizeBytes, unsigned bindingIndex) {
    Buffer b; b.id=id; b.size=sizeBytes; b.binding=bindingIndex; b.flags=0; b.map=nullptr;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingIndex, id);
    buffers_[name] = b;
}

void ShaderTensor::upload(const std::string& name, const void* src, std::ptrdiff_t bytes, std::ptrdiff_t offset) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) { std::cerr << "[ShaderTensor] upload: missing SSBO " << name << "\n"; return; }
    Buffer& b = it->second;
    if (offset + bytes > b.size) { std::cerr << "[ShaderTensor] upload: OOB on " << name << "\n"; return; }
    if (b.map) {
        std::memcpy(static_cast<char*>(b.map) + offset, src, (size_t)bytes);
    } else {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, b.id);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, bytes, src);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
}

void ShaderTensor::downloadSync(const std::string& name, void* dst, std::ptrdiff_t bytes, std::ptrdiff_t offset) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) { std::cerr << "[ShaderTensor] downloadSync: missing SSBO " << name << "\n"; return; }
    Buffer& b = it->second;
    if (offset + bytes > b.size) { std::cerr << "[ShaderTensor] downloadSync: OOB on " << name << "\n"; return; }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b.id);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, bytes, dst);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ShaderTensor::rebindSSBO(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) return;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, it->second.binding, it->second.id);
}

unsigned ShaderTensor::createUBO(const std::string& name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                                 unsigned mapFlags, const void* initialData) {
    // Clean up existing UBO with the same name to avoid leaks
    auto it = ubos_.find(name);
    if (it != ubos_.end()) {
        if (it->second.map) {
            glBindBuffer(GL_UNIFORM_BUFFER, it->second.id);
            glUnmapBuffer(GL_UNIFORM_BUFFER);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
        }
        glDeleteBuffers(1, &it->second.id);
        ubos_.erase(it);
    }
    UBO u; u.binding = bindingIndex; u.size = sizeBytes; u.flags = mapFlags;
    glGenBuffers(1, &u.id);
    glBindBuffer(GL_UNIFORM_BUFFER, u.id);
    glBufferData(GL_UNIFORM_BUFFER, sizeBytes, initialData, GL_DYNAMIC_DRAW);
    if (mapFlags & GL_MAP_WRITE_BIT) {
        GLbitfield acc = mapFlags & ~(GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
        u.map = glMapBufferRange(GL_UNIFORM_BUFFER, 0, sizeBytes, acc);
    }
    glBindBufferBase(GL_UNIFORM_BUFFER, bindingIndex, u.id);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    ubos_[name] = u;
    return u.id;
}

unsigned ShaderTensor::ensureUBO(const std::string& name, std::ptrdiff_t sizeBytes, unsigned bindingIndex,
                                 unsigned mapFlags, const void* initialData) {
    auto it = ubos_.find(name);
    if (it != ubos_.end()) {
        UBO& u = it->second;
        glBindBuffer(GL_UNIFORM_BUFFER, u.id);
        if (u.map) {
            glUnmapBuffer(GL_UNIFORM_BUFFER);
            u.map = nullptr;
        }
        if (u.size != sizeBytes) {
            glBufferData(GL_UNIFORM_BUFFER, sizeBytes, initialData, GL_DYNAMIC_DRAW);
            u.size = sizeBytes;
        } else if (initialData) {
            glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeBytes, initialData);
        }
        u.binding = bindingIndex;
        glBindBufferBase(GL_UNIFORM_BUFFER, u.binding, u.id);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        return u.id;
    }
    return createUBO(name, sizeBytes, bindingIndex, mapFlags, initialData);
}

void ShaderTensor::updateUBO(const std::string& name, const void* src, std::ptrdiff_t bytes, std::ptrdiff_t offset) {
    auto it = ubos_.find(name);
    if (it == ubos_.end()) { std::cerr << "[ShaderTensor] updateUBO: missing UBO " << name << "\n"; return; }
    UBO& u = it->second;
    if (offset + bytes > u.size) { std::cerr << "[ShaderTensor] updateUBO: OOB on " << name << "\n"; return; }
    if (u.map) {
        std::memcpy(static_cast<char*>(u.map) + offset, src, (size_t)bytes);
    } else {
        glBindBuffer(GL_UNIFORM_BUFFER, u.id);
        glBufferSubData(GL_UNIFORM_BUFFER, offset, bytes, src);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
}

ShaderTensor::ReadbackHandle ShaderTensor::startReadback(const std::string& srcName, std::ptrdiff_t bytes, std::ptrdiff_t srcOffset) {
    auto it = buffers_.find(srcName);
    if (it == buffers_.end()) { std::cerr << "[ShaderTensor] startReadback: missing SSBO " << srcName << "\n"; return {}; }
    Buffer& src = it->second;
    RB rb{};
    glGenBuffers(1, &rb.staging);
    glBindBuffer(GL_COPY_READ_BUFFER, src.id);
    glBindBuffer(GL_COPY_WRITE_BUFFER, rb.staging);
    glBufferData(GL_COPY_WRITE_BUFFER, bytes, nullptr, GL_STREAM_READ);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, srcOffset, 0, bytes);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    rb.size = bytes; rb.offset = 0;
    rb.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    std::uint64_t id = ++nextRbId_;
    readbacks_[id] = rb;
    ShaderTensor::ReadbackHandle handle;
    handle.id = id;
    return handle;
}

bool ShaderTensor::pollReadback(ReadbackHandle h) {
    auto it = readbacks_.find(h.id);
    if (it == readbacks_.end()) return false;
    GLsync f = it->second.fence;
    if (!f) return true;
    GLint res = 0;
    glGetSynciv(f, GL_SYNC_STATUS, sizeof(res), nullptr, &res);
    return res == GL_SIGNALED;
}

bool ShaderTensor::waitReadback(ReadbackHandle h, std::uint64_t timeoutNanoseconds) {
    auto it = readbacks_.find(h.id);
    if (it == readbacks_.end()) return false;
    GLsync f = it->second.fence;
    if (!f) return true;
    GLenum r = glClientWaitSync(f, GL_SYNC_FLUSH_COMMANDS_BIT, timeoutNanoseconds);
    return r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED;
}

const void* ShaderTensor::mapReadback(ReadbackHandle h) {
    auto it = readbacks_.find(h.id);
    if (it == readbacks_.end()) return nullptr;
    auto& rb = it->second;
    if (rb.ptr) return rb.ptr;
    glBindBuffer(GL_COPY_WRITE_BUFFER, rb.staging);
    rb.ptr = glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, rb.size, GL_MAP_READ_BIT);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    return rb.ptr;
}

void ShaderTensor::finishReadback(ReadbackHandle h) {
    auto it = readbacks_.find(h.id);
    if (it == readbacks_.end()) return;
    auto& rb = it->second;
    if (rb.ptr) {
        glBindBuffer(GL_COPY_WRITE_BUFFER, rb.staging);
        glUnmapBuffer(GL_COPY_WRITE_BUFFER);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        rb.ptr = nullptr;
    }
    if (rb.fence) glDeleteSync(rb.fence);
    if (rb.staging) { glDeleteBuffers(1, &rb.staging); }
    readbacks_.erase(it);
}

void ShaderTensor::sync() {
    // make SSBO writes visible without stalling the whole GPU
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}
