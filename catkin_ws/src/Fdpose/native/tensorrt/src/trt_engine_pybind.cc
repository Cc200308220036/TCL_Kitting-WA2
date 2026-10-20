#include <NvInfer.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << msg << std::endl;
    }
};

Logger logger;

class TensorRTEngine {
public:
    TensorRTEngine(int inputs_num, int outputs_num)
        : inputs_num_(inputs_num), outputs_num_(outputs_num) {
        TORCH_CHECK(inputs_num == 2 && (outputs_num == 1 || outputs_num == 2),
                    "FDPose expects 2 inputs and 1 or 2 outputs");
    }
    bool loadEngineModel(const std::string& eng_path);
    bool inferEngineModel(const std::vector<at::Tensor>& inputs,
                          const std::vector<at::Tensor>& outputs);

private:
    // 声明顺序保证析构时先释放 context，再释放 engine 和 runtime。
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    int inputs_num_;
    int outputs_num_;
    int device_ = -1;
    std::vector<std::string> names_;

    static void checkCuda(cudaError_t result, const char* operation) {
        TORCH_CHECK(result == cudaSuccess, operation, ": ", cudaGetErrorString(result));
    }
};

bool TensorRTEngine::loadEngineModel(const std::string& eng_path) {
    // 重载失败后不能继续使用上一次的 context 或 tensor 名称。
    context_.reset();
    engine_.reset();
    runtime_.reset();
    names_.clear();
    std::ifstream file(eng_path, std::ios::binary | std::ios::ate);
    TORCH_CHECK(file.good(), "Cannot open TensorRT engine: ", eng_path);
    const auto length = file.tellg();
    TORCH_CHECK(length > 0, "Empty/unreadable TensorRT engine: ", eng_path);
    std::vector<char> data(static_cast<size_t>(length));
    file.seekg(0);
    TORCH_CHECK(static_cast<bool>(file.read(data.data(), data.size())),
                "Incomplete TensorRT engine read: ", eng_path);

    checkCuda(cudaGetDevice(&device_), "cudaGetDevice");
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    TORCH_CHECK(runtime_, "Cannot create TensorRT runtime: ", eng_path);
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    TORCH_CHECK(engine_, "Cannot deserialize TensorRT engine: ", eng_path,
                "; rebuild with tools/build/prepare_tensorrt.py");

    // 按 ONNX 名称绑定，不能假设 TensorRT 总是先枚举输入再枚举输出。
    names_ = {"A", "B"};
    if (outputs_num_ == 1) names_.push_back("score_logit");
    else { names_.push_back("trans"); names_.push_back("rot"); }
    TORCH_CHECK(engine_->getNbIOTensors() == inputs_num_ + outputs_num_,
                "Unexpected TensorRT I/O count: ", eng_path);
    for (size_t i = 0; i < names_.size(); ++i) {
        const char* name = names_[i].c_str();
        const auto mode = i < static_cast<size_t>(inputs_num_)
            ? nvinfer1::TensorIOMode::kINPUT : nvinfer1::TensorIOMode::kOUTPUT;
        TORCH_CHECK(engine_->getTensorIOMode(name) == mode, "Unexpected I/O name/mode: ", name);
        TORCH_CHECK(engine_->getTensorDataType(name) == nvinfer1::DataType::kFLOAT,
                    "Expected float32 I/O: ", name);
        TORCH_CHECK(engine_->getTensorLocation(name) == nvinfer1::TensorLocation::kDEVICE
                    && engine_->getTensorFormat(name) == nvinfer1::TensorFormat::kLINEAR,
                    "Expected linear CUDA tensor: ", name);
        const auto dims = engine_->getTensorShape(name);
        TORCH_CHECK(dims.nbDims > 0, "Invalid tensor shape: ", name);
        for (int d = 0; d < dims.nbDims; ++d)
            TORCH_CHECK(dims.d[d] > 0, "Only fixed-shape FDPose engines are supported: ", name);
    }
    context_.reset(engine_->createExecutionContext());
    TORCH_CHECK(context_, "Cannot create TensorRT execution context: ", eng_path);
    return true;
}

bool TensorRTEngine::inferEngineModel(const std::vector<at::Tensor>& inputs,
                                    const std::vector<at::Tensor>& outputs) {
    TORCH_CHECK(context_, "TensorRT engine is not loaded");
    TORCH_CHECK(inputs.size() == static_cast<size_t>(inputs_num_)
                && outputs.size() == static_cast<size_t>(outputs_num_), "Wrong TensorRT I/O count");
    int current_device;
    checkCuda(cudaGetDevice(&current_device), "cudaGetDevice");
    TORCH_CHECK(current_device == device_, "TensorRT engine used on a different CUDA device");
    for (size_t i = 0; i < names_.size(); ++i) {
        const auto& tensor = i < inputs.size() ? inputs[i] : outputs[i - inputs.size()];
        const char* name = names_[i].c_str();
        TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.get_device() == device_,
                    "Expected tensor on engine CUDA device: ", name);
        TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.is_contiguous(),
                    "Expected contiguous float32 tensor: ", name);
        const auto dims = engine_->getTensorShape(name);
        TORCH_CHECK(tensor.dim() == dims.nbDims, "Wrong tensor rank: ", name);
        for (int d = 0; d < dims.nbDims; ++d)
            TORCH_CHECK(tensor.size(d) == dims.d[d], "Wrong tensor shape: ", name,
                        ", dimension ", d, " expected ", dims.d[d], " got ", tensor.size(d));
        TORCH_CHECK(context_->setTensorAddress(name, tensor.data_ptr()), "Cannot bind tensor: ", name);
    }
    // 与 PyTorch 输入生产使用同一流；保留同步返回语义，错误直接传到 Python。
    const auto stream = c10::cuda::getCurrentCUDAStream(device_).stream();
    const bool submitted = context_->enqueueV3(stream);
    checkCuda(cudaStreamSynchronize(stream), "TensorRT cudaStreamSynchronize");
    TORCH_CHECK(submitted, "TensorRT enqueueV3 failed");
    return true;
}

PYBIND11_MODULE(pyTRTEng, m) {
    pybind11::class_<TensorRTEngine>(m, "TensorRTEngine")
        .def(pybind11::init<int, int>())
        .def("loadEngineModel", &TensorRTEngine::loadEngineModel)
        .def("inferEngineModel", &TensorRTEngine::inferEngineModel);
}
