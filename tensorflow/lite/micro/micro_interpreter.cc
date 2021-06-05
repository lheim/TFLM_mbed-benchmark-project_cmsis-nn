/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_optional_debug_tools.h"


#ifdef BENCHMARK_LAYERS
  #include "benchmark.h"
  Benchmark benchmark_layers;
  #ifdef ENERGY_MEASUREMENT
    DigitalOut layer_gpio(D1);
  #endif
#endif // BENCHMARK_LAYERS

namespace tflite {
namespace {

const char* OpNameFromRegistration(const TfLiteRegistration* registration) {
  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    return registration->custom_name;
  } else {
    return EnumNameBuiltinOperator(BuiltinOperator(registration->builtin_code));
  }
}

}  // namespace

namespace internal {

TfLiteStatus ContextHelper::AllocatePersistentBuffer(TfLiteContext* ctx,
                                                     size_t bytes, void** ptr) {
  return reinterpret_cast<ContextHelper*>(ctx->impl_)
      ->allocator_->AllocatePersistentBuffer(bytes, ptr);
}

TfLiteStatus ContextHelper::RequestScratchBufferInArena(TfLiteContext* ctx,
                                                        size_t bytes,
                                                        int* buffer_idx) {
  ContextHelper* helper = reinterpret_cast<ContextHelper*>(ctx->impl_);
  return helper->allocator_->RequestScratchBufferInArena(
      helper->current_node_idx_, bytes, buffer_idx);
}

void* ContextHelper::GetScratchBuffer(TfLiteContext* ctx, int buffer_idx) {
  return reinterpret_cast<ContextHelper*>(ctx->impl_)
      ->allocator_->GetScratchBuffer(buffer_idx);
}

void ContextHelper::ReportOpError(struct TfLiteContext* context,
                                  const char* format, ...) {
  ContextHelper* helper = static_cast<ContextHelper*>(context->impl_);
  va_list args;
  va_start(args, format);
  TF_LITE_REPORT_ERROR(helper->error_reporter_, format, args);
  va_end(args);
}

}  // namespace internal

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const OpResolver& op_resolver,
                                   uint8_t* tensor_arena,
                                   size_t tensor_arena_size,
                                   ErrorReporter* error_reporter)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter),
      allocator_(&context_, model_, tensor_arena, tensor_arena_size,
                 error_reporter_),
      tensors_allocated_(false),
      context_helper_(error_reporter_, &allocator_) {
  const flatbuffers::Vector<flatbuffers::Offset<SubGraph>>* subgraphs =
      model->subgraphs();
  if (subgraphs->size() != 1) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Only 1 subgraph is currently supported.\n");
    initialization_status_ = kTfLiteError;
    return;
  }
  subgraph_ = (*subgraphs)[0];
  tensors_ = subgraph_->tensors();
  operators_ = subgraph_->operators();

  context_.impl_ = static_cast<void*>(&context_helper_);
  context_.ReportError = context_helper_.ReportOpError;
  context_.recommended_num_threads = 1;

  // If the system is big endian then convert weights from the flatbuffer from
  // little to big endian on startup so that it does not need to be done during
  // inference.
  // NOTE: This requires that the flatbuffer is held in memory which can be
  // modified by this process.
  if (!FLATBUFFERS_LITTLEENDIAN) {
    for (size_t t = 0; t < tensors_size(); ++t) {
      TfLiteTensor* thisTensor = &context_.tensors[t];
      if (thisTensor->allocation_type == kTfLiteMmapRo)
        CorrectTensorEndianness(thisTensor);
    }
  }

  initialization_status_ = kTfLiteOk;

  #ifdef BENCHMARK_LAYERS
      TF_LITE_REPORT_ERROR(error_reporter_, "\n");
      #ifdef ENERGY_MEASUREMENT
        layer_gpio = 0;
      #endif
  #endif // BENCHMARK_LAYERS

}

MicroInterpreter::~MicroInterpreter() {
  if (node_and_registrations_ != nullptr) {
    for (size_t i = 0; i < operators_->size(); ++i) {
      TfLiteNode* node = &(node_and_registrations_[i].node);
      const TfLiteRegistration* registration =
          node_and_registrations_[i].registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->free != nullptr) {
        registration->free(&context_, node->user_data);
      }
    }
  }
}

void MicroInterpreter::CorrectTensorEndianness(TfLiteTensor* tensorCorr) {
  int32_t tensorSize = 1;
  for (int d = 0; d < tensorCorr->dims->size; ++d)
    tensorSize *= reinterpret_cast<const int32_t*>(tensorCorr->dims->data)[d];

  switch (tensorCorr->type) {
    case TfLiteType::kTfLiteFloat32:
      CorrectTensorDataEndianness(tensorCorr->data.f, tensorSize);
      break;
    case TfLiteType::kTfLiteFloat16:
      CorrectTensorDataEndianness(tensorCorr->data.f16, tensorSize);
      break;
    case TfLiteType::kTfLiteInt64:
      CorrectTensorDataEndianness(tensorCorr->data.i64, tensorSize);
      break;
    case TfLiteType::kTfLiteInt32:
      CorrectTensorDataEndianness(tensorCorr->data.i32, tensorSize);
      break;
    case TfLiteType::kTfLiteInt16:
      CorrectTensorDataEndianness(tensorCorr->data.i16, tensorSize);
      break;
    case TfLiteType::kTfLiteComplex64:
      CorrectTensorDataEndianness(tensorCorr->data.c64, tensorSize);
      break;
    default:
      // Do nothing for other data types.
      break;
  }
}

template <class T>
void MicroInterpreter::CorrectTensorDataEndianness(T* data, int32_t size) {
  for (int32_t i = 0; i < size; ++i) {
    data[i] = flatbuffers::EndianScalar(data[i]);
  }
}

TfLiteStatus MicroInterpreter::AllocateTensors() {
  TF_LITE_ENSURE_OK(&context_, allocator_.AllocateNodeAndRegistrations(
                                   op_resolver_, &node_and_registrations_));

  // Only allow AllocatePersistentBuffer in Init stage.
  context_.AllocatePersistentBuffer = context_helper_.AllocatePersistentBuffer;
  context_.RequestScratchBufferInArena = nullptr;
  context_.GetScratchBuffer = nullptr;

  for (size_t i = 0; i < operators_->size(); ++i) {
    context_helper_.SetNodeIndex(i);
    auto* node = &(node_and_registrations_[i].node);
    auto* registration = node_and_registrations_[i].registration;
    size_t init_data_size;
    const char* init_data;
    if (registration->builtin_code == BuiltinOperator_CUSTOM) {
      init_data = reinterpret_cast<const char*>(node->custom_initial_data);
      init_data_size = node->custom_initial_data_size;
    } else {
      init_data = reinterpret_cast<const char*>(node->builtin_data);
      init_data_size = 0;
    }
    if (registration->init) {
      node->user_data =
          registration->init(&context_, init_data, init_data_size);
    }
  }
  context_helper_.SetNodeIndex(-1);

  // Both AllocatePersistentBuffer and RequestScratchBufferInArena is available
  // in Prepare stage.
  context_.RequestScratchBufferInArena =
      context_helper_.RequestScratchBufferInArena;
  for (size_t i = 0; i < operators_->size(); ++i) {
    // Set node idx to annotate the lifetime for scratch buffers.
    context_helper_.SetNodeIndex(i);
    auto* node = &(node_and_registrations_[i].node);
    auto* registration = node_and_registrations_[i].registration;
    if (registration->prepare) {
      TfLiteStatus prepare_status = registration->prepare(&context_, node);
      if (prepare_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Node %s (number %df) failed to prepare with status %d",
            OpNameFromRegistration(registration), i, prepare_status);
        return kTfLiteError;
      }
    }
  }
  context_helper_.SetNodeIndex(-1);

  // Prepare is done, we're ready for Invoke. Memory allocation is no longer
  // allowed. Kernels can only fetch scratch buffers via GetScratchBuffer.
  context_.AllocatePersistentBuffer = nullptr;
  context_.RequestScratchBufferInArena = nullptr;
  context_.GetScratchBuffer = context_helper_.GetScratchBuffer;

  TF_LITE_ENSURE_OK(&context_, allocator_.FinishTensorAllocation());
  tensors_allocated_ = true;
  return kTfLiteOk;
}
TfLiteStatus MicroInterpreter::Invoke() {
  if (initialization_status_ != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Invoke() called after initialization failed\n");
    return kTfLiteError;
  }

  #ifdef BENCHMARK_LAYERS
      benchmark_layers.init();
  #endif // BENCHMARK_LAYERS



  // Ensure tensors are allocated before the interpreter is invoked to avoid
  // difficult to debug segfaults.
  if (!tensors_allocated_) {
    AllocateTensors();
  }

  for (size_t i = 0; i < operators_->size(); ++i) {
    #ifdef BENCHMARK_LAYERS
      benchmark_layers.start();
      #ifdef ENERGY_MEASUREMENT
        layer_gpio = !layer_gpio;
      #endif // ENERGY_MEASUREMENT
    #endif // BENCHMARK_LAYERS

    auto* node = &(node_and_registrations_[i].node);
    auto* registration = node_and_registrations_[i].registration;

    if (registration->invoke) {
      TfLiteStatus invoke_status = registration->invoke(&context_, node);

      #ifdef BENCHMARK_LAYERS
        // #ifdef ENERGY_MEASUREMENT
        //    layer_gpio = 0;
        // #endif // ENERGY_MEASUREMENT
        benchmark_layers.stop();
        #ifndef NO_REPORTING
          TF_LITE_REPORT_ERROR(error_reporter_, "\t\tLayer_%d_%s\n\t\t%u", i, OpNameFromRegistration(registration), benchmark_layers.read());
        #endif
        // #ifdef ENERGY_MEASUREMENT
        //   wait_us(500000);
        // #endif // ENERGY_MEASUREMENT
        benchmark_layers.clear();
      #endif // BENCHMARK_LAYERS

      if (invoke_status == kTfLiteError) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Node %s (number %d) failed to invoke with status %d",
            OpNameFromRegistration(registration), i, invoke_status);
        return kTfLiteError;
      } else if (invoke_status != kTfLiteOk) {
        return invoke_status;
      }
    }
  }
  #ifdef BENCHMARK_LAYERS
    #ifdef ENERGY_MEASUREMENT
     layer_gpio = 0;
    #endif // ENERGY_MEASUREMENT
  #endif // BENCHMARK_LAYERS
  return kTfLiteOk;
}

TfLiteTensor* MicroInterpreter::input(size_t index) {
  const flatbuffers::Vector<int32_t>* inputs = subgraph_->inputs();
  const size_t length = inputs->size();
  if ((index < 0) || (index >= length)) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Input index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  return &(context_.tensors[inputs->Get(index)]);
}

TfLiteTensor* MicroInterpreter::output(size_t index) {
  const flatbuffers::Vector<int32_t>* outputs = subgraph_->outputs();
  const size_t length = outputs->size();
  if ((index < 0) || (index >= outputs->size())) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Output index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  return &(context_.tensors[outputs->Get(index)]);
}

TfLiteTensor* MicroInterpreter::tensor(size_t index) {
  const size_t length = tensors_size();
  if ((index < 0) || (index >= tensors_size())) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Tensor index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  return &context_.tensors[index];
}

TfLiteStatus MicroInterpreter::ResetVariableTensors() {
  const size_t length = tensors_size();
  for (size_t i = 0; i < length; ++i) {
    TfLiteTensor* cur_tensor = tensor(i);
    if (cur_tensor->is_variable) {
      TfLiteStatus status = tflite::ResetVariableTensor(cur_tensor);
      if (status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Failed to reset variable tensor at index: %d", i);
        return status;
      }
    }
  }
  return kTfLiteOk;
}

}  // namespace tflite
