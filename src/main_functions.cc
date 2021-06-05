/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef INPUT_LENGTH
  #define INPUT_LENGTH 1024
#endif
#define INPUT_TYPE tkTFLiteFloat32

#ifndef OUTPUT_LENGTH
  #define OUTPUT_LENGTH 10
#endif
#define OUTPUT_TYPE tkTFLiteFloat32

#ifdef CYCLES
  char BENCHMARK_UNIT[] = "cycles";
#else
  char BENCHMARK_UNIT[] = "us";
#endif


#include "main_functions.h"

#include "constants.h"
#include "output_handler.h"
#include "model_data.h"
#include "benchmark.h"

#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "mbed.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

int input_length = 1;
int output_length = 1;

int input_dim;
int output_dim;

float output_arr[OUTPUT_LENGTH];
//int probability_beforedec;
//int probability_afterdec;

Benchmark benchmark_inference;

// This variable is volatile to prevent the compiler from optimizing.
// The implemented blocking reading function was not able to continue
// using optimization levels starting from -O1.
volatile int uart_status = -1;

float buffer_image[INPUT_LENGTH];

#ifndef ENERGY_MEASUREMENT
  DigitalOut inference_led(LED1);
  DigitalOut input_led(LED2);
#else
  DigitalOut inference_gpio(D0);
  DigitalOut layer_gpio(D1); // also defined in micro_interpreter.cc
  DigitalOut input_gpio(D2);
  DigitalOut tbd1_gpio(D3);
  DigitalOut tbd2_gpio(D4);
#endif


}  // namespace


inline void print_tflitetype(int type)
{
  switch(type)
  {
    case 0:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTFLiteNoType");
      break;
    case 1:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTFLiteFloat32");
      break;
    case 2:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTfLiteInt32");
      break;
    case 3:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTfLiteUInt8");
      break;
    case 4:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTfLiteInt64");
      break;
    case 5:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTfLiteString");
      break;
    case 6:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTfLiteBool");
      break;
    case 7:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTfLiteInt16");
      break;
    case 8:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTfLiteComplex64");
      break;
    case 9:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTfLiteInt8");
      break;
    case 10:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTfLiteFloat16");
      break;
    case 11:
      TF_LITE_REPORT_ERROR(error_reporter,"datatype:\n\tkTfLiteFloat64");
      break;
  }
}


inline void gather_model_information()
{
  input_dim = input->dims->size;
  output_dim = output->dims->size;

    //   TF_LITE_REPORT_ERROR(error_reporter, "+++++++++++++++");
    //   TF_LITE_REPORT_ERROR(error_reporter, "Model information");
    //   TF_LITE_REPORT_ERROR(error_reporter, "_____________");
  TF_LITE_REPORT_ERROR(error_reporter,"model name/s:\n\t%s", model_name);

  TF_LITE_REPORT_ERROR(error_reporter,"# dimension/s:\n\t%d", input->dims->size);
  print_tflitetype(input->type);
  for(int i = 0; i < input_dim; i++)
  {
    input_length *= input->dims->data[i];
    TF_LITE_REPORT_ERROR(error_reporter, "%d. input dimension:\n\t%d", i, input->dims->data[i]);
  }
  TF_LITE_REPORT_ERROR(error_reporter, "total input length:\n\t%d", input_length);


  TF_LITE_REPORT_ERROR(error_reporter, "\n\noutput length:\n\t%d", output_length);
  TF_LITE_REPORT_ERROR(error_reporter,"# dimension/s:\n\t%d", output->dims->size);
  print_tflitetype(output->type);
  for(int i = 0; i < output_dim; i++)
  {
    output_length *= output->dims->data[i];
    TF_LITE_REPORT_ERROR(error_reporter, "%d. output dimension:\n\t%d", i, output->dims->data[i]);
  }
  TF_LITE_REPORT_ERROR(error_reporter, "total output length:\n\t%d", output_length);
  
  TF_LITE_REPORT_ERROR(error_reporter, "\n\nbenchmark unit:\n\t%s", BENCHMARK_UNIT);

    //   TF_LITE_REPORT_ERROR(error_reporter, "+++++++++++++++");
    //   TF_LITE_REPORT_ERROR(error_reporter, "Starting inference ...");

}


// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  // TODO Only pull the operations which are necessary for the given model.
  static tflite::ops::micro::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;


  gather_model_information();
  #ifndef ENERGY_MEASUREMENT
    inference_led = 0;
    input_led = 0;
  #else
    inference_gpio = 0;
    input_gpio = 0;
    layer_gpio = 0;
    tbd1_gpio = 0;
    tbd2_gpio = 0;
  #endif

  #ifndef BENCHMARK_LAYERS
    benchmark_inference.init();
  #endif
}


void read_event(int event) {
    //TF_LITE_REPORT_ERROR(error_reporter, "The read callback was just triggered.");
    uart_status = 1;
    #ifndef ENERGY_MEASUREMENT
      input_led = 0;
    #else
      input_gpio = 0;
    #endif
}

// The name of this function is important for Arduino compatibility.
void loop() {
    #ifndef NO_MANUAL_INPUT
        if(inference_count == 0)
        {   
            #ifndef ENERGY_MEASUREMENT
              input_led = 1;
            #else
              input_gpio = 1;
            #endif

            // for some reason this target only supports the blocking read function with two arugments
            #ifdef TARGET_STM32F469
              pc.read((uint8_t*) buffer_image, INPUT_LENGTH * sizeof(float));
              #ifndef ENERGY_MEASUREMENT
                input_led = 0;
              #else 
                input_gpio = 0;
              #endif
            #else
              pc.read((uint8_t*) buffer_image, INPUT_LENGTH * sizeof(float), read_event);

              while(uart_status != 1)
              {
                  // block as long we didn't read anything
              }
              uart_status = 0;
            #endif
        }
    #endif

    #ifndef ENERGY_MEASUREMENT
      inference_led = 1;
    #else
      inference_gpio = 1;
    #endif

    // Place our just received image into the buffer
    // TODO: Make this dynamic for the input format. At the moment this only works for float32.
   
   for(int i = 0; i < input_length; i++)
    {
        #ifndef NO_MANUAL_INPUT
            input->data.f[i] = buffer_image[i];
        #else
            input->data.f[i] = input_example[i];
        #endif

    }


  #ifndef BENCHMARK_LAYERS
    if(inference_count == 0)
    {
        benchmark_inference.start();
    }
  #endif

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    return;
  }
  

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle){
    
    
    #ifndef BENCHMARK_LAYERS
      benchmark_inference.stop();
    #endif
    
    #ifndef ENERGY_MEASUREMENT
      inference_led = 0;
    #else
      inference_gpio = 0;
    #endif
    #ifndef NO_REPORTING
      pc.printf("_start_report_\n\n");
      #ifndef BENCHMARK_LAYERS
        pc.printf("Number of inferences\n%d\n", inference_count);
        pc.printf("Duration of inferences in %s\n%lu\n", BENCHMARK_UNIT, benchmark_inference.read());
        benchmark_inference.clear();
      #endif
      // Read the predicted values from the model's output tensor
        for(int i = 0; i < output_length; i++)
        {
          // TODO: Make this dynamic for the input format. At the moment this only works for float32.
          output_arr[i] = output->data.f[i];
          pc.printf("\tClass %d (%.10f)\n\t%a\n", i, output_arr[i], output_arr[i]);

          //probability_beforedec = static_cast<int>(output_arr[i] * 100);
          //probability_afterdec = static_cast<int>((output_arr[i] * 100 - probability_beforedec) * 10000);
          //TF_LITE_REPORT_ERROR(error_reporter, "Class %d: %d%.%d%%", i , probability_beforedec, probability_afterdec);
        } 
      pc.printf("_end_report_\n\n");
    #endif //NO_PREDICTIONS

    inference_count = 0;
  }
}
