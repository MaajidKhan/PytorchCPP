#include <iostream>
#include <memory>
#include <stdio.h>
// OPENCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
// TORCH
#include <torch/script.h>
#include <torch/torch.h>

#define DEFAULT_HEIGHT 720
#define DEFAULT_WIDTH 1280
#define IMG_SIZE 512

// PROTOTYPES
cv::Mat frame_prediction(cv::Mat frame, torch::jit::Module model);
torch::jit::Module load_model(std::string model_name);

int main() {
  // Set torch module
  torch::jit::script::Module module;
  // OPENCV
  cv::VideoCapture cap;
  cv::Mat frame;
  cap.open("../videos/driving.mp4");

  if (!cap.isOpened()) {
    std::cerr << "\nCannot open video\n";
  }

  std::cout << "\nPress spacebar to terminate\n";
  // Load Model
  try {
    module = load_model("traced_lanesNet.pt");
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
  }

  for (;;) {
    cap.read(frame);
    if (frame.empty()) {
      std::cerr << "\nError:Blank Frame\n";
    }

    frame = frame_prediction(frame, module);
    cv::imshow("video", frame);

    if (cv::waitKey(1) >= 27) { // Press space bar to close the window
      break;
    }
  }
}

torch::jit::Module load_model(std::string model_name) {
  std::string directory = "../models/" + model_name;
  torch::jit::Module module = torch::jit::load(directory);
  //module.to(torch::kCUDA); //If you want to use CUDA
  module.eval();
  std::cout << "MODEL LOADED";
  return module;
}

cv::Mat frame_prediction(cv::Mat frame, torch::jit::Module model) {
  // Needed for Overlay
  double alpha = 0.1;
  double beta = (1 - alpha);
  cv::Mat frame_copy, dst;
  // Torch model input
  std::vector<torch::jit::IValue> input;
  // Mean and std (used to normalize the input to avoid the input becoming jaggered)
  std::vector<double> mean = {0.406, 0.456, 0.485};
  std::vector<double> std = {0.225, 0.224, 0.229};
  cv::resize(frame, frame, cv::Size(IMG_SIZE, IMG_SIZE));
  frame_copy = frame;
  frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f); //Normalize the frame with float
  // CV2 to Tensor
  torch::Tensor frame_tensor =
      torch::from_blob(frame.data, {1, IMG_SIZE, IMG_SIZE, 3}); // 1, 512, 512, 3 can't be inputed to the model.

  //changing the dimensions of the model input by reordering
  frame_tensor = frame_tensor.permute({0, 3, 1, 2}); // 1, 3, 512, 512
  frame_tensor = torch::data::transforms::Normalize<>(mean, std)(frame_tensor);
  //frame_tensor = frame_tensor.to(torch::kCUDA); ////If you want to use CUDA
  input.push_back(frame_tensor);

  // Forward Pass
  auto pred = model.forward(input).toTensor().detach().to(torch::kCPU);
  pred = pred.mul(100).clamp(0.255).to(torch::kU8); // Make the output between 0 to 255 to make it an RGB

  // Tensor -> CV2
  cv::Mat output_mat(cv::Size{IMG_SIZE, IMG_SIZE}, CV_8UC1, pred.data_ptr());
  cv::cvtColor(output_mat, output_mat, cv::COLOR_GRAY2RGB); //Adding colors. changing the output matrix from Grayscale to RGB
  cv::applyColorMap(output_mat, output_mat, cv::COLORMAP_TWILIGHT_SHIFTED); // colormap functions shifts this colors as mentioned in the above comment.

  // OVERLAY ORIGINAL FRAME AND PREDICTION
  cv::addWeighted(frame_copy, alpha, output_mat, beta, 0.0, dst); // combines our output with original frame
  cv::resize(dst, dst, cv::Size(DEFAULT_WIDTH, DEFAULT_HEIGHT)); //Resize this back to the original (default width, default height)
  return dst;
}
