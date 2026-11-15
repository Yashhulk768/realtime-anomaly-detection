// main.cpp
// ---------------------------------------------------------------------------
// C++ inference for the surveillance anomaly detector.
//
// Runs the INT8 (or FP32) ONNX model with the ONNX Runtime C++ API and OpenCV,
// using exactly the same preprocessing as the Python training code.
//
//   Image mode : prints class probabilities and inference time; optional benchmark.
//   Video mode : analyses N frames per second, smooths predictions over recent
//                frames, prints timestamped alerts and a latency summary.
//
// Usage:
//   anomaly_detect <model.onnx> <image-or-video> [--threads N] [--fps F]
//                  [--bench N] [--threshold T] [--window W]
// ---------------------------------------------------------------------------

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

// Must match training (see train.py / the saved checkpoint)
constexpr int kSize = 128;
const std::vector<std::string> kClasses = {"Arson", "Burglary", "Explosion", "Fighting", "Normal"};
constexpr std::array<float, 3> kMean = {0.485f, 0.456f, 0.406f};  // ImageNet, RGB order
constexpr std::array<float, 3> kStd = {0.229f, 0.224f, 0.225f};
constexpr int kNormalIndex = 4;
constexpr int kMinConsecutive = 2;  // abnormal predictions in a row before an alert

using Clock = std::chrono::steady_clock;

struct Options {
  std::string model;
  std::string input;
  int threads = 1;
  double fps = 5.0;
  int bench = 0;
  float threshold = 0.5f;
  int window = 5;
};

void PrintUsage() {
  std::cout << "Usage: anomaly_detect <model.onnx> <image-or-video> [options]\n"
            << "  --threads N     CPU threads for inference (default 1)\n"
            << "  --fps F         video: frames analysed per second (default 5)\n"
            << "  --bench N       image: time N extra runs and report latency stats\n"
            << "  --threshold T   video: alert confidence threshold (default 0.5)\n"
            << "  --window W      video: smoothing window in analysed frames (default 5)\n";
}

bool ParseArgs(int argc, char** argv, Options& opt) {
  if (argc < 3) return false;
  opt.model = argv[1];
  opt.input = argv[2];
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (i + 1 >= argc) return false;
    std::string v = argv[++i];
    if (a == "--threads") opt.threads = std::max(1, std::stoi(v));
    else if (a == "--fps") opt.fps = std::max(0.1, std::stod(v));
    else if (a == "--bench") opt.bench = std::max(0, std::stoi(v));
    else if (a == "--threshold") opt.threshold = std::stof(v);
    else if (a == "--window") opt.window = std::max(1, std::stoi(v));
    else return false;
  }
  return true;
}

bool IsImage(const std::string& path) {
  auto dot = path.find_last_of('.');
  if (dot == std::string::npos) return false;
  std::string ext = path.substr(dot);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::string FormatTime(double seconds) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%02d:%04.1f", static_cast<int>(seconds / 60),
                std::fmod(seconds, 60.0));
  return buf;
}

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1) + 0.5);
  return v[std::min(idx, v.size() - 1)];
}

void PrintLatency(const std::vector<double>& ms) {
  double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
  std::cout << std::fixed << std::setprecision(2)
            << "Inference latency over " << ms.size() << " runs: mean " << mean
            << " ms | p50 " << Percentile(ms, 50) << " ms | p95 " << Percentile(ms, 95)
            << " ms | ~" << std::setprecision(0) << 1000.0 / mean << " frames/s\n";
}

// ---------------------------------------------------------------------------
// Classifier: wraps the ONNX Runtime session
// ---------------------------------------------------------------------------
class Classifier {
 public:
  Classifier(const std::string& model_path, int threads)
      : env_(ORT_LOGGING_LEVEL_WARNING, "anomaly"),
        session_(nullptr),
        memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        input_(3 * kSize * kSize) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, model_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_.GetInputNameAllocated(0, alloc).get();
    output_name_ = session_.GetOutputNameAllocated(0, alloc).get();
  }

  // Returns softmax probabilities; inference_ms receives the model run time.
  std::vector<float> Classify(const cv::Mat& bgr, double& inference_ms) {
    Preprocess(bgr);

    std::array<int64_t, 4> shape = {1, 3, kSize, kSize};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_, input_.data(), input_.size(), shape.data(), shape.size());
    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};

    auto t0 = Clock::now();
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    inference_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    const float* logits = outputs[0].GetTensorData<float>();
    return Softmax(logits, kClasses.size());
  }

 private:
  // Same steps as training: BGR->RGB, resize, scale to 0-1, normalise, HWC->CHW.
  void Preprocess(const cv::Mat& bgr) {
    cv::Mat rgb, resized, f;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(kSize, kSize), 0, 0, cv::INTER_AREA);
    resized.convertTo(f, CV_32FC3, 1.0 / 255.0);

    const int plane = kSize * kSize;
    for (int y = 0; y < kSize; ++y) {
      const cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
      for (int x = 0; x < kSize; ++x) {
        for (int c = 0; c < 3; ++c) {
          input_[c * plane + y * kSize + x] = (row[x][c] - kMean[c]) / kStd[c];
        }
      }
    }
  }

  static std::vector<float> Softmax(const float* z, size_t n) {
    float max_z = *std::max_element(z, z + n);
    std::vector<float> p(n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += (p[i] = std::exp(z[i] - max_z));
    for (auto& v : p) v /= sum;
    return p;
  }

  Ort::Env env_;
  Ort::Session session_;
  Ort::MemoryInfo memory_;
  std::string input_name_, output_name_;
  std::vector<float> input_;
};

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------
int RunImage(Classifier& clf, const Options& opt) {
  cv::Mat img = cv::imread(opt.input, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Could not read image: " << opt.input << "\n";
    return 1;
  }
  double ms = 0.0;
  auto probs = clf.Classify(img, ms);

  std::vector<size_t> order(probs.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });

  std::cout << "Prediction: " << kClasses[order[0]] << " (" << std::fixed << std::setprecision(1)
            << probs[order[0]] * 100 << "%)\n";
  for (size_t i : order) {
    std::cout << "  " << std::left << std::setw(10) << kClasses[i] << std::right << std::setw(6)
              << std::setprecision(1) << probs[i] * 100 << "%\n";
  }
  std::cout << "First-run inference time: " << std::setprecision(2) << ms << " ms\n";

  if (opt.bench > 0) {
    for (int i = 0; i < 10; ++i) clf.Classify(img, ms);  // warm-up
    std::vector<double> times;
    times.reserve(opt.bench);
    for (int i = 0; i < opt.bench; ++i) {
      clf.Classify(img, ms);
      times.push_back(ms);
    }
    PrintLatency(times);
  }
  return 0;
}

int RunVideo(Classifier& clf, const Options& opt) {
  cv::VideoCapture cap(opt.input);
  if (!cap.isOpened()) {
    std::cerr << "Could not open video: " << opt.input << "\n";
    return 1;
  }
  double src_fps = cap.get(cv::CAP_PROP_FPS);
  if (src_fps <= 0) src_fps = 30.0;
  int step = std::max(1, static_cast<int>(std::lround(src_fps / opt.fps)));

  std::deque<std::vector<float>> recent;
  std::vector<double> times;
  std::vector<double> mean_probs(kClasses.size(), 0.0);
  int analysed = 0, consecutive = 0, alerts = 0, alert_cls = -1;
  long idx = 0;
  cv::Mat frame;

  std::cout << "Analysing " << opt.fps << " frames/s (every " << step << " frames, source "
            << src_fps << " fps)...\n";

  while (cap.read(frame)) {
    if (idx % step == 0) {
      double ms = 0.0;
      recent.push_back(clf.Classify(frame, ms));
      if (static_cast<int>(recent.size()) > opt.window) recent.pop_front();
      times.push_back(ms);
      ++analysed;

      // moving average over the recent window
      std::vector<float> smooth(kClasses.size(), 0.0f);
      for (const auto& p : recent)
        for (size_t c = 0; c < p.size(); ++c) smooth[c] += p[c] / recent.size();
      for (size_t c = 0; c < smooth.size(); ++c) mean_probs[c] += smooth[c];

      int top = static_cast<int>(std::max_element(smooth.begin(), smooth.end()) - smooth.begin());
      bool abnormal = top != kNormalIndex && smooth[top] >= opt.threshold;
      consecutive = abnormal ? consecutive + 1 : 0;

      if (consecutive >= kMinConsecutive) {
        if (alert_cls != top) {
          std::cout << "[" << FormatTime(idx / src_fps) << "] ALERT " << kClasses[top] << " ("
                    << std::fixed << std::setprecision(0) << smooth[top] * 100 << "%)\n";
          ++alerts;
        }
        alert_cls = top;
      } else {
        alert_cls = -1;
      }
    }
    ++idx;
  }

  if (analysed == 0) {
    std::cerr << "No frames could be read from the video.\n";
    return 1;
  }

  int best = 0;
  for (size_t c = 1; c < mean_probs.size(); ++c)
    if (mean_probs[c] > mean_probs[best]) best = static_cast<int>(c);

  std::cout << "\nSummary\n"
            << "  Video length:      " << std::fixed << std::setprecision(1) << idx / src_fps << " s\n"
            << "  Frames analysed:   " << analysed << "\n"
            << "  Alerts raised:     " << alerts << "\n"
            << "  Most likely class: " << kClasses[best] << " (" << std::setprecision(0)
            << mean_probs[best] / analysed * 100 << "% average confidence)\n  ";
  PrintLatency(times);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!ParseArgs(argc, argv, opt)) {
    PrintUsage();
    return 1;
  }
  try {
    Classifier clf(opt.model, opt.threads);
    std::cout << "Model: " << opt.model << " | threads: " << opt.threads << "\n";
    return IsImage(opt.input) ? RunImage(clf, opt) : RunVideo(clf, opt);
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
  }
  return 1;
}
