#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <tuple>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iterator>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

auto getFaceBox(Net net, Mat const& fr, double conf_threshold) -> std::vector<std::vector<int>>
{
    cv::Mat frame = fr.clone();
    double inScaleFactor = 1.0;
    Size size = Size(300, 300);
    // std::vector<int> meanVal = {104, 117, 123};
    Scalar meanVal = Scalar(104, 117, 123);

    cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, size, meanVal, true, false);
    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    std::vector<std::vector<int>> bboxes;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        if(confidence > conf_threshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
            bboxes.emplace_back(std::vector<int>{x1, y1, x2, y2});
        }
    }
    return bboxes;
}

enum class eAgeGroup {
  infant,
  toddler,
  children,
  young_adult,
  adult
};

enum class eGender {
  male,
  female
};

auto to_string(eAgeGroup ageGroup) -> const char* {
  switch(ageGroup) {
  case eAgeGroup::infant:
    return "infant";
  case eAgeGroup::toddler:
    return "toddler";
  case eAgeGroup::children:
    return "children";
  case eAgeGroup::young_adult:
    return "young_adult";
  case eAgeGroup::adult:
    return "adult";
  }
}

auto to_string(eGender gender) -> const char* {
  switch(gender) {
    case eGender::male:
      return "male";
    case eGender::female:
      return "female";
  }
}

struct Item {
  std::vector<int> bounding_box;//cv::Rect bounding_box;
  std::string gender;//eGender gender;
  float genderConfidenceScore;
  std::string ageGroup;//eAgeGroup ageGroup;
  float ageGroupConfidenceScore;
};

struct PictureResults {
  std::string path;
  Item item;
};

void to_json(nlohmann::json& j, Item const& p)
{
  j = {{"bounding_box", p.bounding_box},
       {"gender", p.gender},
       {"genderConfidenceScore", p.genderConfidenceScore},
       {"ageGroup",  p.ageGroup},
       {"ageGroupConfidenceScore", p.ageGroupConfidenceScore}};
}

class AgeAndGender {
  public:
    AgeAndGender(
        string faceProto = "opencv_face_detector.pbtxt",
        string faceModel = "opencv_face_detector_uint8.pb",
        string ageProto = "age_deploy.prototxt",
        string ageModel = "age_net.caffemodel",
        string genderProto = "gender_deploy.prototxt",
        string genderModel = "gender_net.caffemodel",
        std::vector<std::string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)","(38-43)", "(48-53)", "(60-100)"})
        : _ageNet{readNet(ageModel, ageProto)}
        , _genderNet{readNet(genderModel, genderProto)}
        , _faceNet{readNet(faceModel, faceProto)}
        , _ageList{ageList}
        , _genderList{"Male", "Female"}
    {
    }

    auto processImage(cv::Mat const& frame) -> std::vector<Item> {
      auto bboxes = getFaceBox(_faceNet, frame, 0.7);
      if (bboxes.empty()) {
        std::cout << "No face detected, checking next frame." << std::endl;
        return {};
      }
      std::vector<Item> items;
      for (auto const& it : bboxes) {
      //for (auto it = begin(bboxes); it != end(bboxes); ++it) {
        cv::Rect rec(it.at(0) - padding,
                     it.at(1) - padding,
                     it.at(2) - it.at(0) + 2*padding,
                     it.at(3) - it.at(1) + 2*padding);
        cv::Mat face = frame(rec); // take the ROI of box on the frame
        cv::Mat blob = blobFromImage(face, 1, Size(227, 227), MODEL_MEAN_VALUES, false);
        _genderNet.setInput(blob);
        std::vector<float> genderPreds = _genderNet.forward();
        // printing gender here
        // find max element index
        // distance function does the argmax() work in C++
        int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
        auto item = Item{};
        item.bounding_box = it;
        item.gender = _genderList[max_index_gender];
        item.genderConfidenceScore = genderPreds[max_index_gender];

        // Uncomment if you want to iterate through the gender_preds vector
        //for(auto it=begin(gender_preds); it != end(gender_preds); ++it) {
        //  cout << *it << endl;
        //}

        _ageNet.setInput(blob);
        std::vector<float> agePreds = _ageNet.forward();
        // uncomment below code if you want to iterate through the age_preds vector
        //cout << "PRINTING AGE_PREDS" << endl;
        //for(auto it = age_preds.begin(); it != age_preds.end(); ++it) {
        //  cout << *it << endl;
        //}

        // finding maximum indicd in the age_preds vector
        int max_indice_age = std::distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
        item.ageGroup = _ageList[max_indice_age];
        item.ageGroupConfidenceScore = agePreds[max_indice_age];
        //string label = gender + ", " + age; // label
        //cv::putText(frameFace, label, Point(it->at(0), it->at(1) -15), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 255), 2, cv::LINE_AA);
        //imshow("Frame", frameFace);
        //imwrite("out.jpg",frameFace);
        items.push_back(item);
      }
      return items;
    }

  private:
    Net _ageNet;
    Net _genderNet;
    Net _faceNet;
    Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);
    std::vector<std::string> _ageList;
    std::vector<std::string> _genderList;
    int padding{20};
};

int main(int argc, char** argv)
{
    if (argc < 7) {
      std::cout << "Usage: ageDetect -path <fileName or path to folder> -json <jsonFileName> -outPath <folderName>" << std::endl;
    }
    std::string path = "";
    std::string jsonPath = "./results.json";
    std::string outPath = "./results/";
    for (int i = 1; i < argc; i += 2) {
      auto str = std::string(argv[i]);
      if (str == "-path") {
        path = argv[++i];
      } else if (str == "-json") {
        jsonPath = argv[++i];
      } else if (str == "-outPath") {
        outPath = argv[++i];
      }
    }
    bool isFolder = std::filesystem::is_directory(path);
    std::cout << "std::filesystem::is_directory: " << isFolder << ", " << path << std::endl;
    std::cout << "jsonPath: " << jsonPath << std::endl;
    std::cout << "outPath:" << outPath << std::endl;
    std::filesystem::create_directories(outPath);

    std::vector<Item> items;
    AgeAndGender ageAndGender{};
    VideoCapture cap;
    if (path.empty()) {
      cap.open(0);
      while(waitKey(1) < 0) {
        Mat frame;
        cap.read(frame);
        if (frame.empty())
        {
            waitKey();
            break;
        }
        items = ageAndGender.processImage(frame);
      }
    } else if (isFolder) {
        for (auto file : std::filesystem::directory_iterator(path)) {
          cv::Mat frame = cv::imread(file.path().string(), cv::IMREAD_COLOR);
          items = ageAndGender.processImage(frame);
        }
    } else {
      cv::Mat frame = cv::imread(path, cv::IMREAD_COLOR);
      items = ageAndGender.processImage(frame);
    }

    nlohmann::json outJson;
    outJson = items;
    std::ofstream output_file(jsonPath);
    output_file << outJson << std::endl;

    return 0;
}
