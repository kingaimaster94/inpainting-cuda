#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <vector>

#include "./src/inpaint.h"


const char* keys =
{
    "{i || input image name}"
    "{m || mask image name}"
    "{o || output image name}"
};

#if 1
int main(int argc, const char** argv)
{
    bool printHelp = (argc == 1);
    printHelp = printHelp || (argc == 2 && std::string(argv[1]) == "--help");
    printHelp = printHelp || (argc == 2 && std::string(argv[1]) == "-h");

    if (printHelp)
    {
        printf("\nThis sample demonstrates shift-map image inpainting\n"
            "Call:\n"
            "    inpainting -i=<string> -m=<string> [-o=<string>]\n\n");
        return 0;
    }

    cv::CommandLineParser parser(argc, argv, keys);
    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    std::string inFilename = parser.get<std::string>("i");
    std::string maskFilename = parser.get<std::string>("m");
    std::string outFilename = parser.get<std::string>("o");

    cv::Mat src = cv::imread(inFilename, cv::IMREAD_COLOR);
    if (src.empty())
    {
        printf("Cannot read image file: %s\n", inFilename.c_str());
        return -1;
    }

    //cv::cvtColor(src, src, cv::COLOR_BGR2Lab);

    cv::Mat mask = cv::imread(maskFilename, cv::IMREAD_GRAYSCALE);
    if (mask.empty())
    {
        printf("Cannot read image file: %s\n", maskFilename.c_str());
        return -1;
    }
    //cv::threshold(mask, mask, 128, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat res(src.size(), src.type());

    //CInpainting inpaint;
    int time = clock();
    //inpaint.preprocessing(src, mask);
    //cv::cvtColor(src, src, COLOR_BGR2Luv);
    //inpaint.inpaint(src, mask, res
    inpaint(src, mask, res);
    std::cout << "time = " << (clock() - time)
        / double(CLOCKS_PER_SEC) << std::endl;

    //cv::cvtColor(res, res, cv::COLOR_Luv2BGR);

    if (outFilename == "")
    {
        cv::namedWindow("inpainting result", 1);
        cv::imshow("inpainting result", res);

        cv::waitKey(0);
    }
    else
        cv::imwrite(outFilename, res);

    return 0;
}
#else
using namespace cv;
using namespace std;
static void TrackSlider(int, void*);
static void addSlider(String sliderName, String windowName, int minSlider, int maxSlider, int valDefault, int* valSlider, void(*f)(int, void*), void* r);
vector<int> colorSpace = { COLOR_BGR2GRAY,COLOR_BGR2HSV,COLOR_BGR2YUV,COLOR_BGR2XYZ };

struct OilImage {
    String winName = "Oil painting";
    int size;
    int dynRatio;
    int colorSpace;
    Mat img;
};

const String keys =
"{Help h usage ? help  |     | Print this message   }"
"{v                    | 0   | video index }"
"{a                    | 700   | API index }"
"{s                    | 10   | neighbouring size }"
"{d                    | 1   | dynamic ratio }"
"{c                    | 0   | color space }"
"{@arg1                |     | file path}"
;


int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    String filename = parser.get<String>(0);
    OilImage p;
    p.dynRatio = parser.get<int>("d");
    p.size = parser.get<int>("s");
    p.colorSpace = parser.get<int>("c");
    if (p.colorSpace < 0 || p.colorSpace >= static_cast<int>(colorSpace.size()))
    {
        std::cout << "Color space must be >= 0 and <" << colorSpace.size() << "\n";
        return EXIT_FAILURE;
    }
    if (!filename.empty())
    {
        p.img = imread(filename);
        if (p.img.empty())
        {
            std::cout << "Check file path!\n";
            return EXIT_FAILURE;
        }
        Mat dst;
        xphoto::oilPainting(p.img, dst, p.size, p.dynRatio, colorSpace[p.colorSpace]);
        imshow("oil painting effect", dst);
        waitKey();
        return 0;
    }
    VideoCapture v(parser.get<int>("v") + parser.get<int>("a"));
    v >> p.img;
    p.winName = "Oil Painting";
    namedWindow(p.winName);
    addSlider("DynRatio", p.winName, 1, 127, p.dynRatio, &p.dynRatio, TrackSlider, &p);
    addSlider("Size", p.winName, 1, 100, p.size, &p.size, TrackSlider, &p);
    addSlider("ColorSpace", p.winName, 0, static_cast<int>(colorSpace.size() - 1), p.colorSpace, &p.colorSpace, TrackSlider, &p);
    while (waitKey(20) != 27)
    {
        v >> p.img;
        imshow("Original", p.img);
        TrackSlider(0, &p);
        waitKey(10);
    }
    return 0;
}

void addSlider(String sliderName, String windowName, int minSlider, int maxSlider, int valDefault, int* valSlider, void(*f)(int, void*), void* r)
{
    createTrackbar(sliderName, windowName, valSlider, 1, f, r);
    setTrackbarMin(sliderName, windowName, minSlider);
    setTrackbarMax(sliderName, windowName, maxSlider);
    setTrackbarPos(sliderName, windowName, valDefault);
}

void TrackSlider(int, void* r)
{
    OilImage* p = (OilImage*)r;
    Mat dst;
    p->img = p->img / p->dynRatio;
    p->img = p->img * p->dynRatio;
    xphoto::oilPainting(p->img, dst, p->size, p->dynRatio, colorSpace[p->colorSpace]);
    if (!dst.empty())
    {
        imshow(p->winName, dst);
    }
}

#endif