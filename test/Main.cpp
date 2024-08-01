#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <vector>

#include "inpaint.h"


const char* keys =
{
    "{i || input image name}"
    "{m || mask image name}"
    "{o || output image name}"
};

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

    cv::Mat mask = cv::imread(maskFilename, cv::IMREAD_GRAYSCALE);
    if (mask.empty())
    {
        printf("Cannot read image file: %s\n", maskFilename.c_str());
        return -1;
    }
    //

    cv::Mat res(src.size(), src.type());

    int time = clock();
    inpaint(src, mask, res);
    std::cout << "time = " << (clock() - time)
        / double(CLOCKS_PER_SEC) << std::endl;

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
