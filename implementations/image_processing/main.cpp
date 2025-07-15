#include <iostream>
#include <string>
#include <exception>
#include<vector>
#include<cmath>
#include<array>
#include<algorithm>
#include "image.hpp"
#include "filtering.hpp" // This header should contain all your filter declarations

int main() {
    // 1. Load the source image from disk
    Image originalImage;
    const std::string inputFilename = "dog.ppm";

    if (!originalImage.loadFromFile(inputFilename)) {
        std::cerr << "Error: Could not load '" << inputFilename << "'.\n";
        std::cerr << "Please ensure the file exists in the correct directory and is a valid PPM (P5/P6) file." << std::endl;
        return 1;
    }
    std::cout << "Successfully loaded '" << inputFilename << "'. Beginning tests...\n" << std::endl;

    // 2. Test all your image processing functions within a try-catch block
    try {
        // --- Test Sharpening ---
        std::cout << "Applying sharpen filter..." << std::endl;
        Image sharpImage = sharpen(originalImage);
        sharpImage.save("output_sharpened.ppm");

        // --- Test Edge Detection ---
        std::cout << "Applying edge detection (Sobel)..." << std::endl;
        Image edgeImage = detectEdges(originalImage);
        edgeImage.save("output_edges.pgm"); // Saved as .pgm (grayscale)

        // --- Test Gaussian Blur ---
        std::cout << "Applying Gaussian blur..." << std::endl;
        Image gaussianImage = gaussianBlur(originalImage, 15, 5.0);
        gaussianImage.save("output_gaussian_blur.ppm");

        // --- Test Box Blur ---
        std::cout << "Applying box blur..." << std::endl;
        Image boxImage = boxFilter(originalImage, 9);
        boxImage.save("output_box_blur.ppm");

        // --- Test Median Blur ---
        std::cout << "Applying median blur..." << std::endl;
        Image medianImage = medianBlur(originalImage, 5);
        medianImage.save("output_median_blur.ppm");

        // --- Test Brightness Adjustment ---
        std::cout << "Adjusting brightness (+50)..." << std::endl;
        Image brightImage = adjustBrightness(originalImage, 50);
        brightImage.save("output_bright.ppm");

        // --- Test Contrast Adjustment ---
        std::cout << "Adjusting contrast (x1.8)..." << std::endl;
        Image contrastImage = adjustContrast(originalImage, 1.8);
        contrastImage.save("output_contrast.ppm");

    } catch (const std::exception& e) {
        std::cerr << "\nAn error occurred during image processing: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nAll image processing tests complete." << std::endl;
    std::cout << "Check the directory for output files (e.g., 'output_sharpened.ppm')." << std::endl;

    return 0;
}
