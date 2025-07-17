#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include "image.hpp"
#include "filtering.hpp"

// Main function to run the segmentation pipeline on a list of images
int main() {
    // 1. List of input images to process
    const std::vector<std::string> inputFilenames = {
        "image1.ppm",
        "image2.ppm",
        "image3.ppm",
        "image4.ppm",
        "image5.ppm"
    };

    std::cout << "--- Starting Batch Instance Segmentation ---" << std::endl;

    // 2. Loop through each filename and process the image
    for (const std::string& inputFilename : inputFilenames) {
        std::cout << "\nProcessing: " << inputFilename << "..." << std::endl;

        try {
            // Load the source image
            Image originalImage;
            if (!originalImage.loadFromFile(inputFilename)) {
                std::cerr << "Error: Could not load '" << inputFilename << "'. Skipping." << std::endl;
                continue; // Skip to the next image
            }

            // Run the complete instance segmentation pipeline
            // This single function should call all the necessary steps internally:
            // threshold -> opening -> distance transform -> find markers -> watershed
            Image segmentedImage = instanceSegment(originalImage);

            // Create a unique output filename for the segmented image
            // e.g., "image1.ppm" -> "segmented_image1.pgm"
            std::string outputFilename = "segmented_" + inputFilename;
            size_t dot_pos = outputFilename.rfind('.');
            if (dot_pos != std::string::npos) {
                outputFilename.replace(dot_pos, outputFilename.length(), ".pgm");
            } else {
                outputFilename += ".pgm";
            }

            // Save the final result
            if (segmentedImage.save(outputFilename)) {
                std::cout << "Successfully saved result to '" << outputFilename << "'" << std::endl;
            } else {
                std::cerr << "Error: Failed to save '" << outputFilename << "'" << std::endl;
            }

        } catch (const std::exception& e) {
            // Catch any errors that occur during the segmentation of a single image
            std::cerr << "An error occurred while processing " << inputFilename << ": " << e.what() << std::endl;
        }
    }

    std::cout << "\n--- Batch Processing Complete ---" << std::endl;
    std::cout << "Check the directory for the 'segmented_*.pgm' files." << std::endl;

    return 0;
}
