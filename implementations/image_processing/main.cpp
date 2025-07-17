#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <iomanip>
#include "image.hpp"
#include "filtering.hpp"

// Function to remap segmentation labels to more visible values
Image enhanceSegmentationVisibility(const Image& segmentedImage) {
    int width = segmentedImage.getWidth();
    int height = segmentedImage.getHeight();
    
    // First, find all unique labels in the segmented image
    std::vector<int> histogram = const_cast<Image&>(segmentedImage).create_histogram();
    std::vector<unsigned char> unique_labels;
    
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            unique_labels.push_back(static_cast<unsigned char>(i));
        }
    }
    
    std::cout << "Found " << unique_labels.size() << " unique labels: ";
    for (size_t i = 0; i < unique_labels.size(); i++) {
        std::cout << (int)unique_labels[i];
        if (i < unique_labels.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Create enhanced RGB image with better visibility
    Image enhancedImage(width, height, ImageType::RGB);
    
    // Predefined color palette for different regions
    std::vector<std::array<unsigned char, 3>> colors = {
        {0, 0, 0},       // Black for background (label 0)
        {255, 0, 0},     // Red
        {0, 255, 0},     // Green
        {0, 0, 255},     // Blue
        {255, 255, 0},   // Yellow
        {255, 0, 255},   // Magenta
        {0, 255, 255},   // Cyan
        {255, 128, 0},   // Orange
        {128, 0, 255},   // Purple
        {255, 192, 203}, // Pink
        {128, 128, 128}, // Gray
        {255, 165, 0},   // Orange
        {0, 128, 0},     // Dark Green
        {128, 0, 0},     // Dark Red
        {0, 0, 128},     // Dark Blue
        {128, 128, 0}    // Olive
    };
    
    // Map labels to colors
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char original_label = segmentedImage.getGrey(j, i);
            std::array<unsigned char, 3> color = {0, 0, 0}; // Default to black
            
            // Find the position of this label in unique_labels
            for (size_t k = 0; k < unique_labels.size(); k++) {
                if (unique_labels[k] == original_label) {
                    if (k < colors.size()) {
                        color = colors[k];
                    } else {
                        // Generate pseudo-random colors for labels beyond predefined palette
                        unsigned char r = static_cast<unsigned char>((k * 73) % 256);
                        unsigned char g = static_cast<unsigned char>((k * 151) % 256);
                        unsigned char b = static_cast<unsigned char>((k * 211) % 256);
                        color = {r, g, b};
                    }
                    break;
                }
            }
            
            enhancedImage.setRGB(j, i, color);
        }
    }
    
    return enhancedImage;
}

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

            // Enhance segmentation visibility
            Image enhancedSegmentedImage = enhanceSegmentationVisibility(segmentedImage);

            // Create a unique output filename for the segmented image
            // e.g., "image1.ppm" -> "segmented_image1.pgm"
            std::string outputFilename = "segmented_" + inputFilename;
            size_t dot_pos = outputFilename.rfind('.');
            if (dot_pos != std::string::npos) {
                outputFilename.replace(dot_pos, outputFilename.length(), ".pgm");
            } else {
                outputFilename += ".pgm";
            }

            // Display histogram of segmented image
            std::cout << "\n=== Histogram Analysis of Original Segmented Image ===" << std::endl;
            Image tempSegmented = segmentedImage; // Create a copy for histogram
            std::vector<int> histogram = tempSegmented.create_histogram();
            
            // Count non-zero bins and show distribution
            int non_zero_bins = 0;
            int total_pixels = segmentedImage.getWidth() * segmentedImage.getHeight();
            
            std::cout << "Pixel value distribution:" << std::endl;
            for (int i = 0; i < 256; i++) {
                if (histogram[i] > 0) {
                    non_zero_bins++;
                    double percentage = (double)histogram[i] / total_pixels * 100.0;
                    std::cout << "  Value " << i << ": " << histogram[i] << " pixels (" 
                              << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
                }
            }
            
            std::cout << "Total non-zero bins: " << non_zero_bins << std::endl;
            std::cout << "Total pixels: " << total_pixels << std::endl;
            
            // Check if image is completely black
            if (histogram[0] == total_pixels) {
                std::cout << "WARNING: Segmented image is completely black!" << std::endl;
            } else if (non_zero_bins <= 2) {
                std::cout << "WARNING: Very few distinct values - segmentation may have failed!" << std::endl;
            } else {
                std::cout << "Segmentation appears to have " << (non_zero_bins - 1) << " distinct regions" << std::endl;
            }
            std::cout << "============================================" << std::endl;

            // Create enhanced visibility version
            std::cout << "\n=== Creating Enhanced RGB Visibility Version ===" << std::endl;
            Image enhancedSegmentation = enhanceSegmentationVisibility(segmentedImage);
            std::cout << "Enhanced segmentation created as RGB image with color-coded regions." << std::endl;

            // Save the original segmentation result
            if (segmentedImage.save(outputFilename)) {
                std::cout << "Successfully saved original segmentation to '" << outputFilename << "'" << std::endl;
            } else {
                std::cerr << "Error: Failed to save '" << outputFilename << "'" << std::endl;
            }
            
            // Save the enhanced visibility version as RGB
            std::string enhancedFilename = "enhanced_" + outputFilename;
            // Change extension to .ppm for RGB format
            size_t enhanced_dot_pos = enhancedFilename.rfind('.');
            if (enhanced_dot_pos != std::string::npos) {
                enhancedFilename.replace(enhanced_dot_pos, enhancedFilename.length(), ".ppm");
            } else {
                enhancedFilename += ".ppm";
            }
            
            if (enhancedSegmentation.save(enhancedFilename)) {
                std::cout << "Successfully saved enhanced RGB segmentation to '" << enhancedFilename << "'" << std::endl;
            } else {
                std::cerr << "Error: Failed to save '" << enhancedFilename << "'" << std::endl;
            }

        } catch (const std::exception& e) {
            // Catch any errors that occur during the segmentation of a single image
            std::cerr << "An error occurred while processing " << inputFilename << ": " << e.what() << std::endl;
        }
    }

    std::cout << "\n--- Batch Processing Complete ---" << std::endl;
    std::cout << "Check the directory for:" << std::endl;
    std::cout << "  - 'segmented_*.pgm' files (original greyscale segmentation)" << std::endl;
    std::cout << "  - 'enhanced_*.ppm' files (RGB color-coded segmentation)" << std::endl;

    return 0;
}
