#include"image.hpp"
#include"filtering.hpp"
#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<array>
#include<algorithm>

Image adjustBrightness(const Image &image, int amount){
    ImageType t = image.getType();
    int width = image.getWidth();
    int height = image.getHeight();
    Image result = image;
    if(t == ImageType::GREYSCALE){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                unsigned char pixel = result.getGrey(j,i);
                int new_pixel = std::max(0, std::min(255, pixel + amount));
                result.setGrey(j, i, static_cast<unsigned char>(new_pixel));
            }
        }
    }
    else{ 
        for(int i= 0; i < height; i++){
            for(int j = 0; j < width; j++){
                std::array<unsigned char, 3> pixel = result.getRGB(j,i);
                int newR = std::max(0, std::min(255, pixel[0] + amount));
                int newG = std::max(0, std::min(255, pixel[1] + amount));
                int newB = std::max(0, std::min(255, pixel[2] + amount));
                result.setRGB(j, i, {
                    static_cast<unsigned char>(newR),
                    static_cast<unsigned char>(newG),
                    static_cast<unsigned char>(newB)
                });
            }
        }
    }
    return result;
}

Image adjustContrast(const Image &image, double factor){
    // new point is (point-128)*factor + 128
    ImageType t = image.getType();
    int width = image.getWidth();
    int height = image.getHeight();
    Image result = image;
    if(t == ImageType::GREYSCALE){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                unsigned char pixel = result.getGrey(j,i);
                double new_pixel = factor*(pixel - 128.) + 128;
                new_pixel = std::max(0., std::min(255., new_pixel));
                result.setGrey(j, i, static_cast<unsigned char>(new_pixel));
            }
        }
    }
    else{ 
        for(int i= 0; i < height; i++){
            for(int j = 0; j < width; j++){
                std::array<unsigned char, 3> pixel = result.getRGB(j,i);
                double newR = factor*(pixel[0]-128.) + 128.;
                newR = std::min(0., std::max(255., newR));
                double newG = factor*(pixel[1]-128.) + 128.;
                newG = std::min(0., std::max(255., newG));
                double newB = factor*(pixel[2]-128.) + 128.;
                newB = std::min(0., std::max(255., newB));
                result.setRGB(j, i, {
                    static_cast<unsigned char>(newR),
                    static_cast<unsigned char>(newG),
                    static_cast<unsigned char>(newB)
                });
            }
        }
    }
    return result;
}

Image convolve(const Image &image, const std::vector<std::vector<float>> &kernel){
    int width = image.getWidth();
    int height = image.getHeight();
    ImageType t = image.getType();

    int kernelHeight = kernel.size();
    if(kernelHeight==0) return image;
    int kernelWidth = kernel[0].size();
    std::pair<int, int> kernelCentre = {kernelWidth/2, kernelHeight/2};

    Image result(width, height, t);
    if(t == ImageType::GREYSCALE){
        for(int i = 0; i < height; i++){
            for(int j =0; j < width; j++){
                float value = 0.f;
                
                for(int y = 0; y < kernelHeight; y++){
                    for(int x = 0; x < kernelWidth; x++){
                        int newX = j + (x - kernelCentre.first);
                        int newY = i + (y - kernelCentre.second);
                        if(newX >= 0 && newX < width && newY >= 0 && newY < height) {
                            value += image.getGrey(newX, newY)*kernel[y][x];
                        }

                    }
                }
                int val = std::max(0, std::min(static_cast<int>(value), 255));
                result.setGrey(j, i, static_cast<unsigned char>(val));
            }
        }
    }
    else{
        for(int i= 0; i < height; i++){
            for(int j = 0; j < width; j++){
                std::array<float, 3>value = {0., 0., 0.};

                for(int y = 0; y < kernelHeight; y++){
                    for(int x = 0; x < kernelWidth; x++){
                        int newX = j + (x - kernelCentre.first);
                        int newY = i + (y - kernelCentre.second);
                        if(newX >= 0 && newX < width && newY >= 0 && newY < height){
                            std::array<unsigned char, 3>pixel = image.getRGB(newX, newY);
                            for(int _ = 0; _ < 3; _++){
                                value[_] += pixel[_]*kernel[y][x]; 
                            }
                        }
                    }
                }
                std::array<unsigned char, 3>val = {0, 0, 0};
                for(int _ = 0; _ < 3; _++){
                    val[_] = static_cast<unsigned char>(std::max(0, std:: min(255, static_cast<int>(value[_]))));
                }
                result.setRGB(j, i, val);
            }
        }
    }
    return result;
}

Image sharpen(const Image &image){
    const std::vector<std::vector<float>>sharpKernel = {
        {0., -1., 0.},
        {-1., 5., -1.},
        {0., -1., 0.}
    };
    return convolve(image, sharpKernel);
}

Image detectEdges(const Image &image){
    int width = image.getWidth();
    int height = image.getHeight();
    ImageType t = image.getType();

    const std::vector<std::vector<float>> sobelX = {
        {-1., 0., 1.},
        {-2., 0., 2.},
        {-1., 0., 1.}
    };
    const std::vector<std::vector<float>> sobelY = {
        {1., 2., 1.},
        {0., 0., 0.},
        {-1., -2., -1.}
    };
    Image greyImage = image;
    if(t == ImageType::RGB){
        greyImage.toGreyScale();
    }
    Image convX = convolve(greyImage, sobelX);
    Image convY = convolve(greyImage, sobelY);

    Image result(width, height, ImageType::GREYSCALE);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            float gx = convX.getGrey(j, i);
            float gy = convY.getGrey(j, i);
            float magnitude = std::sqrt(gx*gx + gy*gy);
            int value = std::min(255, static_cast<int>(magnitude));
            result.setGrey(j, i, static_cast<unsigned char>(value));
        }
    }
    return result;
}

Image boxFilter(const Image &image, int ksize){
    if(ksize <= 0 || ksize%2==0) throw std::invalid_argument("dimension isnt negative and can't be even");
    int height = image.getHeight();
    int width = image.getWidth();
    const std::vector<std::vector<float>> kernel(ksize, std::vector<float>(ksize, 1.f/(ksize*ksize)));
    return convolve(image, kernel);
}

Image gaussianBlur(const Image &image, int size, double sigma){
    if(size<=0 || size%2==0) throw std::invalid_argument("size should be odd");
    if(sigma<=.1) throw std::out_of_range("sigma should be greater be greater than .1");
    std::vector<std::vector<float>>kernel(size, std::vector<float>(size));
    float sum = 0;
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            int dx = i - size/2;
            int dy = j - size/2;
            kernel[i][j] = std::exp((-(dx*dx+dy*dy))/(2*sigma*sigma));
            sum+=kernel[i][j];
        }
    }
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            kernel[i][j]/=sum;
        }
    }
    return convolve(image , kernel);
}

Image medianBlur(const Image &image, int size){
    if(size <=0 || size % 2==0) throw std::invalid_argument("Size can't be negative and size can only be odd");
    int width = image.getWidth();
    int height = image.getHeight();
    ImageType t = image.getType();
    Image result(width, height, image.getType());
    if(t == ImageType::GREYSCALE){
        for(int i = 0; i < height; i++){
            for(int j= 0; j < width; j++){
                std::vector<unsigned char> neighbours;
                for(int x = 0; x < size; x++){
                    for(int y = 0; y < size; y++){
                        int newX = j + x - size/2;
                        int newY = i + y - size/2;
                        if(newX >= 0 && newX < width && newY >= 0 && newY < height){
                            neighbours.push_back(image.getGrey(newX, newY));
                        }
                    }
                }
                std::sort(neighbours.begin(), neighbours.end());
                result.setGrey(j, i, neighbours[neighbours.size()/2]);
            }
        }
    }
    else{
        for(int i = 0; i < height; i++){
            for(int j= 0; j < width; j++){
                std::vector<unsigned char> r_neighbours;
                std::vector<unsigned char> g_neighbours;
                std::vector<unsigned char> b_neighbours;
                for(int x = 0; x < size; x++){
                    for(int y = 0; y < size; y++){
                        int newX = j + x - size/2;
                        int newY = i + y - size/2;
                        if(newX >= 0 && newX < width && newY >= 0 && newY < height){
                            r_neighbours.push_back(image.getRGB(newX, newY)[0]);
                            g_neighbours.push_back(image.getRGB(newX, newY)[1]);
                            b_neighbours.push_back(image.getRGB(newX, newY)[2]);
                        }
                    }
                }
                std::sort(r_neighbours.begin(), r_neighbours.end());
                std::sort(g_neighbours.begin(), g_neighbours.end());
                std::sort(b_neighbours.begin(), b_neighbours.end());
                result.setRGB(j, i, {
                    r_neighbours[r_neighbours.size()/2],
                    g_neighbours[g_neighbours.size()/2],
                    b_neighbours[b_neighbours.size()/2]
                });
            }
        }
    }
    return result;
}