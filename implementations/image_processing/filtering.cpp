#include"image.hpp"
#include"filtering.hpp"
#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<array>
#include<queue>
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

Image convolve(const Image &image, const std::vector<std::vector<double>> &kernel){
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
                double value = 0.f;
                
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
                std::array<double, 3>value = {0., 0., 0.};

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
    const std::vector<std::vector<double>>sharpKernel = {
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

    const std::vector<std::vector<double>> sobelX = {
        {-1., 0., 1.},
        {-2., 0., 2.},
        {-1., 0., 1.}
    };
    const std::vector<std::vector<double>> sobelY = {
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
            double gx = convX.getGrey(j, i);
            double gy = convY.getGrey(j, i);
            double magnitude = std::sqrt(gx*gx + gy*gy);
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
    const std::vector<std::vector<double>> kernel(ksize, std::vector<double>(ksize, 1.f/(ksize*ksize)));
    return convolve(image, kernel);
}

Image gaussianBlur(const Image &image, int size, double sigma){
    if(size<=0 || size%2==0) throw std::invalid_argument("size should be odd");
    if(sigma<=.1) throw std::out_of_range("sigma should be greater be greater than .1");
    std::vector<std::vector<double>>kernel(size, std::vector<double>(size));
    double sum = 0;
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

Image threshold(Image &image){
    int width = image.getWidth();
    int height = image.getHeight();
    Image result = image;
    ImageType t = image.getType();
    result.toGreyScale();
    std::vector<int>histogram = result.create_histogram();
    std::vector<double>cdf(256, 0.);
    double sum = 0.;
    for(int i = 0;i < 256; i++){
        sum+= static_cast<double>(histogram[i]);
        cdf[i] = sum/(height*width);
    }
    double total_sum = 0.;
    for(int i = 0;i < 256; i++){
        total_sum +=  (double)(i*histogram[i]);
    }

    double background_sum = 0.;
    double foreground_sum = 0.;
    int background_weight = 0;

    double variance = 0.;
    int best_thr = 0;
    for (int t = 0; t < 256; ++t) {

        background_weight += histogram[t];
        if (background_weight == 0) continue;

        int foreground_weight = width*height - background_weight;
        if (foreground_weight == 0) break;
        background_sum += (double)(t * histogram[t]);

        double background_mean = background_sum / background_weight;
        double foreground_mean = (total_sum - background_sum) / foreground_weight;
        
        double variance_between = (double)background_weight * (double)foreground_weight *
                                  (background_mean - foreground_mean) * (background_mean - foreground_mean);

        if (variance_between > variance) {
            variance = variance_between;
            best_thr = t;
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (result.getGrey(i, j) > best_thr) {
                result.setGrey(i, j, 255);
            } else {
                result.setGrey(i, j, 0);
            }
        }
    }

    return result;
}

Image dilate(Image& image) {
    int height = image.getHeight();
    int width = image.getWidth();
    ImageType t = image.getType();

    if (t == ImageType::RGB) {
        throw std::logic_error("Dilation only works for grayscale images in this implementation.");
    }
    Image result(width, height, ImageType::GREYSCALE);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char max_val = 0;
            for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
                for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
                    int neighbor_x = j + kernel_x;
                    int neighbor_y = i + kernel_y;

                    if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                        max_val = std::max(max_val, image.getGrey(neighbor_x, neighbor_y));
                    }
                }
            }
            result.setGrey(j, i, max_val);
        }
    }
    return result;
}


Image erode(Image& image) {
    int height = image.getHeight();
    int width = image.getWidth();
    ImageType t = image.getType();

    if (t == ImageType::RGB) {
        throw std::logic_error("Erosion only works for grayscale images in this implementation.");
    }
    Image result(width, height, ImageType::GREYSCALE); 

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) { 
            unsigned char min_val = 255; 
            for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
                for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
                    int neighbor_x = j + kernel_x;
                    int neighbor_y = i + kernel_y;
                    if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                        min_val = std::min(min_val, image.getGrey(neighbor_x, neighbor_y));
                    }
                }
            }
            result.setGrey(j, i, min_val); 
        }
    }
    return result;
}

void Opening(Image &image){
    image = erode(image);
    image = dilate(image);
}

void Closing(Image &image){
    image = dilate(image);
    image = erode(image);
}

Image distanceTransform(const Image &image){
    int width = image.getWidth();
    int height = image.getHeight();
    ImageType t = image.getType();
    if(t!= ImageType::GREYSCALE) throw std::logic_error("Only for binary images");
    std::vector<std::vector<double>>dist(height, std::vector<double>(width, 0.));
    const double max_val = (double)width*height;
    
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if(image.getGrey(j, i)==255) dist[i][j] = max_val;
        }
    }

    for(int i = 0; i < height; i++){
        for(int j =0; j < width; j++){
            if(i > 0) dist[i][j] = std::min(dist[i-1][j]+1, dist[i][j]);
            if(j > 0) dist[i][j] = std::min(dist[i][j], 1+dist[i][j-1]);
        }
    }

    for(int i = height - 1; i >= 0; i--){
        for(int j = width - 1; j >= 0; j--){
            if(i < height - 1) dist[i][j] =  std::min(dist[i+1][j]+1, dist[i][j]);
            if(j < width - 1) dist[i][j] = std::min(dist[i][j+1]+1, dist[i][j]);
        }
    }

    Image result(width, height, ImageType::GREYSCALE);
    double max_dist = 0;
    for (const auto& row : dist) {
        for (double val : row) {
            if (val > max_dist && val < max_val) {
                max_dist = val;
            }
        }
    }

    if (max_dist > 0) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                unsigned char normalized_val = static_cast<unsigned char>((dist[i][j] / max_dist) * 255.0f);
                result.setGrey(j, i, normalized_val);
            }
        }
    }
    
    return result;
}

Image connectedComponents(const Image &binaryImage) {
    int width = binaryImage.getWidth();
    int height = binaryImage.getHeight();
    Image labeledImage(width, height, ImageType::GREYSCALE); 
    int currentLabel = 1;

    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));

    int dx[] = {1, -1, 0, 0, 1, -1, 1, -1};
    int dy[] = {0, 0, -1, 1, 1, -1, -1, 1};

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (binaryImage.getGrey(j, i) == 255 && !visited[i][j]) {
                std::queue<std::pair<int, int>> q;
                q.push({j, i});
                visited[i][j] = true;

                while (!q.empty()) {
                    auto [x, y] = q.front();
                    q.pop();
                    labeledImage.setGrey(x, y, static_cast<unsigned char>(currentLabel)); 

                    for (int k = 0; k < 8; ++k) {
                        int nx = x + dx[k];
                        int ny = y + dy[k];

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                            !visited[ny][nx] && binaryImage.getGrey(nx, ny) == 255) {
                            visited[ny][nx] = true;
                            q.push({nx, ny});
                        }
                    }
                }
                currentLabel++; 
            }
        }
    }
    return labeledImage;
}

Image findMarkers(const Image &distanceMap, double threshold_factor){
    int width = distanceMap.getWidth();
    int height = distanceMap.getHeight();
    unsigned char max_dist = 0;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            max_dist = std::max(max_dist, distanceMap.getGrey(j, i));
        }
    }

    unsigned char thrsld = static_cast<unsigned char>(max_dist * threshold_factor);

    Image binaryMarkers(width, height, ImageType::GREYSCALE);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if(distanceMap.getGrey(j, i) > thrsld) {
                binaryMarkers.setGrey(j, i, 255);
            } else {
                binaryMarkers.setGrey(j, i, 0);
            }
        }
    }

    return connectedComponents(binaryMarkers);
}  

Image watershed(const Image &image, Image &markedImage){
    int width = image.getWidth();
    int height = image.getHeight();
    if(image.getType()!= ImageType::GREYSCALE || markedImage.getType()!=ImageType::GREYSCALE){
        throw std::logic_error("both should be binary only");
    }
    std::priority_queue<std::tuple<int,int,int>, std::vector<std::tuple<int,int,int>>, std::greater<>>q;

    int dx[] = {1,-1, 0, 0, 1, -1, 1, -1};
    int dy[] = {0, 0, -1, 1, 1, -1, -1, 1};

    Image segmentationResult(width, height, ImageType::GREYSCALE);

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            unsigned char marker_label = markedImage.getGrey(j, i);
            if(marker_label > 0){
                q.push({image.getGrey(j, i), j, i});
                segmentationResult.setGrey(j, i, marker_label);
            }
        }
    }
    
    while(!q.empty()){
        auto [brightness, x, y] = q.top(); 
        q.pop();

        unsigned char current_label = segmentationResult.getGrey(x, y);

        for(int i = 0; i < 8; i++){
            int nx = x + dx[i];
            int ny = y + dy[i];
            if(nx>=0&&nx<width && ny>=0&&ny<height){
                if(segmentationResult.getGrey(nx,ny)==0){
                    segmentationResult.setGrey(nx, ny, current_label);
                    q.push({image.getGrey(nx, ny), nx, ny});
                }
            }
        }
    }

    return segmentationResult;
}

Image instanceSegment(Image &image){
    int width = image.getWidth();
    int height = image.getHeight();

    Image temp = image;
    temp.toGreyScale(); 

    Image binaryImage = threshold(temp);
    Opening(binaryImage);

    Image distance_map = distanceTransform(binaryImage);
    
    Image markedImage = findMarkers(distance_map);

    Image watershed_input(width, height, ImageType::GREYSCALE);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            watershed_input.setGrey(j, i, 255 - distance_map.getGrey(j, i));
        }
    }
    
    Image segmentedImage = watershed(watershed_input, markedImage);
    return segmentedImage;
}
