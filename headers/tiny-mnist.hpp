#pragma once

#include<bits/stdc++.h>

#define DATAPAIRS (10000)
#define IMGPIXELS (28 * 28)

namespace mnist {
    class MNISTDatabase {
    private:
        std::vector<int8_t> learn;
        std::vector<int8_t> answer;

    public:
        explicit MNISTDatabase() : learn(DATAPAIRS * IMGPIXELS), answer(DATAPAIRS) {
            const std::string img_data = "t10k-images-idx3-ubyte";
            const std::string lab_data = "t10k-labels-idx1-ubyte";

            std::ifstream s;
            s.open(img_data, std::ios_base::binary);
            for(int i = 0; i < DATAPAIRS; i++) {
                s.read(reinterpret_cast<char*>(learn.data() + i * IMGPIXELS), IMGPIXELS);
            }
            s.close();

            s.open(lab_data, std::ios_base::binary);
            for(int i = 0; i < DATAPAIRS; i++) {
                s.read(reinterpret_cast<char*>(answer.data() + i), 1);
            }
            s.close();

        };

        bool get_data(int idx, std::vector<double>& img, int& label) const {
            if(idx >= DATAPAIRS) return true;

            img = std::vector<double>(IMGPIXELS);
            for(int i = 0; i < IMGPIXELS; i++) {
                img[i] = learn[idx * IMGPIXELS + i] / 255.0;
            }

            label = answer[idx];
            return false;
        }
    };
}