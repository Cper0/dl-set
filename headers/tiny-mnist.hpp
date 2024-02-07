#pragma once

#include<bits/stdc++.h>

#define DATAPAIRS (10000)
#define IMGPIXELS (28 * 28)

namespace mnist {
    class MNISTDatabase {
    private:
        std::vector<uint8_t> learn;
        std::vector<uint8_t> answer;

        int img_count;
        int label_count;

        int byte_swap(int v) {
            int n = 0;
            n |= (v & 0x000000FF) << (8 * 3);
            n |= (v & 0x0000FF00) << (8 * 1);
            n |= (v & 0x00FF0000) >> (8 * 1);
            n |= (v & 0xFF000000) >> (8 * 3);
            return n;
        }

        void load_img_data(const std::string& p, std::vector<uint8_t>& data) {
            std::ifstream s(p, std::ios_base::binary);

            int i, c, w, h;
            s.read(reinterpret_cast<char*>(&i), sizeof(i));
            s.read(reinterpret_cast<char*>(&c), sizeof(c));
            s.read(reinterpret_cast<char*>(&w), sizeof(w));
            s.read(reinterpret_cast<char*>(&h), sizeof(h));

            const int count = byte_swap(c);
            const int width = byte_swap(w);
            const int height = byte_swap(h);

            img_count = count;

            const int img_size = width * height;

            data = std::vector<uint8_t>(1LL * img_size * count);
            std::cout << width << "," << height << "," << data.size() << std::endl;
            for(int i = 0; i < count; i++) {
                s.read(reinterpret_cast<char*>(&data[i * img_size]), img_size);
            }

            s.close();
        };

        void load_label_data(const std::string& p, std::vector<uint8_t>& data) {
            std::ifstream s(p, std::ios_base::binary);

            int i, c;
            s.read(reinterpret_cast<char*>(&i), sizeof(i));
            s.read(reinterpret_cast<char*>(&c), sizeof(c));

            const int count = byte_swap(c);

            label_count = count;

            data = std::vector<uint8_t>(count);
            for(int i = 0; i < count; i++) {
                s.read(reinterpret_cast<char*>(&data[i]), 1);
            }

            s.close();
        };

    public:
        explicit MNISTDatabase() : learn(), answer() {
            const std::string img_data = "train-images-idx3-ubyte";
            const std::string lab_data = "train-labels-idx1-ubyte";

            load_img_data(img_data, learn);
            load_label_data(lab_data, answer);
        };

        bool get_data(int idx, std::vector<double>& img, int& label) const {
            if(idx >= DATAPAIRS) return true;

            img = std::vector<double>(IMGPIXELS);
            for(int i = 0; i < IMGPIXELS; i++) {
                img[i] = (static_cast<double>(learn[idx * IMGPIXELS + i])) / 255.0;
                assert(img[i] >= 0.0);
            }

            label = answer[idx];
            return false;
        }

        int get_train_size() const noexcept { return std::min(img_count, label_count); };
    };
}