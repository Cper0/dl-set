#pragma once

#include<bits/stdc++.h>

namespace exmath {
    using cell = double;

    class Matrix {
    private:
        std::vector<std::vector<cell>> data;

    public:
        explicit Matrix() = default;
        explicit Matrix(int w, int h) : data(h, std::vector<cell>(w)) {};
        explicit Matrix(const std::initializer_list<cell> a) : data(1, std::vector<cell>(a.begin(), a.end())) {};
        explicit Matrix(const std::initializer_list<std::initializer_list<cell>> a) : data(a.begin(), a.end()) {};

        Matrix operator+(const Matrix& m) const {
            if(!(width() == m.width() && height() == m.height())) {
                throw std::exception();
            }

            Matrix n = Matrix(width(), height());
            for(int i = 0; i < height(); i++) {
                for(int j = 0; j < width(); j++) {
                    n.set(j, i, get(j, i) + m.get(j, i));
                }
            }

            return n;
        };

        Matrix operator-(const Matrix& m) const {
            if(!(width() == m.width() && height() == m.height())) {
                throw std::exception();
            }

            Matrix n = Matrix(width(), height());
            for(int i = 0; i < height(); i++) {
                for(int j = 0; j < width(); j++) {
                    n.set(j, i, get(j, i) - m.get(j, i));
                }
            }

            return n;
        };

        Matrix operator*(const Matrix& m) const {
            if(width() != m.height()) {
                throw std::exception();
            }

            Matrix n = Matrix(m.width(), height());
            for(int y = 0; y < n.height(); y++) {
                for(int x = 0; x < n.width(); x++) {
                    cell sum = 0;
                    for(int i = 0; i < width(); i++) {
                        sum += get(i, y) * m.get(x, i);
                    }
                    n.set(x, y, sum);
                }
            }

            return n;
        };

        void scalar(double x) {
            for(int i = 0; i < height(); i++) {
                for(int j = 0; j < width(); j++) {
                    set(j, i, get(j, i) * x);
                }
            }
        }

        int width() const noexcept { return data[0].size(); };
        int height() const noexcept { return data.size(); };

        cell get(int x, int y) const { return data[y][x]; };
        void set(int x, int y, cell v) { data[y][x] = v; };
    };
}