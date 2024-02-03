#include<bits/stdc++.h>
#include"exmath.hpp"

void relu(exmath::Matrix& m) {
    for(int i = 0; i < m.width(); i++) {
        if(m.get(i, 0) > 0.0) {
            m.set(i, 0, 1);
        }
        else {
            m.set(i, 0, 0);
        }
    }
}

double error(const exmath::Matrix&& m, double t) {
    double ans = 0;
    for(int i = 0; i < m.width(); i++) {
        ans += std::pow(m.get(i, 0) - t, 2); 
    }
    ans /= m.width();
    return ans;
}

exmath::Matrix numerical_gradient(const exmath::Matrix& X, const exmath::Matrix& W, double t) {
    exmath::Matrix R = W;
    for(int y = 0; y < W.height(); y++) {
        for(int x = 0; x < W.width(); x++) {
            constexpr double s = 0.0001;

            auto F = W;
            F.set(x, y, F.get(x, y) + s);
            relu(F);
            const double f = error(X * F, t);

            auto B = W;
            B.set(x, y, B.get(x, y) - s);
            relu(B);
            const double b = error(X * B, t);

            R.set(x, y, (f - b) / 2.0 / s);
        }
    }
    return R;
}

int main() {
    exmath::Matrix X = exmath::Matrix({1,2,3});
    exmath::Matrix W = exmath::Matrix(
        {
            {1,2},
            {3,4},
            {5,6}
        }
    );

    constexpr double t = 2;
    constexpr double lr = 0.7;

    auto a = X * W;
    relu(a);

    std::cout << error(a, t) << std::endl;


    return 0;
}