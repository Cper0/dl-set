#include<bits/stdc++.h>
#include"exmath.hpp"
#include"tiny-mnist.hpp"


exmath::Matrix W1 = exmath::Matrix(100, 784);
exmath::Matrix W2 = exmath::Matrix(10, 100);


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

double two_pow_error(const exmath::Matrix& m, double t) {
    double ans = 0;
    for(int i = 0; i < m.width(); i++) {
        ans += std::pow(m.get(i, 0) - (i == t) ? 1.0 : 0.0, 2); 
    }
    ans /= m.width();
    return ans;
}

double flow(const exmath::Matrix& X, int t) {
    auto Y1 = X * W1;
    relu(Y1);

    auto Y2 = Y1 * W2;
    relu(Y2);

    const auto e = two_pow_error(Y2, t);
    return e;
}

exmath::Matrix numerical_gradient_unit(const exmath::Matrix& X, int t, exmath::Matrix& W) {
    constexpr double s = 0.0001;

    exmath::Matrix N(W);

    for(int y = 0; y < W.height(); y++) {
        for(int x = 0; x < W.width(); x++) {
            const auto reserve = W.get(x, y);

            W.set(x, y, reserve + s);
            const auto f = flow(X, t);
            W.set(x, y, reserve - s);
            const auto g = flow(X, t);
            
            const auto d = (f - g) / 2 / s;
            N.set(x, y, d);

            W.set(x, y, reserve);
        }
    }

    return N;
}

void update(const exmath::Matrix& X, int t, double lr) {
    auto D1 = numerical_gradient_unit(X, t, W1);
    auto D2 = numerical_gradient_unit(X, t, W2);

    D1.scalar(lr);
    D2.scalar(lr);

    W1 = W1 - D1;
    W2 = W2 - D2;
}

int main() {
    mnist::MNISTDatabase db = mnist::MNISTDatabase();

    for(int i = 0; i < 100; i++) {
        std::vector<double> a;
        int b;
        
        if(db.get_data(i, a, b)) {
            std::cout << "Error has occured while getting data from MNIST." << std::endl;
        }

        const exmath::Matrix X = exmath::Matrix(a);
        const int t = b;
        constexpr double lr = 0.1;

        const double accuracy = flow(X, t);
        update(X, t, lr);

        std::cout << "accuracy[" << i << "]=" << accuracy << std::endl;
        
        int o = 1;
        std::cout << "continue to enter '0':";
        std::cin >> o;
        if(o != 0) break;
    }

    return 0;
}