#include<bits/stdc++.h>
#include<glm/exponential.hpp>
#include"exmath.hpp"
#include"tiny-mnist.hpp"


exmath::Matrix W1 = exmath::Matrix(100, 784);
exmath::Matrix B1 = exmath::Matrix(100, 1);

exmath::Matrix W2 = exmath::Matrix(10, 100);
exmath::Matrix B2 = exmath::Matrix(10, 1);

void dump(const exmath::Matrix& m) {
    std::stringstream ss;
    ss << "{\n";
    for(int y = 0; y < m.height(); y++) {
        ss << " {";
        for(int x = 0; x < m.width(); x++) {
            ss << m.get(x, y);
            if(x + 1 < m.width()) {
                ss << ",";
            }
        }
        
        ss << "}";
        if(y + 1 < m.height()) {
            ss << ",\n";
        }
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
}

double sig(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

void sigmoid(exmath::Matrix &m) {
    for(int i = 0; i < m.width(); i++) {
        m.set(
            i,
            0,
            sig(m.get(i, 0))
        );
    }
}

void softmax(exmath::Matrix& M) {
    double sig_sum = 0;
    for(int i = 0; i < M.width(); i++) {
        const double d = M.get(i, 0);

        M.set(i, 0, sig(d));
        sig_sum += sig(d);
    }

    for(int i = 0; i < M.width(); i++) {
        M.set(i, 0, M.get(i, 0) / sig_sum);
    }
}

double cross_entropy_error(const exmath::Matrix& m, int t) {
    return -std::log(m.get(t, 0));
}

exmath::Matrix work(const exmath::Matrix& X) {
    auto Y1 = X * W1 + B1;
    sigmoid(Y1);

    auto Y2 = Y1 * W2 + B2;
    softmax(Y2);

    return Y2;
}

double flow(const exmath::Matrix& X, int t) {
    const auto e = cross_entropy_error(work(X), t);
    return e;
}

exmath::Matrix numerical_gradient_unit(const exmath::Matrix& X, int t, exmath::Matrix& W) {
    constexpr double s = 0.0001;

    exmath::Matrix N(W);
    
    for(int y = 0; y < W.height(); y++) {
        for(int x = 0; x < W.width(); x++) {
            const auto reserve = W.get(x, y);

            {
                W.set(x, y, reserve + s);
                const auto f = flow(X, t);
                W.set(x, y, reserve - s);
                const auto g = flow(X, t);
                
                const auto d = (f - g) / 2.0 / s;
                N.set(x, y, d);
            }

            W.set(x, y, reserve);
        }
    }

    return N;
}



void update(const exmath::Matrix& X, int t, double lr) {
    auto D1 = numerical_gradient_unit(X, t, W1);
    auto D2 = numerical_gradient_unit(X, t, W2);
    auto D3 = numerical_gradient_unit(X, t, B1);
    auto D4 = numerical_gradient_unit(X, t, B2);

    D1.scalar(lr);
    D2.scalar(lr);
    D3.scalar(lr);
    D4.scalar(lr);

    W1 = W1 - D1;
    W2 = W2 - D2;
    B1 = B1 - D3;
    B2 = B2 - D4;
}


int main() {
    mnist::MNISTDatabase db = mnist::MNISTDatabase();

    double accuracy = 0;
    int right = 0;

    for(int i = 0; i < 100; i++) {
        std::vector<double> a;
        int b;
        
        if(db.get_data(i, a, b)) {
            std::cout << "Error has occured while getting data from MNIST." << std::endl;
        }
        
        const exmath::Matrix X = exmath::Matrix(a);
        const int t = b;
        constexpr double lr = 10.0;

        const exmath::Matrix Z = X;

        update(X, t, lr);

        const auto R = work(X);
        dump(R);

        auto it = std::max_element(R.D()[0].begin(), R.D()[0].end());
        if(std::distance(R.D()[0].begin(), it) == t) {
            right++;
        }

        accuracy = 1.0 * right / (i + 1);
        std::cout << "accuracy=" << accuracy << std::endl;
    }

    return 0;
}