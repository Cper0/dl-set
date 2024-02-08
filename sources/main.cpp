#include<bits/stdc++.h>
#include<armadillo>
#include"exmath.hpp"
#include"tiny-mnist.hpp"

constexpr double rand_std = 0.01;

constexpr int batch_size = 100;

constexpr double lr = 10;

arma::mat W1 = arma::mat(100, 784, arma::fill::randu) * rand_std;
arma::vec B1 = arma::vec(100, arma::fill::randu) * rand_std;

arma::mat W2 = arma::mat(10, 100, arma::fill::randu) * rand_std;
arma::vec B2 = arma::vec(10, arma::fill::randu) * rand_std;

arma::vec gen_one_hot(int t, int s) {
    auto V = arma::vec(s, arma::fill::zeros);
    V(t) = 1.0;
    return V;
}


arma::vec sigmoid(const arma::vec& M) {
    return 1.0 / (1.0 + arma::exp(-M));
}

arma::vec softmax(const arma::vec& M) {
    const auto Y = sigmoid(M);
    const double s = arma::sum(Y);

    return Y / (s + 0.0001);
}

double cross_entropy_error(const arma::vec& X, int t) {
    return -std::log(X(t));
}

std::tuple<arma::mat,arma::mat,arma::mat,arma::mat> gradient(const arma::vec& X, const arma::vec& T) {
    const arma::mat Z1 = W1 * X + B1;
    const arma::mat Y1 = sigmoid(Z1);

    const arma::mat Z2 = W2 * Y1 + B2;
    const arma::mat Y2 = softmax(Z2);

    //逆伝播法による勾配計算
    const arma::mat dZ2 = (Y2 - T) / batch_size;
    const arma::mat dB2 = dZ2;
    //const arma::mat dW2 = Y1.t() * dB2;
    const arma::mat dW2 = dB2 * Y1.t();
    const arma::mat dY1 = W2.t() * dB2;
    
    /*
    100*1 10*1
    10*100
    
    */

    const arma::mat dZ1 = arma::dot(Y1,1.0 - Y1) * dY1;
    const arma::mat dB1 = dZ1;
    const arma::mat dW1 = dB1 * X.t();

    return std::tie(dW1, dB1, dW2, dB2);
}

void update(const arma::vec& X, const arma::vec& T) {
    const auto [dW1, dB1, dW2, dB2] = gradient(X, T);

    W1 -= dW1 * lr;
    B1 -= dB1 * lr;

    W2 -= dW2 * lr;
    B2 -= dB2 * lr;
}

arma::vec predict(const arma::vec& X, int t) {
    const auto Z1 = W1 * X + B1;
    const auto Y1 = sigmoid(Z1);


    const auto Z2 = W2 * Y1 + B2;
    const auto Y2 = softmax(Z2);

    return Y2;
}

int main() {
    mnist::MNISTDatabase db = mnist::MNISTDatabase();

    double accuracy = 0;
    int right = 0;

    const int max_epochs = db.get_train_size() / batch_size;

    for(int j = 0; j < max_epochs; j++) {
        for(int i = 0; i < batch_size; i++) {
            std::vector<double> a;
            int t;
            
            if(db.get_data(i, a, t)) {
                std::cout << "Error has occured while getting data from MNIST." << std::endl;
            }
            
            const arma::vec X(a);
            const arma::vec T = gen_one_hot(t, 10);

            update(X, T);

            const auto Y = predict(X, t);
			const double ce_error = cross_entropy_error(Y, t);
            const int tries = j * batch_size + i + 1;
			
			if(Y.index_max() == t) {
				right++;
			}
			
            std::cout << "[" << tries << "] ce_error=" << ce_error << " correct=" << std::round(100.0 * right / tries) << std::endl;
        }
    } 

    return 0;
}