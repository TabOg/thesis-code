#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>

struct Dataset {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::vector<double> beta;

    void print_dataset_to_folder(std::string folder_name) {

    std::string y_filename = folder_name + "/y.txt";
    std::string X_filename = folder_name + "/X.txt";

    std::ofstream yfile;
    yfile.open(y_filename);
    yfile << std::fixed << std::setprecision(17);
    for (std::size_t i = 0; i < size(y); i++) {
        yfile << y[i] << " ";
    }
    yfile.close();

    std::ofstream Xfile;
    Xfile.open(X_filename);
    Xfile << std::fixed << std::setprecision(17);
    for (std::size_t i = 0; i < size(X); i++) {
        for (std::size_t j = 0; j < size(X[0]); j++) {
            Xfile << X[i][j] << " ";
            }
        Xfile << std::endl;
        }
    Xfile.close();
    }
};

Dataset make_scaled_dataset(std::size_t num_rows, std::size_t d, double noise_stddev = 0.25) {
    std::mt19937 rng(std::random_device{}());

    // each x_ij in [-1, 1]
    std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);

    // noise distribution
    std::normal_distribution<double> gaussian_dist(0.0, noise_stddev);

    Dataset data;
    data.X.assign(num_rows, std::vector<double>(d));
    data.beta.assign(d, 0.0);
    data.y.assign(num_rows, 0.0);

    // sample synthetic weights
    for (std::size_t j = 0; j < d; j++) {
        data.beta[j] = uniform_dist(rng);
    }

    std::vector<double> unscaled_y(num_rows, 0.0);
    double max_abs_y = 0.0;
    for (std::size_t i = 0; i < num_rows; i++) {
        for (std::size_t j = 0; j < d; j++) {
            double x_ij = uniform_dist(rng);
            data.X[i][j] = x_ij;
            unscaled_y[i] += x_ij * data.beta[j];
        }
        double noise = gaussian_dist(rng);
        unscaled_y[i] += noise;

        max_abs_y = std::max(max_abs_y, std::abs(unscaled_y[i]));
    }

    if (max_abs_y > 0.0) {
        double scale = 1.0 / max_abs_y;

        for (std::size_t i = 0; i < num_rows; i++) {
            data.y[i] = unscaled_y[i] * scale;
        }

        for (std::size_t i = 0; i < d; i++) {
            data.beta[i] *= scale;
        }
    }
    return data;
}



// int main() {
//     std::size_t num_rows = 10;
//     std::size_t d = 10;
//     Dataset database = make_scaled_database(num_rows, d);

//     std::cout << " ### beta = ";
//     for (size_t i = 0; i < d; i++) {
//         std::cout << database.beta[i] << "\t";
//     }
//     std::cout << std::endl;

//     for (size_t i = 0; i < num_rows; i++) {
//         for (size_t j = 0; j < d; j++) {
//         std::cout << database.X[i][j] << "\t";
//         }
//         std::cout << database.y[i] << std::endl;
//     }

//     return 0;
// }