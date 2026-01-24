#include "gen_data.cpp"
#include "seal/seal.h"
#include <array>
#include <filesystem>

using namespace seal;

void print_parameters_to_folder(const seal::SEALContext &context, std::string folder_name)
{
    auto &context_data = *context.key_context_data();
    std::string param_filename = folder_name + "/params.txt";

    std::ofstream param_file;
    param_file.open(param_filename);

    std::string scheme_name;
    switch (context_data.parms().scheme())
    {
    case seal::scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case seal::scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    case seal::scheme_type::bgv:
        scheme_name = "BGV";
        break;
    default:
        throw std::invalid_argument("unsupported scheme");
    }
    param_file << "/" << std::endl;
    param_file << "| Encryption parameters :" << std::endl;
    param_file << "|   scheme: " << scheme_name << std::endl;
    param_file << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    param_file << "|   coeff_modulus size: ";
    param_file << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
    {
        param_file << coeff_modulus[i].bit_count() << " + ";
    }
    param_file << coeff_modulus.back().bit_count();
    param_file << ") bits" << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == seal::scheme_type::bfv)
    {
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    }

    param_file << "\\" << std::endl;
    param_file.close();
}

int main()
{
    std::size_t trials = 1000;
    const std::size_t num_rows = 10;
    const std::size_t num_features = 5;
    const std::size_t num_iterations = 20;
    const double lamb = 1;
    int Delta = 35;

    std::string folder_name = std::to_string(num_rows) + "_" + std::to_string(num_features) + "_" + std::to_string(num_iterations) + "_" + std::to_string(Delta) + "_" + std::to_string(trials);
    std::filesystem::create_directory(folder_name);

    std::cout << std::fixed << std::setprecision(17);
    Dataset dataset = make_scaled_dataset(num_rows, num_features);
    std::cout << "printing dataset:" << std::endl;

    for (size_t i = 0; i < num_rows; i++)
    {
        for (size_t j = 0; j < num_features; j++)
        {
            std::cout << dataset.X[i][j] << "\t";
        }
        std::cout << "|\t" << dataset.y[i] << std::endl;
    }

    dataset.print_dataset_to_folder(folder_name);

    EncryptionParameters params(scheme_type::ckks);

    size_t poly_modulus_degree = 512;

    double scale = pow(2, Delta);

    params.set_poly_modulus_degree(poly_modulus_degree);
    std::vector<int> mod_chain = {Delta + 20};
    for (std::size_t i = 0; i < num_iterations + 2; i++)
    {
        mod_chain.push_back(Delta);
    }
    mod_chain.push_back(Delta + 20);
    params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, mod_chain));

    SEALContext context(params, true, sec_level_type::none);
    print_parameters_to_folder(context, folder_name);
    KeyGenerator keygen(context);
    CKKSEncoder encoder(context);

    for (size_t trial = 0; trial < trials; trial++)
    {
        std::cout << "#### TRIAL " << trial << std::endl;

        auto secret_key = keygen.secret_key();
        RelinKeys relin_keys;
        keygen.create_relin_keys(relin_keys);
        Encryptor encryptor(context, secret_key);
        Evaluator evaluator(context);
        Decryptor decryptor(context, secret_key);

        Plaintext denominator;
        encoder.encode(1.0 / num_rows, scale, denominator);

        std::array<std::array<Ciphertext, num_features>, num_rows> database;
        std::array<Ciphertext, num_rows> labels;

        for (std::size_t i = 0; i < num_rows; i++)
        {
            Plaintext ptxt;
            encoder.encode(dataset.y[i], scale, ptxt);
            encryptor.encrypt_symmetric(ptxt, labels[i]);

            for (std::size_t j = 0; j < num_features; j++)
            {
                encoder.encode(dataset.X[i][j], scale, ptxt);
                encryptor.encrypt_symmetric(ptxt, database[i][j]);
            }
        }

        std::vector<double> Y_msg = dataset.X[0];
        std::array<Ciphertext, num_features> Y_ctxt = database[0];

        for (std::size_t j = 0; j < num_features; j++)
        {
            for (std::size_t i = 0; i < num_rows; i++)
            {
                if (i == 0)
                {
                    // plaintext version
                    Y_msg[j] *= dataset.y[i];

                    // encrypted version
                    evaluator.multiply_inplace(Y_ctxt[j], labels[i]);
                }

                else
                {
                    // plaintext version
                    Y_msg[j] += dataset.X[i][j] * dataset.y[i];

                    // ciphertext version
                    Ciphertext ctxt;
                    evaluator.multiply(database[i][j], labels[i], ctxt);
                    evaluator.add_inplace(Y_ctxt[j], ctxt);
                }
            }
            evaluator.relinearize_inplace(Y_ctxt[j], relin_keys);
            evaluator.rescale_to_next_inplace(Y_ctxt[j]);

            // divide by number of rows
            Y_msg[j] /= num_rows;
            evaluator.mod_switch_to_inplace(denominator, Y_ctxt[j].parms_id());
            evaluator.multiply_plain_inplace(Y_ctxt[j], denominator);
            evaluator.rescale_to_next_inplace(Y_ctxt[j]);
        }

        std::vector<std::vector<double>> M_msg(num_features, std::vector<double>(num_features));
        std::array<std::array<Ciphertext, num_features>, num_features> M_ctxt;
        for (std::size_t j = 0; j < num_features; j++)
        {
            for (std::size_t k = j; k < num_features; k++)
            {
                for (std::size_t i = 0; i < num_rows; i++)
                {
                    if (i == 0)
                    {
                        // plaintext version
                        M_msg[j][k] = dataset.X[i][j] * dataset.X[i][k];
                        // encrypted version
                        evaluator.multiply(database[i][j], database[i][k], M_ctxt[j][k]);
                    }
                    else
                    {
                        // plaintext version
                        M_msg[j][k] += dataset.X[i][j] * dataset.X[i][k];

                        // ciphertext version
                        Ciphertext ctxt;
                        evaluator.multiply(database[i][j], database[i][k], ctxt);
                        evaluator.add_inplace(M_ctxt[j][k], ctxt);
                    }
                }
                evaluator.relinearize_inplace(M_ctxt[j][k], relin_keys);
                evaluator.rescale_to_next_inplace(M_ctxt[j][k]);

                // divide by number of rows
                M_msg[j][k] /= num_rows;

                evaluator.multiply_plain_inplace(M_ctxt[j][k], denominator);
                evaluator.rescale_to_next_inplace(M_ctxt[j][k]);

                if (j != k)
                {
                    M_msg[k][j] = M_msg[j][k];
                    M_ctxt[k][j] = M_ctxt[j][k];
                }
            }
        }

        std::array<Ciphertext, num_features> beta_ctxt;
        std::vector<double> beta_msg(num_features);

        for (std::size_t iteration = 0; iteration < num_iterations; iteration++)
        {
            double alpha = 1.0 / (1 + iteration);

            if (iteration == 0)
            {
                Plaintext alpha_ptxt;
                encoder.encode(alpha, Y_ctxt[0].parms_id(), scale, alpha_ptxt);
                for (std::size_t j = 0; j < num_features; j++)
                {
                    // plaintext version
                    beta_msg[j] = alpha * Y_msg[j];

                    // encrypted version
                    evaluator.multiply_plain(Y_ctxt[j], alpha_ptxt, beta_ctxt[j]);
                    evaluator.rescale_to_next_inplace(beta_ctxt[j]);
                }
            }

            else
            {
                std::vector<double> new_beta_msg(num_features, 0.0);
                std::array<Ciphertext, num_features> new_beta_ctxt;
                double alpha_y_scale = scale * beta_ctxt[0].scale();
                Plaintext alpha_y;
                encoder.encode(alpha, Y_ctxt[0].parms_id(), alpha_y_scale, alpha_y);

                Plaintext alpha_m;
                encoder.encode(-alpha, M_ctxt[0][0].parms_id(), scale, alpha_m);

                for (std::size_t j = 0; j < num_features; j++)
                {
                    new_beta_msg[j] = alpha * Y_msg[j];
                    evaluator.multiply_plain(Y_ctxt[j], alpha_y, new_beta_ctxt[j]);
                    evaluator.rescale_to_next_inplace(new_beta_ctxt[j]);
                    evaluator.mod_reduce_to_inplace(new_beta_ctxt[j], beta_ctxt[j].parms_id());

                    for (std::size_t k = 0; k < num_features; k++)
                    {
                        if (j == k)
                        {
                            new_beta_msg[j] += (1 - lamb * alpha - alpha * M_msg[j][j]) * beta_msg[j];

                            Ciphertext alpha_m_ctxt;
                            evaluator.multiply_plain(M_ctxt[j][j], alpha_m, alpha_m_ctxt);
                            evaluator.rescale_to_next_inplace(alpha_m_ctxt);
                            evaluator.mod_reduce_to_inplace(alpha_m_ctxt, beta_ctxt[j].parms_id());

                            Plaintext one_minus_;
                            encoder.encode(1 - lamb * alpha, alpha_m_ctxt.parms_id(), alpha_m_ctxt.scale(), one_minus_);
                            evaluator.add_plain_inplace(alpha_m_ctxt, one_minus_);

                            evaluator.multiply_inplace(alpha_m_ctxt, beta_ctxt[j]);

                            evaluator.add_inplace(new_beta_ctxt[j], alpha_m_ctxt);
                        }
                        else
                        {
                            new_beta_msg[j] -= alpha * beta_msg[k] * M_msg[j][k];

                            Ciphertext alpha_m_ctxt;
                            evaluator.multiply_plain(M_ctxt[j][k], alpha_m, alpha_m_ctxt);
                            evaluator.rescale_to_next_inplace(alpha_m_ctxt);
                            evaluator.mod_reduce_to_inplace(alpha_m_ctxt, beta_ctxt[k].parms_id());

                            evaluator.multiply_inplace(alpha_m_ctxt, beta_ctxt[k]);
                            evaluator.add_inplace(new_beta_ctxt[j], alpha_m_ctxt);
                        }
                    }
                    evaluator.relinearize_inplace(new_beta_ctxt[j], relin_keys);
                    evaluator.rescale_to_next_inplace(new_beta_ctxt[j]);
                }
                beta_msg = new_beta_msg;
                beta_ctxt = new_beta_ctxt;
            }
        }

        if (trial == 0)
        {
            std::string beta_msg_filename = folder_name + "/beta_msg.txt";
            std::ofstream beta_msg_file;
            beta_msg_file.open(beta_msg_filename);
            beta_msg_file << std::fixed << std::setprecision(17);
            for (std::size_t j = 0; j < num_features; j++)
            {
                beta_msg_file << beta_msg[j] << " ";
            }
            beta_msg_file << std::endl;
            beta_msg_file.close();

            std::string beta_ctxt_filename = folder_name + "/beta_ctxt.txt";
            std::ofstream beta_ctxt_file;
            beta_ctxt_file.open(beta_ctxt_filename);
            beta_ctxt_file.close();
        }
        // decrypt and write to file
        std::string beta_ctxt_filename = folder_name + "/beta_ctxt.txt";
        std::ofstream beta_ctxt_file;
        beta_ctxt_file.open(beta_ctxt_filename, std::fstream::app);
        beta_ctxt_file << std::fixed << std::setprecision(17);
        for (std::size_t j = 0; j < num_features; j++)
        {
            Plaintext ptxt;
            decryptor.decrypt(beta_ctxt[j], ptxt);
            std::vector<double> beta_noisy;
            encoder.decode(ptxt, beta_noisy);
            beta_ctxt_file << beta_noisy[0] << " ";
        }
        beta_ctxt_file << std::endl;
        beta_ctxt_file.close();
    }
    return 0;
}

//         if (j == 1) {
// std::cout << "j = " << j << ", k = " << k << " expecting to add = " << -alpha * beta_msg[k] * M_msg[j][k] << std::endl;
// Plaintext ptxt;
// decryptor.decrypt(alpha_m_ctxt, ptxt);
// std::vector<double> msg;
// encoder.decode(ptxt, msg);
// for (size_t j = 0; j < 10; j++) {
//     std::cout << msg[j] << "\t";
// }
// std::cout << std::endl << std::endl;
// }

//             std::cout << "j = " << j << ", expecting new beta = " << new_beta_msg[j] << std::endl;
// Plaintext ptxt;
// decryptor.decrypt(new_beta_ctxt[j], ptxt);
// std::vector<double> msg;
// encoder.decode(ptxt, msg);
// for (size_t j = 0; j < 10; j++) {
//     std::cout << msg[j] << "\t";
// }
// std::cout << std::endl << std::endl;

// std::cout << "j = " << j << ", expecting Y = " << Y_msg[j] << std::endl;
// Plaintext ptxt;
// decryptor.decrypt(Y_ctxt[j], ptxt);
// std::vector<double> msg;
// encoder.decode(ptxt, msg);
// for (size_t j = 0; j < 10; j++) {
//     std::cout << msg[j] << "\t";
// }
// std::cout << std::endl << std::endl;

// std::cout << "j = " << j << ", k = " << k << "expecting M = " << M_msg[j][k] << std::endl;
// Plaintext ptxt;
// decryptor.decrypt(M_ctxt[j][k], ptxt);
// std::vector<double> msg;
// encoder.decode(ptxt, msg);
// for (size_t x = 0; x < 10; x++) {
//     std::cout << msg[x] << "\t";
// }
// std::cout << std::endl << std::endl;

// decryptor.decrypt(labels[i], ptxt);
// std::vector<double> msg;
// encoder.decode(ptxt, msg);
// for (size_t j = 0; j < 10; j++) {
//     std::cout << msg[j] << "\t";
// }
// std::cout << std::endl;