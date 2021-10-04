/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief TODO: brief description
 */

#include "plssvm/core.hpp"

#include <cstddef>    // std::size_t
#include <exception>  // std::exception
#include <fstream>    // std::ofstream
#include <iostream>   // std::cerr, std::endl, std::cout
#include <vector>     // std::vector

// TODO: rewrite using now implemented functions?

int main(int argc, char *argv[]) {
    try {
        using real_type = double;

        // parse SVM parameter from command line
        plssvm::parameter_predict<real_type> params{ argc, argv };

        // create SVM
        auto svm = plssvm::make_csvm(params);

        std::ofstream out{ params.predict_filename };
        std::vector<real_type> labels = svm->predict_label(*params.test_data_ptr);
        for (real_type label : labels) {
            out << label << '\n';
        }
        out.close();

        // print achieved accuracy if possible
        if (params.value_ptr) {
            std::size_t correct = 0;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                // check of prediction was correct if correct label exists
                if ((*params.value_ptr)[i] * labels[i] > real_type{ 0.0 }) {
                    ++correct;
                }
            }
            std::cout << "Accuracy = " << static_cast<real_type>(correct) / static_cast<real_type>(params.test_data_ptr->size()) * 100
                      << "% (" << correct << "/" << static_cast<real_type>(params.test_data_ptr->size()) << ") (classification)" << std::endl;
        }

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
