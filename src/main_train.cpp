/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-train` executable used for training a C-SVM model.
 */

#include "plssvm/core.hpp"

#include "plssvm/detail/cmd/data_set_variants.hpp"         // plssvm::detail::cmd::data_set_factory
#include "plssvm/detail/cmd/parser_train.hpp"              // plssvm::detail::cmd::parser_train
#include "plssvm/detail/logger.hpp"                        // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
    #include "plssvm/detail/tracking/hardware_sampler.hpp"          // plssvm::detail::tracking::hardware_sampler
    #include "plssvm/detail/tracking/hardware_sampler_factory.hpp"  // plssvm::detail::tracking::make_hardware_sampler
#endif

#include <chrono>     // std::chrono::{steady_clock, duration}
#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl
#include <variant>    // std::visit

using namespace std::chrono_literals;

int main(int argc, char *argv[]) {
    try {
        const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

        // parse SVM parameter from command line
        plssvm::detail::cmd::parser_train cmd_parser{ argc, argv };

        // output used parameter
        plssvm::detail::log(plssvm::verbosity_level::full,
                            "\ntask: training\n{}\n\n\n",
                            plssvm::detail::tracking::tracking_entry{ "parameter", "", cmd_parser });

        // create data set
        std::visit([&](auto &&data) {
            using label_type = typename std::remove_reference_t<decltype(data)>::label_type;

            // create SVM
            const std::unique_ptr<plssvm::csvm> svm = (cmd_parser.backend == plssvm::backend_type::sycl) ? plssvm::make_csvm(cmd_parser.backend, cmd_parser.target, cmd_parser.csvm_params, plssvm::sycl_implementation_type = cmd_parser.sycl_implementation_type, plssvm::sycl_kernel_invocation_type = cmd_parser.sycl_kernel_invocation_type)
                                                                                                         : plssvm::make_csvm(cmd_parser.backend, cmd_parser.target, cmd_parser.csvm_params);

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
            // initialize hardware sampling
            std::vector<std::unique_ptr<plssvm::detail::tracking::hardware_sampler>> sampler =
                plssvm::detail::tracking::create_hardware_sampler(svm->get_target_platform(), svm->num_available_devices(), PLSSVM_HARDWARE_SAMPLING_INTERVAL);
            // start sampling
            std::for_each(sampler.begin(), sampler.end(), std::mem_fn(&plssvm::detail::tracking::hardware_sampler::start_sampling));
#endif

            // only specify plssvm::max_iter if it isn't its default value
            const plssvm::model<label_type> model =
                cmd_parser.max_iter.is_default() ? svm->fit(data,
                                                            plssvm::epsilon = cmd_parser.epsilon,
                                                            plssvm::classification = cmd_parser.classification,
                                                            plssvm::solver = cmd_parser.solver)
                                                 : svm->fit(data,
                                                            plssvm::epsilon = cmd_parser.epsilon,
                                                            plssvm::max_iter = cmd_parser.max_iter,
                                                            plssvm::classification = cmd_parser.classification,
                                                            plssvm::solver = cmd_parser.solver);
            // save model to file
            model.save(cmd_parser.model_filename);

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
            // stop sampling
            std::for_each(sampler.begin(), sampler.end(), std::mem_fn(&plssvm::detail::tracking::hardware_sampler::stop_sampling));
            // write samples to yaml file
            std::for_each(sampler.cbegin(), sampler.cend(), [&](const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> &s) {
                PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HARDWARE_SAMPLER_ENTRY(*s);
            });
#endif
        },
                   plssvm::detail::cmd::data_set_factory(cmd_parser));

        const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        plssvm::detail::log(plssvm::verbosity_level::full,
                            "\nTotal runtime: {}\n",
                            plssvm::detail::tracking::tracking_entry{ "", "total_time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(cmd_parser.performance_tracking_filename);

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
