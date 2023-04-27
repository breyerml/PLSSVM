/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/performance_tracker.hpp"

#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT
#include "plssvm/detail/cmd/parser_predict.hpp"          // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"            // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"            // plssvm::detail::cmd::parser_train
#include "plssvm/detail/utility.hpp"                     // plssvm::detail::current_date_time
#include "plssvm/version/git_metadata/git_metadata.hpp"  // plssvm::version::git_metadata::commit_sha1
#include "plssvm/version/version.hpp"                    // plssvm::version::{version, detail::target_platforms}

#include "fmt/chrono.h"                                  // format std::chrono types
#include "fmt/core.h"                                    // fmt::format
#include "fmt/ostream.h"                                 // format types with an operator<< overload

#include <fstream>                                       // std::ofstream
#include <iostream>                                      // std::ios_base::app
#include <string>                                        // std::string
#include <unordered_map>                                 // std::unordered_multimap
#include <utility>                                       // std::pair

namespace plssvm::detail {

void performance_tracker::add_tracking_entry(const tracking_entry<std::string> &entry) {
    tracking_statistics.emplace(entry.entry_category, fmt::format("{}{}: \"{}\"\n", entry.entry_category.empty() ? "" : "  ", entry.entry_name, entry.entry_value));
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_train> &entry) {
    tracking_statistics.emplace(entry.entry_category, fmt::format("  task:                        train\n"
                                                                  "  kernel_type:                 {}\n"
                                                                  "  degree:                      {}\n"
                                                                  "  gamma:                       {}\n"
                                                                  "  coef0:                       {}\n"
                                                                  "  cost:                        {}\n"
                                                                  "  epsilon:                     {}\n"
                                                                  "  max_iter:                    {}\n"
                                                                  "  backend:                     {}\n"
                                                                  "  target:                      {}\n"
                                                                  "  sycl_kernel_invocation_type: {}\n"
                                                                  "  sycl_implementation_type:    {}\n"
                                                                  "  strings_as_labels:           {}\n"
                                                                  "  float_as_real_type:          {}\n"
                                                                  "  input_filename:              \"{}\"\n"
                                                                  "  model_filename:              \"{}\"\n",
                                                                  entry.entry_value.csvm_params.kernel_type.value(),
                                                                  entry.entry_value.csvm_params.degree.value(),
                                                                  entry.entry_value.csvm_params.gamma.value(),
                                                                  entry.entry_value.csvm_params.coef0.value(),
                                                                  entry.entry_value.csvm_params.cost.value(),
                                                                  entry.entry_value.epsilon.value(),
                                                                  entry.entry_value.max_iter.value(),
                                                                  entry.entry_value.backend,
                                                                  entry.entry_value.target,
                                                                  entry.entry_value.sycl_kernel_invocation_type,
                                                                  entry.entry_value.sycl_implementation_type,
                                                                  entry.entry_value.strings_as_labels,
                                                                  entry.entry_value.float_as_real_type,
                                                                  entry.entry_value.input_filename,
                                                                  entry.entry_value.model_filename));
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_predict> &entry) {
    tracking_statistics.emplace(entry.entry_category, fmt::format("  task:                     predict\n"
                                                                  "  backend:                  {}\n"
                                                                  "  target:                   {}\n"
                                                                  "  sycl_implementation_type: {}\n"
                                                                  "  strings_as_labels:        {}\n"
                                                                  "  float_as_real_type:       {}\n"
                                                                  "  input_filename:           \"{}\"\n"
                                                                  "  model_filename:           \"{}\"\n"
                                                                  "  predict_filename:         \"{}\"\n",
                                                                  entry.entry_value.backend,
                                                                  entry.entry_value.target,
                                                                  entry.entry_value.sycl_implementation_type,
                                                                  entry.entry_value.strings_as_labels,
                                                                  entry.entry_value.float_as_real_type,
                                                                  entry.entry_value.input_filename,
                                                                  entry.entry_value.model_filename,
                                                                  entry.entry_value.predict_filename));
}

void performance_tracker::add_tracking_entry(const tracking_entry<cmd::parser_scale> &entry) {
    tracking_statistics.emplace(entry.entry_category, fmt::format("  task:               scale\n"
                                                                  "  lower:              {}\n"
                                                                  "  upper:              {}\n"
                                                                  "  format:             {}\n"
                                                                  "  strings_as_labels:  {}\n"
                                                                  "  float_as_real_type: {}\n"
                                                                  "  input_filename:     \"{}\"\n"
                                                                  "  scaled_filename:    \"{}\"\n"
                                                                  "  save_filename:      \"{}\"\n"
                                                                  "  restore_filename:   \"{}\"\n",
                                                                  entry.entry_value.lower,
                                                                  entry.entry_value.upper,
                                                                  entry.entry_value.format,
                                                                  entry.entry_value.strings_as_labels,
                                                                  entry.entry_value.float_as_real_type,
                                                                  entry.entry_value.input_filename,
                                                                  entry.entry_value.scaled_filename,
                                                                  entry.entry_value.save_filename,
                                                                  entry.entry_value.restore_filename));
}

void performance_tracker::save(const std::string &filename) {
    // append the current performance statistics to an already existing file if possible
    std::ofstream out{ filename, std::ios_base::app };
    PLSSVM_ASSERT(out.good(), fmt::format("Couldn't save performance tracking results in '{}'!", filename));

    // begin a new YAML document (only with "---" multiple YAML docments in a single file are allowed)
    out << "---\n";

    // output metadata information
    out << fmt::format(
        "meta_data:\n"
        "  date:                    \"{}\"\n"
        "  PLSSVM_TARGET_PLATFORMS: \"{}\"\n"
        "  commit:                  {}\n"
        "  version:                 {}\n"
        "\n",
        plssvm::detail::current_date_time(),
        version::detail::target_platforms,
        version::git_metadata::commit_sha1().empty() ? "unknown" : version::git_metadata::commit_sha1(),
        version::version);

    // output the actual (performance) statistics
    std::unordered_multimap<std::string, std::string>::iterator group_iter;  // iterate over all groups
    std::unordered_multimap<std::string, std::string>::iterator entry_iter;  // iterate over all entries in a specific group
    for (group_iter = tracking_statistics.begin(); group_iter != tracking_statistics.end(); group_iter = entry_iter) {
        // get the current group
        const std::string &group = group_iter->first;
        // find the range of all entries in the current group
        const std::pair key_range = tracking_statistics.equal_range(group);

        // output the group name, if it is not the empty string
        if (!group.empty()) {
            out << group << ":\n";
        }
        // output all performance statistic entries of the current group
        for (entry_iter = key_range.first; entry_iter != key_range.second; ++entry_iter) {
            out << fmt::format("{}", entry_iter->second);
        }
        out << '\n';
    }
}

}  // namespace plssvm::detail