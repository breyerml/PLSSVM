/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all possible classification types, i.e., one vs. all (OAA) and one vs. one (OAO).
 */

#ifndef PLSSVM_CLASSIFICATION_TYPES_HPP_
#define PLSSVM_CLASSIFICATION_TYPES_HPP_
#pragma once

#include <iosfwd>       // forward declare std::ostream and std::istream
#include <string_view>  // std::string_view

namespace plssvm {

enum class classification_type {
    /** Use the one vs. all classification strategy for the multi-class SVM (default). */
    oaa,
    /** Use the one vs. one classification strategy for the multi-class SVM. */
    oao
};

/**
 * @brief Output the @p classification to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the classification type to
 * @param[in] classification the classification type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, classification_type classification);

/**
 * @brief Use the input-stream @p in to initialize the @p classification type.
 * @param[in,out] in input-stream to extract the classification type from
 * @param[in] classification the classification type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, classification_type &classification);

/**
 * @brief In contrast to operator>> return the full name of the provided @p classification strategy.
 * @param[in] classification the multi-class classification strategy
 * @return the full name of the multi-class classification strategy (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view classification_type_to_full_string(classification_type classification);

}  // namespace plssvm

#endif  // PLSSVM_CLASSIFICATION_TYPES_HPP_
