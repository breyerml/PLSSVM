/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/kernel_function_types.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include "fmt/core.h"  // fmt::format

#include <ios>          // std::ios::failbit
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const kernel_function_type kernel) {
    switch (kernel) {
        case kernel_function_type::linear:
            return out << "linear";
        case kernel_function_type::polynomial:
            return out << "polynomial";
        case kernel_function_type::rbf:
            return out << "rbf";
    }
    return out << "unknown";
}

std::string_view kernel_function_type_to_math_string(const kernel_function_type kernel) noexcept {
    switch (kernel) {
        case kernel_function_type::linear:
            return "u'*v";
        case kernel_function_type::polynomial:
            return "(gamma*u'*v+coef0)^degree";
        case kernel_function_type::rbf:
            return "exp(-gamma*|u-v|^2)";
    }
    return "unknown";
}

std::istream &operator>>(std::istream &in, kernel_function_type &kernel) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "linear" || str == "0") {
        kernel = kernel_function_type::linear;
    } else if (str == "polynomial" || str == "poly" || str == "1") {
        kernel = kernel_function_type::polynomial;
    } else if (str == "rbf" || str == "2") {
        kernel = kernel_function_type::rbf;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm
