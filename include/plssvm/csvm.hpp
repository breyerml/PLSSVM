/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends and implements the functionality shared by all of them.
 */

#pragma once

#include "plssvm/kernel_types.hpp"     // plssvm::kernel_type
#include "plssvm/parameter.hpp"        // plssvm::parameter
#include "plssvm/target_platform.hpp"  // plssvm::target_platform

#include <cstddef>      // std::size_t
#include <memory>       // std::shared_ptr
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

namespace plssvm {

/**
 * @brief Base class for all C-SVM backends.
 * @tparam T the type of the data
 */
template <typename T>
class csvm {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;
    /// Unsigned integer type.
    using size_type = std::size_t;

    //*************************************************************************************************************************************//
    //                                                      special member functions                                                       //
    //*************************************************************************************************************************************//
    /**
     * @brief Construct a new C-SVM with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit csvm(const parameter<T> &params);

    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~csvm() = default;

    /**
     * @brief Disable copy-constructor.
     */
    csvm(const csvm &) = delete;
    // clang-format off
    /**
     * @brief Explicitly allow move-construction.
     */
    csvm(csvm &&) noexcept = default;
    // clang-format on

    //*************************************************************************************************************************************//
    //                                                             IO functions                                                            //
    //*************************************************************************************************************************************//

    /**
     * @brief Write the calculated model to the given file.
     * @details Writes the model using the libsvm format:
     * @code
     * svm_type c_svc
     * kernel_type linear
     * nr_class 2
     * total_sv 5
     * rho 0.37332362
     * label 1 -1
     * nr_sv 2 3
     * SV
     * -0.17609704 0:-1.117828e+00 1:-2.908719e+00 2:6.663834e-01 3:1.097883e+00
     * 0.883819 0:-5.282118e-01 1:-3.358810e-01 2:5.168729e-01 3:5.460446e-01
     * -0.47971326 0:-2.098121e-01 1:6.027694e-01 2:-1.308685e-01 3:1.080525e-01
     * -0.23146635 0:5.765022e-01 1:1.014056e+00 2:1.300943e-01 3:7.261914e-01
     * 0.0034576654 0:1.884940e+00 1:1.005186e+00 2:2.984999e-01 3:1.646463e+00
     * @endcode
     * @throws unsupported_kernel_type_exception if the kernel_type cannot be recognized
     * @param[in] filename name of the file to write the model information to
     */
    void write_model(const std::string &filename);

    //*************************************************************************************************************************************//
    //                                                             learn model                                                             //
    //*************************************************************************************************************************************//
    /**
     * @brief Learns the Support Vectors given the data in the provided parameter class.
     * @details Performs 2 steps:
     * 1. Load the data onto the used device (e.g. one or more GPUs)
     * 2. Learn the model by solving a minimization problem using the Conjugated Gradients algorithm
     */
    void learn();
    // TODO: absolute vs relative residual

    //*************************************************************************************************************************************//
    //                                                               predict                                                               //
    //*************************************************************************************************************************************//

    /**
     * @brief Evaluates the model on the data used for training.
     * @return The fraction of correct labeled training data in percent. ([[nodiscard]])
     */
    [[nodiscard]] real_type accuracy();

    /**
     * @brief Uses the already learned model to predict the class of a (new) data point.
     * @param[in] point the data point to predict
     * @return a negative `real_type` value if the prediction for data point point is the negative class and a positive `real_type` value otherwise ([[nodiscard]])
     */
    [[nodiscard]] real_type predict(const std::vector<real_type> &point);

    /**
     * @brief Uses the already learned model to predict the class of an (new) point
     * @param[in] point the data point to predict
     * @return -1.0 if the prediction for point is the negative class and +1 otherwise ([[nodiscard]])
     */
    [[nodiscard]] real_type predict_label(const std::vector<real_type> &point);
    /**
     * @brief Uses the already learned model to predict the class of multiple (new) points
     * @param[in] points the points to predict
     * @return a `std::vector<real_type>` filled with -1 for each prediction for a data point the negative class and +1 otherwise ([[nodiscard]])
     */
    [[nodiscard]] std::vector<real_type> predict_label(const std::vector<std::vector<real_type>> &points);

  protected:
    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    /**
     * @brief Initialize the data on the respective device(s) (e.g. GPUs).
     */
    virtual void setup_data_on_device() = 0;
    /**
     * @brief Generate the vector `q`, a subvector of the least-squares matrix equation.
     * @return the generated `q` vector
     */
    [[nodiscard]] virtual std::vector<real_type> generate_q() = 0;
    /**
     * @brief Solves the equation \f$Ax = b\f$ using the Conjugated Gradients algorithm.
     * @details Solves using a slightly modified version of the CG algorithm described by [Jonathan Richard Shewchuk](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf):
     * \image html cg.png
     * @param[in] b the right-hand side of the equation \f$Ax = b\f$
     * @param[in] imax the maximum number of CG iterations
     * @param[in] eps error tolerance
     * @param[in] q subvector of the least-squares matrix equation
     * @return the alpha values
     */
    virtual std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) = 0;
    /**
     * @brief updates the `w_` vector to the current data and alpha values.
     */
    virtual void update_w() = 0;
    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @param[in] points the data points to predict
     * @return a `std::vector<real_type>` filled with negative values for each prediction for a data point with the negative class and positive values otherwise ([[nodiscard]])
     */
    [[nodiscard]] virtual std::vector<real_type> predict(const std::vector<std::vector<real_type>> &points) = 0;

    //*************************************************************************************************************************************//
    //                                                          kernel functions                                                           //
    //*************************************************************************************************************************************//
    /**
     * @brief Computes the value of the two vectors @p xi and @p xj using the kernel function specified during construction.
     * @param[in] xi the first vector
     * @param[in] xj the second vector
     * @throws unsupported_kernel_type_exception if the kernel_type cannot be recognized
     * @return the value computed by the kernel function
     */
    real_type kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj);

    /**
     * @brief Transforms the 2D data from AoS to a 1D SoA layout, ignoring the last data point and adding boundary points.
     * @param[in] matrix the 2D vector to be transformed into a 1D representation
     * @param[in] boundary the number of boundary cells
     * @param[in] num_points the number of data points of the 2D vector to transform
     * @attention boundary values can contain random numbers
     * @return an 1D vector in a SoA layout
     */
    std::vector<real_type> transform_data(const std::vector<std::vector<real_type>> &matrix, size_type boundary, size_type num_points);

    //*************************************************************************************************************************************//
    //                                              parameter initialized by the constructor                                               //
    //*************************************************************************************************************************************//
    /// The target platform.
    const target_platform target_;
    /// The used kernel function: linear, polynomial or radial basis functions (rbf).
    const kernel_type kernel_;
    /// The degree parameter used in the polynomial kernel function.
    const int degree_;
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    real_type gamma_;
    /// The coef0 parameter used in the polynomial kernel function.
    const real_type coef0_;
    /// The cost parameter in the C-SVM.
    real_type cost_;
    /// The error tolerance parameter for the CG algorithm.
    const real_type epsilon_;
    /// If `true` additional information (e.g. timing information) will be printed during execution.
    const bool print_info_;

    /// The data used the train the SVM.
    const std::shared_ptr<const std::vector<std::vector<real_type>>> data_ptr_{};
    /// The labels associated to each data point.
    std::shared_ptr<const std::vector<real_type>> value_ptr_{};
    /// The result of the CG calculation.
    std::shared_ptr<const std::vector<real_type>> alpha_ptr_{};

    //*************************************************************************************************************************************//
    //                                                         internal variables                                                          //
    //*************************************************************************************************************************************//
    /// The number of data points in the data set.
    size_type num_data_points_{};
    /// The number of features per data point.
    size_type num_features_{};
    /// The bias after learning.
    real_type bias_{};
    /// The bottom right matrix entry multiplied by cost.
    real_type QA_cost_{};
    /// The normal vector used for speeding up the prediction in case of the linear kernel function.
    std::vector<real_type> w_{};
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace plssvm
