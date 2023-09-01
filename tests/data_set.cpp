/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the data_set used for learning an SVM model.
 */

#include "plssvm/data_set.hpp"

#include "plssvm/constants.hpp"                 // plssvm::real_type
#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::data_set_exception
#include "plssvm/file_format_types.hpp"         // plssvm::file_format_type

#include "custom_test_macros.hpp"               // EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_EQ, EXPECT_FLOATING_POINT_NEAR, EXPECT_FLOATING_POINT_2D_VECTOR_EQ, EXPECT_FLOATING_POINT_2D_VECTOR_NEAR
#include "naming.hpp"                           // naming::real_type_label_type_combination_to_name
#include "types_to_test.hpp"                    // util::{real_type_label_type_combination_gtest}
#include "utility.hpp"                          // util::{temporary_file, redirect_output, instantiate_template_file, get_distinct_label, get_correct_data_file_labels, generate_specific_matrix}

#include "gmock/gmock-matchers.h"               // EXPECT_THAT, ::testing::{ContainsRegex, StartsWith}
#include "gtest/gtest.h"                        // EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, ASSERT_EQ, TEST, TYPED_TEST, TYPED_TEST_SUITE, ::testing::Test

#include <cstddef>                              // std::size_t
#include <regex>                                // std::regex, std::regex::extended, std::regex_match
#include <string>                               // std::string
#include <string_view>                          // std::string_view
#include <tuple>                                // std::ignore, std::get
#include <type_traits>                          // std::is_same_v, std::is_integral_v
#include <vector>                               // std::vector

//*************************************************************************************************************************************//
//                                                         scaling nested-class                                                        //
//*************************************************************************************************************************************//

template <typename T>
class DataSetScaling : public ::testing::Test, private util::redirect_output<> {};
TYPED_TEST_SUITE(DataSetScaling, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(DataSetScaling, default_construct_factor) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factor_type = typename scaling_type::factors;

    // create factor
    const factor_type factor{};

    // test values
    EXPECT_EQ(factor.feature, std::size_t{});
    EXPECT_FLOATING_POINT_EQ(factor.lower, plssvm::real_type{});
    EXPECT_FLOATING_POINT_EQ(factor.upper, plssvm::real_type{});
}
TYPED_TEST(DataSetScaling, construct_factor) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factor_type = typename scaling_type::factors;

    // create factor
    const factor_type factor{ 1, plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 } };

    // test values
    EXPECT_EQ(factor.feature, 1);
    EXPECT_FLOATING_POINT_EQ(factor.lower, plssvm::real_type{ -2.5 });
    EXPECT_FLOATING_POINT_EQ(factor.upper, plssvm::real_type{ 2.5 });
}

TYPED_TEST(DataSetScaling, construct_interval) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create scaling class
    const scaling_type scale{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };

    // test whether the values have been correctly set
    EXPECT_FLOATING_POINT_EQ(scale.scaling_interval.first, plssvm::real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(scale.scaling_interval.second, plssvm::real_type{ 1.0 });
    EXPECT_TRUE(scale.scaling_factors.empty());
}
TYPED_TEST(DataSetScaling, construct_invalid_interval) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create scaling class with an invalid interval
    EXPECT_THROW_WHAT((scaling_type{ plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } }),
                      plssvm::data_set_exception,
                      "Inconsistent scaling interval specification: lower (1) must be less than upper (-1)!");
}
TYPED_TEST(DataSetScaling, construct_from_file) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create scaling class
    scaling_type scale{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // test whether the values have been correctly set
    EXPECT_EQ(scale.scaling_interval.first, plssvm::detail::convert_to<plssvm::real_type>("-1.4"));
    EXPECT_EQ(scale.scaling_interval.second, plssvm::detail::convert_to<plssvm::real_type>("2.6"));
    const std::vector<factors_type> correct_factors = {
        factors_type{ 0, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } },
        factors_type{ 1, plssvm::real_type{ 1.1 }, plssvm::real_type{ 2.1 } },
        factors_type{ 3, plssvm::real_type{ 3.3 }, plssvm::real_type{ 4.3 } },
        factors_type{ 4, plssvm::real_type{ 4.4 }, plssvm::real_type{ 5.4 } },
    };
    ASSERT_EQ(scale.scaling_factors.size(), correct_factors.size());
    for (std::size_t i = 0; i < correct_factors.size(); ++i) {
        EXPECT_EQ(scale.scaling_factors[i].feature, correct_factors[i].feature);
        EXPECT_FLOATING_POINT_EQ(scale.scaling_factors[i].lower, correct_factors[i].lower);
        EXPECT_FLOATING_POINT_EQ(scale.scaling_factors[i].upper, correct_factors[i].upper);
    }
}

TYPED_TEST(DataSetScaling, save) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create scaling class
    const scaling_type scale{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save scaling factors
    scale.save(tmp_file.filename);

    // read file and check its content
    plssvm::detail::io::file_reader reader{ tmp_file.filename };
    reader.read_lines('#');

    // check file content
    ASSERT_GE(reader.num_lines(), 2);
    EXPECT_EQ(reader.line(0), "x");
    std::regex reg{ "[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    EXPECT_TRUE(std::regex_match(std::string{ reader.line(1) }, reg));
    reg = std::regex{ "\\+?[1-9]+[0-9]* [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    for (std::size_t i = 2; i < reader.num_lines(); ++i) {
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
    }
}
TYPED_TEST(DataSetScaling, save_empty_scaling_factors) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create scaling class
    const scaling_type scale{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };

    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save scaling factors
    scale.save(tmp_file.filename);

    // read file and check its content
    plssvm::detail::io::file_reader reader{ tmp_file.filename };
    reader.read_lines('#');

    // check the content
    ASSERT_EQ(reader.num_lines(), 2);
    EXPECT_EQ(reader.line(0), "x");
    const std::regex reg{ "[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    EXPECT_TRUE(std::regex_match(std::string{ reader.line(1) }, reg));
}

//*************************************************************************************************************************************//
//                                                      label mapper nested-class                                                      //
//*************************************************************************************************************************************//

template <typename T>
class DataSetLabelMapper : public ::testing::Test {};
TYPED_TEST_SUITE(DataSetLabelMapper, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(DataSetLabelMapper, construct) {
    using label_type = TypeParam;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> distinct_labels = util::get_distinct_label<label_type>();

    // create label mapper
    const label_mapper_type mapper{ distinct_labels };

    // test values
    EXPECT_EQ(mapper.num_mappings(), distinct_labels.size());
    EXPECT_EQ(mapper.labels(), distinct_labels);
    // test mapping
    for (std::size_t i = 0; i < distinct_labels.size(); ++i) {
        EXPECT_EQ(mapper.get_label_by_mapped_index(i), distinct_labels[i]);
        EXPECT_EQ(mapper.get_mapped_index_by_label(distinct_labels[i]), i);
    }
}
TYPED_TEST(DataSetLabelMapper, get_mapped_index_by_label) {
    using label_type = TypeParam;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> distinct_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();

    // create label mapper
    const label_mapper_type mapper{ distinct_labels };

    // test the number of mappings
    ASSERT_EQ(mapper.num_mappings(), distinct_labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const std::size_t label_idx = std::distance(distinct_labels.cbegin(), std::find(distinct_labels.cbegin(), distinct_labels.cend(), labels[i]));
        EXPECT_EQ(mapper.get_mapped_index_by_label(labels[i]), label_idx);
    }
}
TYPED_TEST(DataSetLabelMapper, get_mapped_index_by_invalid_label) {
    using label_type = TypeParam;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> distinct_labels = util::get_distinct_label<label_type>();

    // create label mapper
    const label_mapper_type mapper{ distinct_labels };

    // test the number of mappings
    if constexpr (!std::is_same_v<label_type, bool>) {
        // can't have an unknown labels for bool
        EXPECT_THROW_WHAT(std::ignore = mapper.get_mapped_index_by_label(plssvm::detail::convert_to<label_type>("9")),
                          plssvm::data_set_exception,
                          R"(Label "9" unknown in this label mapping!)");
    } else {
        SUCCEED() << "By definition there can't be unknown labels for the boolean label type.";
    }
}
TYPED_TEST(DataSetLabelMapper, get_label_by_mapped_index) {
    using label_type = TypeParam;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> distinct_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();

    // create label mapper
    const label_mapper_type mapper{ distinct_labels };

    // test the number of mappings
    ASSERT_EQ(mapper.num_mappings(), distinct_labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const std::size_t label_idx = std::distance(distinct_labels.cbegin(), std::find(distinct_labels.cbegin(), distinct_labels.cend(), labels[i]));
        EXPECT_EQ(mapper.get_label_by_mapped_index(label_idx), labels[i]);
    }
}
TYPED_TEST(DataSetLabelMapper, get_label_by_invalid_mapped_index) {
    using label_type = TypeParam;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> distinct_labels = util::get_distinct_label<label_type>();

    // create label mapper
    const label_mapper_type mapper{ distinct_labels };

    // test the number of mappings
    EXPECT_THROW_WHAT(std::ignore = mapper.get_label_by_mapped_index(mapper.num_mappings() + 1),
                      plssvm::data_set_exception,
                      fmt::format("Mapped index \"{}\" unknown in this label mapping!", mapper.num_mappings() + 1));
}
TYPED_TEST(DataSetLabelMapper, num_mappings) {
    using label_type = TypeParam;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_EQ(mapper.num_mappings(), different_labels.size());
}
TYPED_TEST(DataSetLabelMapper, labels) {
    using label_type = TypeParam;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the different labels
    EXPECT_EQ(mapper.labels(), different_labels);
}

template <typename T>
class DataSetLabelMapperDeathTest : public DataSetLabelMapper<T> {};
TYPED_TEST_SUITE(DataSetLabelMapperDeathTest, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(DataSetLabelMapperDeathTest, duplicated_labels) {
    using label_type = TypeParam;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // duplicated labels are not allowed in the label_mapper constructor
    EXPECT_DEATH(label_mapper_type{ util::get_correct_data_file_labels<label_type>() },
                 "The provided labels for the label_mapper must not include duplicated ones!");
}

//*************************************************************************************************************************************//
//                                                           data set class                                                            //
//*************************************************************************************************************************************//

template <typename T>
class DataSet : public ::testing::Test, private util::redirect_output<>, protected util::temporary_file {
    using type = plssvm::real_type;
  protected:
    const std::vector<std::vector<type>> correct_template_file_data_points{
        { type{ -1.117827500607882 }, type{ -2.9087188881250993 }, type{ 0.66638344270039144 }, type{ 1.0978832703949288 } },
        { type{ -0.5282118298909262 }, type{ -0.335880984968183973 }, type{ 0.51687296029754564 }, type{ 0.54604461446026 } },
        { type{ 0.57650218263054642 }, type{ 1.01405596624706053 }, type{ 0.13009428079760464 }, type{ 0.7261913886869387 } },
        { type{ -0.20981208921241892 }, type{ 0.60276937379453293 }, type{ -0.13086851759108944 }, type{ 0.10805254527169827 } },
        { type{ 1.88494043717792 }, type{ 1.00518564317278263 }, type{ 0.298499933047586044 }, type{ 1.6464627048813514 } },
        { type{ -1.1256816275635 }, type{ 2.12541534341344414 }, type{ -0.165126576545454511 }, type{ 2.5164553141200987 } }
    };
};
TYPED_TEST_SUITE(DataSet, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(DataSet, typedefs) {
    using label_type = TypeParam;

    // create a data_set using an existing LIBSVM data set file
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<label_type> data{ this->filename };

    // test internal typedefs
    ::testing::StaticAssertTypeEq<label_type, typename decltype(data)::label_type>();
    EXPECT_TRUE(std::is_integral_v<typename decltype(data)::size_type>);
}

TYPED_TEST(DataSet, construct_arff_from_file_with_label) {
    using label_type = TypeParam;

    // must append .arff to filename so that the correct function is called in the data_set constructor
    this->filename.append(".arff");

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/6x4_TEMPLATE.arff", this->filename);
    const plssvm::data_set<label_type> data{ this->filename };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), plssvm::aos_matrix<plssvm::real_type>{ this->correct_template_file_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    EXPECT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->correct_template_file_data_points.size());
    EXPECT_EQ(data.num_features(), this->correct_template_file_data_points.front().size());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSet, construct_arff_from_file_without_label) {
    using label_type = TypeParam;

    // create data set
    const plssvm::data_set<label_type> data{ PLSSVM_TEST_PATH "/data/arff/3x2_without_label.arff" };

    // check values
    const std::vector<std::vector<plssvm::real_type>> correct_data = {
        { plssvm::real_type{ 1.5 }, plssvm::real_type{ -2.9 } },
        { plssvm::real_type{ 0.0 }, plssvm::real_type{ -0.3 } },
        { plssvm::real_type{ 5.5 }, plssvm::real_type{ 0.0 } }
    };
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), plssvm::aos_matrix<plssvm::real_type>{ correct_data });
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), 3);
    EXPECT_EQ(data.num_features(), 2);
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(DataSet, construct_libsvm_from_file_with_label) {
    using label_type = TypeParam;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<label_type> data{ this->filename };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), plssvm::aos_matrix<plssvm::real_type>{ this->correct_template_file_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    EXPECT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->correct_template_file_data_points.size());
    EXPECT_EQ(data.num_features(), this->correct_template_file_data_points.front().size());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSet, construct_libsvm_from_file_without_label) {
    using label_type = TypeParam;

    // create data set
    const plssvm::data_set<label_type> data{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };

    // check values
    const std::vector<std::vector<plssvm::real_type>> correct_data = {
        { plssvm::real_type{ 1.5 }, plssvm::real_type{ -2.9 } },
        { plssvm::real_type{ 0.0 }, plssvm::real_type{ -0.3 } },
        { plssvm::real_type{ 5.5 }, plssvm::real_type{ 0.0 } }
    };
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), plssvm::aos_matrix<plssvm::real_type>{ correct_data });
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), 3);
    EXPECT_EQ(data.num_features(), 2);
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(DataSet, construct_explicit_arff_from_file) {
    using label_type = TypeParam;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/6x4_TEMPLATE.arff", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, plssvm::file_format_type::arff };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), plssvm::aos_matrix<plssvm::real_type>{ this->correct_template_file_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    EXPECT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->correct_template_file_data_points.size());
    EXPECT_EQ(data.num_features(), this->correct_template_file_data_points.front().size());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSet, construct_explicit_libsvm_from_file) {
    using label_type = TypeParam;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, plssvm::file_format_type::libsvm };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), plssvm::aos_matrix<plssvm::real_type>{ this->correct_template_file_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    EXPECT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->correct_template_file_data_points.size());
    EXPECT_EQ(data.num_features(), this->correct_template_file_data_points.front().size());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(DataSet, construct_scaled_arff_from_file) {
    using label_type = TypeParam;

    // must append .arff to filename so that the correct function is called in the data_set constructor
    this->filename.append(".arff");

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/6x4_TEMPLATE.arff", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->correct_template_file_data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::aos_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    EXPECT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->correct_template_file_data_points.size());
    EXPECT_EQ(data.num_features(), this->correct_template_file_data_points.front().size());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}
TYPED_TEST(DataSet, construct_scaled_libsvm_from_file) {
    using label_type = TypeParam;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, { plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->correct_template_file_data_points, plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::aos_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    EXPECT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->correct_template_file_data_points.size());
    EXPECT_EQ(data.num_features(), this->correct_template_file_data_points.front().size());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

TYPED_TEST(DataSet, construct_scaled_explicit_arff_from_file) {
    using label_type = TypeParam;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/6x4_TEMPLATE.arff", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, plssvm::file_format_type::arff, { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->correct_template_file_data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::aos_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    EXPECT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->correct_template_file_data_points.size());
    EXPECT_EQ(data.num_features(), this->correct_template_file_data_points.front().size());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}
TYPED_TEST(DataSet, construct_scaled_explicit_libsvm_from_file) {
    using label_type = TypeParam;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, plssvm::file_format_type::libsvm, { plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->correct_template_file_data_points, plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::aos_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    EXPECT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->correct_template_file_data_points.size());
    EXPECT_EQ(data.num_features(), this->correct_template_file_data_points.front().size());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

TYPED_TEST(DataSet, scale_too_many_factors) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    // create (invalid) scaling factors
    scaling_type scaling{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };
    scaling.scaling_factors = std::vector<factors_type>{
        factors_type{ 0, 0.0, 0.1 },
        factors_type{ 1, 1.0, 1.1 },
        factors_type{ 2, 2.0, 2.1 },
        factors_type{ 3, 3.0, 3.1 },
        factors_type{ 4, 4.0, 4.1 }
    };

    // try creating a data set with invalid scaling factors
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ this->filename, scaling }),
                      plssvm::data_set_exception,
                      "Need at most as much scaling factors as features in the data set are present (4), but 5 were given!");
}
TYPED_TEST(DataSet, scale_invalid_feature_index) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    // create (invalid) scaling factors
    scaling_type scaling{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };
    scaling.scaling_factors = std::vector<factors_type>{
        factors_type{ 4, 4.0, 4.1 },
        factors_type{ 2, 2.0, 2.1 }
    };

    // try creating a data set with invalid scaling factors
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ this->filename, scaling }),
                      plssvm::data_set_exception,
                      "The maximum scaling feature index most not be greater than 3, but is 4!");
}
TYPED_TEST(DataSet, scale_duplicate_feature_index) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    // create (invalid) scaling factors
    scaling_type scaling{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };
    scaling.scaling_factors = std::vector<factors_type>{
        factors_type{ 1, 1.0, 1.1 },
        factors_type{ 2, 2.0, 2.1 },
        factors_type{ 3, 3.0, 3.1 },
        factors_type{ 2, 2.0, 2.1 }
    };

    // try creating a data set with invalid scaling factors
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ this->filename, scaling }),
                      plssvm::data_set_exception,
                      "Found more than one scaling factor for the feature index 2!");
}

TYPED_TEST(DataSet, construct_from_vector_without_label) {
    using label_type = TypeParam;

    // create data points
    const auto correct_data_points = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(4, 4);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), correct_data_points);
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSet, construct_from_empty_vector) {
    using label_type = TypeParam;

    // create data points
    const plssvm::aos_matrix<plssvm::real_type> correct_data_points{};

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ correct_data_points }),
                      plssvm::data_set_exception,
                      "Data vector is empty!");
}
TYPED_TEST(DataSet, construct_from_vector_with_differing_num_features) {
    using label_type = TypeParam;

    // create data points
    const std::vector<std::vector<plssvm::real_type>> correct_data_points = {
        { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.1 } },
        { plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.1 }, plssvm::real_type{ 1.2 } }
    };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ correct_data_points }),
                      plssvm::matrix_exception,
                      "Each row in the matrix must contain the same amount of columns!");
}
TYPED_TEST(DataSet, construct_from_vector_with_no_features) {
    using label_type = TypeParam;

    // create data points
    const std::vector<std::vector<plssvm::real_type>> correct_data_points = { {}, {} };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ correct_data_points }),
                      plssvm::matrix_exception,
                      "The data to create the matrix must at least have one column!");
}

TYPED_TEST(DataSet, construct_from_vector_with_label) {
    using label_type = TypeParam;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(labels.size(), 4);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points, labels };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), correct_data_points);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), labels);
    EXPECT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSet, construct_from_vector_mismatching_num_data_points_and_labels) {
    using label_type = TypeParam;

    // create data points and labels
    const auto correct_data_points = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(4, 4);
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();

    // create data set
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ correct_data_points,
                                                     std::vector<label_type>{ labels } }),
                      plssvm::data_set_exception,
                      fmt::format("Number of labels ({}) must match the number of data points ({})!", labels.size(), correct_data_points.num_rows()));
}

TYPED_TEST(DataSet, construct_scaled_from_vector_without_label) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create data points
    const auto data_points = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(4, 4);

    // create data set
    const plssvm::data_set<label_type> data{ data_points, scaling_type{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), correct_data_points_scaled);
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}
TYPED_TEST(DataSet, construct_scaled_from_vector_with_label) {
    using label_type = TypeParam;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(labels.size(), 4, plssvm::real_type{ -2.0 }, plssvm::real_type{ 2.0 });

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points, labels, { -1.0, 1.0 } };

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(correct_data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), correct_data_points_scaled);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), labels);
    EXPECT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

template <typename TypeParam>
class DataSetSave : public ::testing::Test, private util::redirect_output<>, protected util::temporary_file {
  protected:
    using label_type = TypeParam;

    const std::vector<label_type> label{ util::get_correct_data_file_labels<label_type>() };
    const plssvm::aos_matrix<plssvm::real_type> data_points{ util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(label.size(), 4) };
};
TYPED_TEST_SUITE(DataSetSave, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(DataSetSave, save_invalid_automatic_format) {
    using label_type = TypeParam;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->data_points, this->label };

    // try to save to temporary file with an unrecognized extension
    EXPECT_THROW_WHAT(data.save("test.txt");,
                      plssvm::data_set_exception,
                      "Unrecognized file extension for file \"test.txt\" (must be one of: .libsvm or .arff)!");
}

TYPED_TEST(DataSetSave, save_libsvm_with_label) {
    using label_type = TypeParam;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->data_points, this->label };
    // save to temporary file
    data.save(this->filename, plssvm::file_format_type::libsvm);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // create regex to check for the correct output
    ASSERT_EQ(reader.num_lines(), this->data_points.num_rows());
    const std::regex reg{ ".+ ([0-9]*:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?){4}", std::regex::extended };
    for (const std::string_view line : reader.lines()) {
        EXPECT_TRUE(std::regex_match(std::string{ line }, reg));
    }
}
TYPED_TEST(DataSetSave, save_libsvm_automatic_format) {
    using label_type = TypeParam;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->data_points, this->label };
    // rename temporary such that it ends with .libsvm
    const std::string old_filename = this->filename;
    this->filename += ".libsvm";
    std::filesystem::rename(old_filename, this->filename);
    // save to temporary file
    data.save(this->filename);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // create regex to check for the correct output
    ASSERT_EQ(reader.num_lines(), this->data_points.num_rows());
    const std::regex reg{ ".+ ([0-9]*:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?){4}", std::regex::extended };
    for (const std::string_view line : reader.lines()) {
        EXPECT_TRUE(std::regex_match(std::string{ line }, reg));
    }
}
TYPED_TEST(DataSetSave, save_libsvm_without_label) {
    using label_type = TypeParam;

    // create data set without labels
    const plssvm::data_set<label_type> data{ this->data_points };
    // save to temporary file
    data.save(this->filename, plssvm::file_format_type::libsvm);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // create regex to check for the correct output
    ASSERT_EQ(reader.num_lines(), this->data_points.num_rows());
    const std::regex reg{ "([0-9]*:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?){4}", std::regex::extended };
    for (const std::string_view line : reader.lines()) {
        EXPECT_TRUE(std::regex_match(std::string{ line }, reg));
    }
}

TYPED_TEST(DataSetSave, save_arff_with_label) {
    using label_type = TypeParam;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->data_points, this->label };
    // save to temporary file
    data.save(this->filename, plssvm::file_format_type::arff);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');

    // create regex to check for the correct output
    const std::size_t num_features = this->data_points.num_cols();
    const std::size_t expected_header_size = num_features + 3;  // num_features + @RELATION + class + @DATA
    ASSERT_EQ(reader.num_lines(), expected_header_size + this->data_points.num_rows());
    // check header
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(0)), ::testing::StartsWith("@relation"));
    for (std::size_t i = 0; i < num_features; ++i) {
        EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(i + 1)), ::testing::ContainsRegex("@attribute .* numeric"));
    }
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(1 + num_features)), ::testing::ContainsRegex("@attribute class \\{.*,.*\\}"));
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(1 + num_features + 1)), ::testing::StartsWith("@data"));
    // check data points
    const std::regex reg{ "([-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?,){4}.+", std::regex::extended };
    for (std::size_t i = expected_header_size; i < reader.num_lines(); ++i) {
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
    }
}
TYPED_TEST(DataSetSave, save_arff_automatic_format) {
    using label_type = TypeParam;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->data_points, this->label };
    // rename temporary such that it ends with .arff
    const std::string old_filename = this->filename;
    this->filename += ".arff";
    std::filesystem::rename(old_filename, this->filename);
    // save to temporary file
    data.save(this->filename);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');

    // create regex to check for the correct output
    const std::size_t num_features = this->data_points.num_cols();
    const std::size_t expected_header_size = num_features + 3;  // num_features + @RELATION + class + @DATA
    ASSERT_EQ(reader.num_lines(), expected_header_size + this->data_points.num_rows());
    // check header
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(0)), ::testing::StartsWith("@relation"));
    for (std::size_t i = 0; i < num_features; ++i) {
        EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(i + 1)), ::testing::ContainsRegex("@attribute .* numeric"));
    }
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(1 + num_features)), ::testing::ContainsRegex("@attribute class \\{.*,.*\\}"));
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(1 + num_features + 1)), ::testing::StartsWith("@data"));
    // check data points
    const std::regex reg{ "([-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?,){4}.+", std::regex::extended };
    for (std::size_t i = expected_header_size; i < reader.num_lines(); ++i) {
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
    }
}
TYPED_TEST(DataSetSave, save_arff_without_label) {
    using label_type = TypeParam;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->data_points };
    // save to temporary file
    data.save(this->filename, plssvm::file_format_type::arff);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');

    // create regex to check for the correct output
    const std::size_t num_features = this->data_points.num_cols();
    const std::size_t expected_header_size = num_features + 2;  // num_features + @RELATION + @DATA
    ASSERT_EQ(reader.num_lines(), expected_header_size + this->data_points.num_rows());
    // check header
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(0)), ::testing::StartsWith("@relation"));
    for (std::size_t i = 0; i < num_features; ++i) {
        EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(i + 1)), ::testing::ContainsRegex("@attribute .* numeric"));
    }
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(1 + num_features)), ::testing::StartsWith("@data"));
    // check data points
    const std::regex reg{ "([-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?,){3}[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    for (std::size_t i = expected_header_size; i < reader.num_lines(); ++i) {
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
    }
}

template <typename TypeParam>
class DataSetGetter : public ::testing::Test, private util::redirect_output<> {
  protected:
    using label_type = TypeParam;

    std::string filename;
    const std::vector<label_type> different_label{ util::get_distinct_label<label_type>() };
    const std::vector<label_type> label{ util::get_correct_data_file_labels<label_type>() };
    const plssvm::aos_matrix<plssvm::real_type> data_points{ util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(label.size(), 4) };
};
TYPED_TEST_SUITE(DataSetGetter, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(DataSetGetter, data) {
    using label_type = TypeParam;

    // create data set without labels
    const plssvm::data_set<label_type> data{ this->data_points };
    // check data getter
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), this->data_points);
}
TYPED_TEST(DataSetGetter, has_labels) {
    using label_type = TypeParam;

    // create data set without labels
    const plssvm::data_set< label_type> data_without_labels{ this->data_points };
    // check has_labels getter
    EXPECT_FALSE(data_without_labels.has_labels());
    // create data set with labels
    const plssvm::data_set<label_type> data_with_labels{ this->data_points, this->label };
    // check has_labels getter
    EXPECT_TRUE(data_with_labels.has_labels());
}
TYPED_TEST(DataSetGetter, labels) {
    using label_type = TypeParam;

    // create data set without labels
    const plssvm::data_set<label_type> data_without_labels{ this->data_points };
    // check labels getter
    EXPECT_FALSE(data_without_labels.labels().has_value());
    // create data set with labels
    const plssvm::data_set<label_type> data_with_labels{ this->data_points, this->label };
    // check labels getter
    EXPECT_TRUE(data_with_labels.labels().has_value());
    EXPECT_EQ(data_with_labels.labels().value().get(), this->label);
}
TYPED_TEST(DataSetGetter, different_labels) {
    using label_type = TypeParam;

    // create data set without labels
    const plssvm::data_set<label_type> data_without_labels{ this->data_points };
    // check different_labels getter
    EXPECT_FALSE(data_without_labels.classes().has_value());
    // create data set with labels
    const plssvm::data_set<label_type> data_with_labels{ this->data_points, this->label };
    // check different_labels getter
    EXPECT_TRUE(data_with_labels.classes().has_value());
    EXPECT_EQ(data_with_labels.classes().value(), this->different_label);
}
TYPED_TEST(DataSetGetter, num_data_points) {
    using label_type = TypeParam;

    // create data set
    const plssvm::data_set<label_type> data{ this->data_points };
    // check num_data_points getter
    EXPECT_EQ(data.num_data_points(), this->data_points.num_rows());
}
TYPED_TEST(DataSetGetter, num_features) {
    using label_type = TypeParam;

    // create data set
    const plssvm::data_set<label_type> data{ this->data_points };
    // check num_features getter
    EXPECT_EQ(data.num_features(), this->data_points.num_cols());
}
TYPED_TEST(DataSetGetter, num_different_labels) {
    using label_type = TypeParam;

    // create data set without labels
    const plssvm::data_set<label_type> data_without_label{ this->data_points };
    // check num_different_labels getter
    EXPECT_EQ(data_without_label.num_classes(), 0);

    // create data set with labels
    const plssvm::data_set<label_type> data_with_label{ this->data_points, this->label };
    // check num_different_labels getter
    EXPECT_EQ(data_with_label.num_classes(), this->different_label.size());
}
TYPED_TEST(DataSetGetter, is_scaled) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create data set
    const plssvm::data_set<label_type> data{ this->data_points };
    // check is_scaled getter
    EXPECT_FALSE(data.is_scaled());

    // create scaled data set
    const plssvm::data_set<label_type> data_scaled{ this->data_points, scaling_type{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };
    // check is_scaled getter
    EXPECT_TRUE(data_scaled.is_scaled());
}
TYPED_TEST(DataSetGetter, scaling_factors) {
    using label_type = TypeParam;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create data set
    const plssvm::data_set<label_type> data{ this->data_points };
    // check scaling_factors getter
    EXPECT_FALSE(data.scaling_factors().has_value());

    // create scaled data set
    const plssvm::data_set<label_type> data_scaled{ this->data_points, scaling_type{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };
    // check scaling_factors getter
    EXPECT_TRUE(data_scaled.scaling_factors().has_value());
    const auto &[ignored, correct_scaling_factors] = util::scale(this->data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    const scaling_type &scaling_factors = data_scaled.scaling_factors().value();
    EXPECT_FLOATING_POINT_EQ(scaling_factors.scaling_interval.first, plssvm::real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(scaling_factors.scaling_interval.second, plssvm::real_type{ 1.0 });
    ASSERT_EQ(scaling_factors.scaling_factors.size(), correct_scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.scaling_factors.size(); ++i) {
        EXPECT_EQ(scaling_factors.scaling_factors[i].feature, std::get<0>(correct_scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(scaling_factors.scaling_factors[i].lower, std::get<1>(correct_scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(scaling_factors.scaling_factors[i].upper, std::get<2>(correct_scaling_factors[i]));
    }
}