/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Tests for the base functionality independent of the used backend.
 */

#include "mock_csvm.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception, plssvm::file_not_found_exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type
#include "plssvm/parameter_train.hpp"        // plssvm::parameter

#include "backends/compare.hpp"  // linear_kernel
#include "utility.hpp"           // util::create_temp_file, util::gtest_expect_floating_point_eq

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ, EXPECT_EQ, EXPECT_THAT, EXPECT_THROW

#include <algorithm>   // std::generate
#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <iterator>    // std::istreambuf_iterator
#include <memory>      // std::make_shared
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string>      // std::string
#include <vector>      // std::vector

template <typename T>
class BASE : public ::testing::Test {};

using testing_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(BASE, testing_types);

TYPED_TEST(BASE, parse_libsvm) {
    // setup C-SVM class
    plssvm::parameter_train<TypeParam> params{ TEST_PATH "/data/5x4.libsvm" };
    params.print_info = false;

    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    // check if sizes match
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (size_type i = 0; i < csvm.get_num_data_points(); ++i) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    // correct values
    std::vector<std::vector<real_type>> expected{
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { -0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827 },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    };

    // check parsed values for correctness
    for (size_type i = 0; i < csvm.get_num_data_points(); ++i) {
        for (size_type j = 0; j < csvm.get_num_features(); ++j) {
            util::gtest_expect_floating_point_eq(csvm.get_data()[i][j], expected[i][j], fmt::format("datapoint: {} feature: {}", i, j));
        }
    }
}

TYPED_TEST(BASE, parse_libsvm_sparse) {
    // setup C-SVM class
    plssvm::parameter_train<TypeParam> params{ TEST_PATH "/data/5x4.sparse.libsvm" };
    params.print_info = false;

    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    // check if sizes match
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (size_type i = 0; i < csvm.get_num_data_points(); ++i) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    // correct values
    std::vector<std::vector<real_type>> expected{
        { 0., 0., 0., 0. },
        { 0., 0., 0.51687296029754564, 0. },
        { 0., 1.01405596624706053, 0., 0. },
        { 0., 0.60276937379453293, 0., -0.13086851759108944 },
        { 0., 0., 0.298499933047586044, 0. },
    };

    // check parsed values for correctness
    for (size_type i = 0; i < csvm.get_num_data_points(); ++i) {
        for (size_type j = 0; j < csvm.get_num_features(); ++j) {
            util::gtest_expect_floating_point_eq(csvm.get_data()[i][j], expected[i][j], fmt::format("datapoint: {} feature: {}", i, j));
        }
    }
}

TYPED_TEST(BASE, parse_arff) {
    // setup C-SVM class
    plssvm::parameter_train<TypeParam> params{ TEST_PATH "/data/5x4.arff" };
    params.print_info = false;

    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    // check if sizes match
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (size_type i = 0; i < csvm.get_num_data_points(); ++i) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    // correct values
    std::vector<std::vector<real_type>> expected{
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { 0., 0.60276937379453293, -0.13086851759108944, 0. },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    };

    // check parsed values for correctness
    for (size_type i = 0; i < csvm.get_num_data_points(); ++i) {
        for (size_type j = 0; j < csvm.get_num_features(); ++j) {
            util::gtest_expect_floating_point_eq(csvm.get_data()[i][j], expected[i][j], fmt::format("datapoint: {} feature: {}", i, j));
        }
    }
}

TYPED_TEST(BASE, parse_libsvm_gamma) {
    plssvm::parameter_train<TypeParam> params;
    params.input_filename = TEST_PATH "/data/5x4.libsvm";
    params.print_info = false;

    // gamma = 1.0 (!= 0.0)
    params.gamma = 1.0;
    params.parse_libsvm(params.input_filename);
    mock_csvm csvm_gamma{ params };
    using real_type_gamma = typename decltype(csvm_gamma)::real_type;
    ASSERT_EQ(csvm_gamma.get_num_data_points(), 5);
    ASSERT_EQ(csvm_gamma.get_num_features(), 4);
    util::gtest_assert_floating_point_eq(real_type_gamma{ 1.0 }, csvm_gamma.get_gamma());

    // gamma = 0.0 -> automatically set to (1.0 / num_features)
    params.gamma = 0.0;
    params.parse_libsvm(params.input_filename);
    mock_csvm csvm_gamma_zero{ params };
    using real_type_gamma_zero = typename decltype(csvm_gamma_zero)::real_type;
    EXPECT_EQ(csvm_gamma_zero.get_num_data_points(), 5);
    EXPECT_EQ(csvm_gamma_zero.get_num_features(), 4);
    util::gtest_assert_floating_point_eq(real_type_gamma_zero{ 1.0 } / csvm_gamma_zero.get_num_features(), csvm_gamma_zero.get_gamma());
}

TYPED_TEST(BASE, parse_arff_gamma) {
    plssvm::parameter_train<TypeParam> params;
    params.input_filename = TEST_PATH "/data/5x4.arff";
    params.print_info = false;

    // gamma = 1.0 (!= 0.0)
    params.gamma = 1.0;
    params.parse_arff(params.input_filename);
    mock_csvm csvm_gamma{ params };
    using real_type_gamma = typename decltype(csvm_gamma)::real_type;
    ASSERT_EQ(csvm_gamma.get_num_data_points(), 5);
    ASSERT_EQ(csvm_gamma.get_num_features(), 4);
    util::gtest_assert_floating_point_eq(real_type_gamma{ 1.0 }, csvm_gamma.get_gamma());

    // gamma = 0.0 -> automatically set to (1.0 / num_features)
    params.gamma = 0.0;
    params.parse_arff(params.input_filename);
    mock_csvm csvm_gamma_zero{ params };
    using real_type_gamma_zero = typename decltype(csvm_gamma_zero)::real_type;
    EXPECT_EQ(csvm_gamma_zero.get_num_data_points(), 5);
    EXPECT_EQ(csvm_gamma_zero.get_num_features(), 4);
    util::gtest_assert_floating_point_eq(real_type_gamma_zero{ 1.0 } / csvm_gamma_zero.get_num_features(), csvm_gamma_zero.get_gamma());
}

// enumerate all type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::rbf>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

template <typename T>
class BASE_write : public ::testing::Test {};
TYPED_TEST_SUITE(BASE_write, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(BASE_write, write_model) {
    // setup C-SVM
    plssvm::parameter_train<typename TypeParam::real_type> params{ TEST_PATH "/data/5x4.libsvm" };
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    mock_csvm csvm{ params };

    // create temporary model file and write model
    std::string model_file = util::create_temp_file();
    // learn model
    csvm.learn();
    // write learned model to file
    csvm.write_model(model_file);

    // read content of model file and delete it
    std::ifstream model_ifs(model_file);
    std::string file_content((std::istreambuf_iterator<char>(model_ifs)), std::istreambuf_iterator<char>());
    std::filesystem::remove(model_file);

    // check model file content for correctness
    switch (params.kernel) {
        case plssvm::kernel_type::linear:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
        case plssvm::kernel_type::polynomial:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type polynomial\ngamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
        case plssvm::kernel_type::rbf:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type rbf\ndegree [0-9]+\ngamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\ncoef0 [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
        default:
            FAIL() << "untested kernel" << params.kernel;
            break;
    }
}

TYPED_TEST(BASE, parse_libsvm_ill_formed) {
    // setup parameters
    plssvm::parameter_train<TypeParam> params{ TEST_PATH "/data/5x4.arff" };
    params.print_info = false;

    // parsing an arff file using the libsvm parser should result in an exception
    EXPECT_THROW(params.parse_libsvm(params.input_filename), plssvm::invalid_file_format_exception);
}

TYPED_TEST(BASE, parse_arff_ill_formed) {
    // setup parameters
    plssvm::parameter_train<TypeParam> params{ TEST_PATH "/data/5x4.libsvm" };
    params.print_info = false;

    // parsing an arff file using the libsvm parser should result in an exception
    EXPECT_THROW(params.parse_arff(params.input_filename), plssvm::invalid_file_format_exception);
}

TYPED_TEST(BASE, parse_libsvm_non_existing_file) {
    // attempting to parse a non-existing file should result in an exception
    EXPECT_THROW(plssvm::parameter_train<TypeParam>{ TEST_PATH "/data/5x4.lib" }, plssvm::file_not_found_exception);
}

TYPED_TEST(BASE, parse_arff_non_existing_file) {
    // attempting to parse a non-existing file should result in an exception
    EXPECT_THROW(plssvm::parameter_train<TypeParam>{ TEST_PATH "/data/5x4.ar" }, plssvm::file_not_found_exception);
}

TYPED_TEST(BASE, transform_data) {
    // setup C-SVM
    plssvm::parameter_train<TypeParam> params{ TEST_PATH "/data/5x4.libsvm" };
    params.print_info = false;

    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    // transform data without and with boundary
    std::vector<real_type> result_no_boundary = csvm.transform_data(0);
    std::vector<real_type> result_boundary = csvm.transform_data(10);

    // check if sizes match
    EXPECT_EQ(result_no_boundary.size(), (csvm.get_num_data_points() - 1) * csvm.get_num_features());
    EXPECT_EQ(result_boundary.size(), (csvm.get_num_data_points() - 1 + 10) * csvm.get_num_features());

    // check transformed content for correctness
    for (size_type datapoint = 0; datapoint < csvm.get_num_data_points() - 1; ++datapoint) {
        for (size_type feature = 0; feature < csvm.get_num_features(); ++feature) {
            util::gtest_expect_floating_point_eq(
                csvm.get_data()[datapoint][feature],
                result_no_boundary[datapoint + feature * (csvm.get_num_data_points() - 1)],
                fmt::format("datapoint: {} feature: {} at index: {}", datapoint, feature, datapoint + feature * (csvm.get_num_data_points() - 1)));

            util::gtest_expect_floating_point_eq(
                csvm.get_data()[datapoint][feature],
                result_boundary[datapoint + feature * (csvm.get_num_data_points() - 1 + 10)],
                fmt::format("datapoint: {} feature: {} at index: {}", datapoint, feature, datapoint + feature * (csvm.get_num_data_points() - 1 + 10)));
        }
    }
}

// generate tests for the kernel functions
template <typename T>
class BASE_kernel : public ::testing::Test {};
TYPED_TEST_SUITE(BASE_kernel, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(BASE_kernel, kernel_function) {
    // setup C-SVM
    plssvm::parameter_train<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;
    // set dummy data
    params.data_ptr = std::make_shared<const std::vector<std::vector<typename decltype(params)::real_type>>>(1);

    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;

    // create dummy data vectors
    constexpr std::size_t size = 512;
    std::vector<real_type> x1(size);
    std::vector<real_type> x2(size);

    // fill vectors with random values
    std::random_device rnd_device;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_type> dist(-1, 2.0);
    std::generate(x1.begin(), x1.end(), [&]() { return dist(gen); });
    std::generate(x2.begin(), x2.end(), [&]() { return dist(gen); });

    // calculated result
    const real_type calculated = csvm.kernel_function(x1, x2);

    // correct result
    const real_type correct = compare::kernel_function<TypeParam::kernel>(x1, x2, csvm);

    // check for correctness
    util::gtest_expect_floating_point_eq(correct, calculated);
}