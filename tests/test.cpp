// #include "CSVM.hpp"
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <unistd.h>

#include "MockCSVM.hpp"
#include "backends/compare.hpp"
#include "plssvm/exceptions/exceptions.hpp"

#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"
#include "plssvm/detail/string_utility.hpp"
#include <plssvm/backends/OpenMP/svm-kernel.hpp>
#include <plssvm/kernel_types.hpp>

TEST(BASE, libsvmFormat) {
    MockCSVM csvm(plssvm::kernel_type::linear, 1., 1., 1., 1., 1., false);
    using real_type = typename MockCSVM::real_type;

    csvm.parse_libsvm(TESTPATH "/data/5x4.libsvm");
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    std::vector<std::vector<real_type>> expected{
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { -0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827 },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(BASE, sparselibsvmFormat) {
    MockCSVM csvm(plssvm::kernel_type::linear, 1., 1., 1., 1., 1., false);
    using real_type = typename MockCSVM::real_type;

    csvm.parse_libsvm(TESTPATH "/data/5x4.sparse.libsvm");
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    std::vector<std::vector<real_type>> expected{
        { 0., 0., 0., 0. },
        { 0., 0., 0.51687296029754564, 0. },
        { 0., 1.01405596624706053, 0., 0. },
        { 0., 0.60276937379453293, 0., -0.13086851759108944 },
        { 0., 0., 0.298499933047586044, 0. },
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(BASE, arffFormat) {
    MockCSVM csvm(plssvm::kernel_type::linear, 1., 1., 1., 1., 1., false);
    using real_type = typename MockCSVM::real_type;

    csvm.parse_arff(TESTPATH "/data/5x4.arff");
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    std::vector<std::vector<real_type>> expected{
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { 0., 0.60276937379453293, -0.13086851759108944, 0. },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(BASE, arffParserGamma) {
    MockCSVM csvm(plssvm::kernel_type::linear, 1., 1., 1., 1., 1., false);
    csvm.parse_arff(TESTPATH "/data/5x4.arff");
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_FLOAT_EQ(1.0, csvm.get_gamma());

    MockCSVM csvm_gammazero(plssvm::kernel_type::linear, 1., 0, 1., 1., 1., false);
    csvm_gammazero.parse_arff(TESTPATH "/data/5x4.arff");
    EXPECT_EQ(csvm_gammazero.get_num_data_points(), 5);
    EXPECT_EQ(csvm_gammazero.get_num_features(), 4);
    EXPECT_FLOAT_EQ(1.0 / csvm_gammazero.get_num_features(), csvm_gammazero.get_gamma());
}

TEST(BASE, libsvmParserGamma) {
    MockCSVM csvm(plssvm::kernel_type::linear, 1., 1., 1., 1., 1., false);
    csvm.parse_libsvm(TESTPATH "/data/5x4.libsvm");
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_FLOAT_EQ(1.0, csvm.get_gamma());

    MockCSVM csvm_gammazero(plssvm::kernel_type::linear, 1., 0, 1., 1., 1., false);
    csvm_gammazero.parse_libsvm(TESTPATH "/data/5x4.libsvm");
    EXPECT_EQ(csvm_gammazero.get_num_data_points(), 5);
    EXPECT_EQ(csvm_gammazero.get_num_features(), 4);
    EXPECT_FLOAT_EQ(1.0 / csvm_gammazero.get_num_features(), csvm_gammazero.get_gamma());
}

TEST(BASE, writeModel) {
    MockCSVM csvm(plssvm::kernel_type::linear, 3.0, 0.0, 0.0, 1., 0.001, false);
    csvm.parse_libsvm(TESTPATH "/data/5x4.libsvm");

    std::string model = std::filesystem::temp_directory_path().string();
    model += "/tmpfile_XXXXXX";
    // create unique temporary file
    int fd = mkstemp(model.data());
    // immediately close file if possible
    if (fd >= 0) {
        close(fd);
    }

    csvm.write_model(model);
    std::ifstream model_ifs(model);
    std::string genfile1((std::istreambuf_iterator<char>(model_ifs)), std::istreambuf_iterator<char>());
    std::filesystem::remove(model.c_str());

    EXPECT_THAT(genfile1, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv 0+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
}

TEST(BASE, libsvmFormatIllFormed) {
    MockCSVM csvm(plssvm::kernel_type::linear, 1., 1., 1., 1., 1., false);
    EXPECT_THROW(csvm.parse_libsvm(TESTPATH "/data/5x4.arff");, plssvm::invalid_file_format_exception);
}

TEST(BASE, arffFormatIllFormed) {
    MockCSVM csvm(plssvm::kernel_type::linear, 1., 1., 1., 1., 1., false);
    EXPECT_THROW(csvm.parse_arff(TESTPATH "/data/5x4.libsvm");, plssvm::invalid_file_format_exception);
}

TEST(BASE, libsvmNoneExistingFile) {
    MockCSVM csvm(plssvm::kernel_type::linear, 1., 1., 1., 1., 1., false);
    EXPECT_THROW(csvm.parse_libsvm(TESTPATH "/data/5x5.ar");, plssvm::file_not_found_exception);
}

TEST(BASE, arffNoneExistingFile) {
    MockCSVM csvm(plssvm::kernel_type::linear, 1., 1., 1., 1., 1., false);
    EXPECT_THROW(csvm.parse_arff(TESTPATH "/data/5x5.lib");, plssvm::file_not_found_exception);
}

TEST(BASE, transform_data) {
    MockCSVM csvm(plssvm::kernel_type::linear, 3.0, 0.0, 0.0, 1.0, 0.001, false);
    using real_type = typename MockCSVM::real_type;

    csvm.parse_libsvm(TESTPATH "/data/5x4.libsvm");
    std::vector<real_type> result0 = csvm.transform_data(0);
    std::vector<real_type> result10 = csvm.transform_data(10);

    EXPECT_EQ(result0.size(), (csvm.get_num_data_points() - 1) * csvm.get_num_features());
    EXPECT_EQ(result10.size(), (csvm.get_num_data_points() - 1 + 10) * csvm.get_num_features());

    for (size_t datapoint = 0; datapoint < csvm.get_num_data_points() - 1; ++datapoint) {
        for (size_t feature = 0; feature < csvm.get_num_features(); ++feature) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[datapoint][feature], result0[datapoint + feature * (csvm.get_num_data_points() - 1)]) << "datapoint: " << datapoint << " feature: " << feature << " at index: " << datapoint + feature * (csvm.get_num_data_points() - 1);
            EXPECT_DOUBLE_EQ(csvm.get_data()[datapoint][feature], result10[datapoint + feature * (csvm.get_num_data_points() - 1 + 10)]) << "datapoint: " << datapoint << " feature: " << feature << " at index: " << datapoint + feature * (csvm.get_num_data_points() - 1 + 10);
        }
    }
}

TEST(BASE, linear) {
    using real_type = typename MockCSVM::real_type;

    const real_type degree = 0.0;
    const real_type gamma = 0.0;
    const real_type coef0 = 0.0;
    const size_t size = 512;
    std::vector<real_type> x1(size);
    std::vector<real_type> x2(size);
    std::generate(x1.begin(), x1.end(), std::rand);
    std::generate(x2.begin(), x2.end(), std::rand);
    real_type correct = linear_kernel(x1, x2);

    MockCSVM csvm(plssvm::kernel_type::linear, degree, gamma, coef0, 1., 0.001, false);
    real_type result = csvm.kernel_function(x1, x2);
    real_type result2 = csvm.kernel_function(x1, x2);

    EXPECT_DOUBLE_EQ(correct, result);
    EXPECT_DOUBLE_EQ(correct, result2);
}
