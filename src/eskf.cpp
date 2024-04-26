//
// Created by meng on 2021/2/19.
//
#include "eskf.h"
#include "gps_tool.h"
#include "common_tool.h"

ErrorStateKalmanFilter::ErrorStateKalmanFilter(const ConfigParameters &config_parameters)
        : config_parameters_(config_parameters) {

    earth_rotation_speed_ = config_parameters_.earth_rotation_speed_;
    /// [q] 为什么加符号？因为定义的北东地坐标系，z轴朝下
    g_ = Eigen::Vector3d(0.0, 0.0, -config_parameters_.earth_gravity_);

    SetCovarianceP(config_parameters_.position_error_prior_std_,
                   config_parameters_.velocity_error_prior_std_,
                   config_parameters_.rotation_error_prior_std_,
                   config_parameters_.gyro_bias_error_prior_std_,
                   config_parameters_.accelerometer_bias_error_prior_std_);

    SetCovarianceR(config_parameters_.gps_position_x_std_,
                   config_parameters_.gps_position_y_std_,
                   config_parameters_.gps_position_z_std_);

    SetCovarianceQ(config_parameters_.gyro_noise_std_, config_parameters_.accelerometer_noise_std_);

    X_.setZero();
    F_.setZero();
    C_.setIdentity();
    G_.block<3, 3>(INDEX_MEASUREMENT_POSI, INDEX_MEASUREMENT_POSI) = Eigen::Matrix3d::Identity();
}

void ErrorStateKalmanFilter::SetCovarianceQ(double gyro_noise, double accel_noise) {
    Q_.setZero();
    Q_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * gyro_noise * gyro_noise;
    Q_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * accel_noise * accel_noise;
}

void ErrorStateKalmanFilter::SetCovarianceR(double position_x_std, double position_y_std, double position_z_std) {
    R_.setZero();
    R_(0, 0) = position_x_std * position_x_std;
    R_(1, 1) = position_y_std * position_y_std;
    R_(2, 2) = position_z_std * position_z_std;
}

void ErrorStateKalmanFilter::SetCovarianceP(double posi_noise, double velocity_noise, double ori_noise,
                                            double gyro_noise, double accel_noise) {
    P_.setZero();
    P_.block<3, 3>(INDEX_STATE_POSI, INDEX_STATE_POSI) = Eigen::Matrix3d::Identity() * posi_noise * posi_noise;
    P_.block<3, 3>(INDEX_STATE_VEL, INDEX_STATE_VEL) = Eigen::Matrix3d::Identity() * velocity_noise * velocity_noise;
    P_.block<3, 3>(INDEX_STATE_ORI, INDEX_STATE_ORI) = Eigen::Matrix3d::Identity() * ori_noise * ori_noise;
    P_.block<3, 3>(INDEX_STATE_GYRO_BIAS, INDEX_STATE_GYRO_BIAS) =
            Eigen::Matrix3d::Identity() * gyro_noise * gyro_noise;
    P_.block<3, 3>(INDEX_STATE_ACC_BIAS, INDEX_STATE_ACC_BIAS) =
            Eigen::Matrix3d::Identity() * accel_noise * accel_noise;
}

bool ErrorStateKalmanFilter::Init(const GPSData &curr_gps_data, const IMUData &curr_imu_data) {
    velocity_ = curr_gps_data.true_velocity;

    Eigen::Quaterniond Q_init = Eigen::AngleAxisd(0 * kDegree2Radian, Eigen::Vector3d::UnitZ()) *
                                Eigen::AngleAxisd(0 * kDegree2Radian, Eigen::Vector3d::UnitY()) *
                                Eigen::AngleAxisd(0 * kDegree2Radian, Eigen::Vector3d::UnitX());

    pose_.block<3, 3>(0, 0) = Q_init.toRotationMatrix();
    pose_.block<3, 1>(0, 3) = curr_gps_data.local_position_ned;

    imu_data_buff_.clear();
    imu_data_buff_.push_back(curr_imu_data);

    curr_gps_data_ = curr_gps_data;

    return true;
}

void ErrorStateKalmanFilter::GetFGY(TypeMatrixF &F, TypeMatrixG &G, TypeVectorY &Y) {
    F = Ft_;
    G = G_;
    Y = Y_;
}

bool ErrorStateKalmanFilter::Correct(const GPSData &curr_gps_data) {
    curr_gps_data_ = curr_gps_data;

    Y_ = curr_gps_data.local_position_ned - pose_.block<3, 1>(0, 3);

    /// 计算卡尔曼增益K
    /// 此代码中观测方程矩阵G（其他文献中一般称为H）是一个定值，因为本文gnss观测的应用非常简单。其他一般应用中，H矩阵可能需要每次都重新计算。
    K_ = P_ * G_.transpose() * (G_ * P_ * G_.transpose() + C_ * R_ * C_.transpose()).inverse();

    P_ = (TypeMatrixP::Identity() - K_ * G_) * P_;
    X_ = X_ + K_ * (Y_ - G_ * X_);

    EliminateError();

    /// [q] 为什么每次都要设为零？因为X是“误差状态error_state”
    ResetState();

    return true;
}

bool ErrorStateKalmanFilter::Predict(const IMUData &curr_imu_data) {
    imu_data_buff_.push_back(curr_imu_data);

    Eigen::Vector3d w_in = Eigen::Vector3d::Zero();

    if (config_parameters_.use_earth_model_) {
        w_in = ComputeNavigationFrameAngularVelocity(); // 时刻 m-1 -> m 地球转动引起的导航系转动角速度
    }
    
    /// 通过imu的量测，来预测状态（姿态、速度、位置）的变化，更新旧状态。（Nominal State）
    UpdateOdomEstimation(w_in);

    double delta_t = curr_imu_data.time - imu_data_buff_.front().time;

    Eigen::Vector3d curr_accel = pose_.block<3, 3>(0, 0) * curr_imu_data.linear_accel;

    /// 卡尔曼滤波预测公式。输出：更新后的状态向量X、协方差矩阵P。（Error State）
    UpdateErrorState(delta_t, curr_accel, w_in);

    imu_data_buff_.pop_front();

    return true;
}

void ErrorStateKalmanFilter::UpdateErrorState(double t, const Eigen::Vector3d &accel, const Eigen::Vector3d &w_in_n) {
    Eigen::Matrix3d F_23 = BuildSkewSymmetricMatrix(accel);
    Eigen::Matrix3d F_33 = -BuildSkewSymmetricMatrix(w_in_n); /// 不考虑w_in_n时，w_in_n为零，F_33为零矩阵

    /// 协方差矩阵的计算 此处实现是否有问题？
    F_.block<3, 3>(INDEX_STATE_POSI, INDEX_STATE_VEL) = Eigen::Matrix3d::Identity();
    F_.block<3, 3>(INDEX_STATE_VEL, INDEX_STATE_ORI) = F_23;
    F_.block<3, 3>(INDEX_STATE_ORI, INDEX_STATE_ORI) = F_33;
    F_.block<3, 3>(INDEX_STATE_VEL, INDEX_STATE_ACC_BIAS) = pose_.block<3, 3>(0, 0);
    F_.block<3, 3>(INDEX_STATE_ORI, INDEX_STATE_GYRO_BIAS) = -pose_.block<3, 3>(0, 0);
    B_.block<3, 3>(INDEX_STATE_VEL, 3) = pose_.block<3, 3>(0, 0);
    B_.block<3, 3>(INDEX_STATE_ORI, 0) = -pose_.block<3, 3>(0, 0);

    /// My implementation according to <Quaternion kinematics for the error-state Kalman filter>
//    F_.block<3, 3>(INDEX_STATE_POSI, INDEX_STATE_VEL) = Eigen::Matrix3d::Identity();
//    F_.block<3, 3>(INDEX_STATE_VEL, INDEX_STATE_ORI) = -pose_.block<3, 3>(0, 0) * BuildSkewSymmetricMatrix(ComputeUnbiasAccel(curr_imu_data.linear_accel));
//    F_.block<3, 3>(INDEX_STATE_ORI, INDEX_STATE_ORI) = -BuildSkewSymmetricMatrix(ComputeUnbiasGyro(curr_imu_data.angle_velocity));
//    F_.block<3, 3>(INDEX_STATE_VEL, INDEX_STATE_ACC_BIAS) = -pose_.block<3, 3>(0, 0);
//    F_.block<3, 3>(INDEX_STATE_ORI, INDEX_STATE_GYRO_BIAS) = -Eigen::Matrix3d::Identity();
//    B_.block<3, 3>(INDEX_STATE_VEL, 3) = Eigen::Matrix3d::Identity();
//    B_.block<3, 3>(INDEX_STATE_ORI, 0) = Eigen::Matrix3d::Identity();
        
    TypeMatrixF Fk = TypeMatrixF::Identity() + F_ * t;
    TypeMatrixB Bk = B_ * t;

    // 用于可观测性分析
    Ft_ = F_ * t;

    /// 卡尔曼滤波状态预测方程。状态转移矩阵 更新状态
    /// eskf中，X_的更新实际上没必要，因为每次predict之前，dx会被设置为零。
    X_ = Fk * X_;
    P_ = Fk * P_ * Fk.transpose() + Bk * Q_ * Bk.transpose();
}

/// 通过imu的观测，来预测状态变化。涉及的状态有：姿态：pose_.block<3, 3>(0, 0)、速度：velocity_、位置：pose_.block<3, 1>(0, 3)
void ErrorStateKalmanFilter::UpdateOdomEstimation(const Eigen::Vector3d &w_in) {
    const auto &last_imu_data = imu_data_buff_.at(0);
    const auto &curr_imu_data = imu_data_buff_.at(1);
    const double delta_t = curr_imu_data.time - last_imu_data.time;

    /// 根据前后帧的imu角速度测量 + bg估计，来估算旋转角度变化；注意这个delta是在imu车体系下。
    Eigen::Vector3d delta_rotation = ComputeDeltaRotation(last_imu_data, curr_imu_data);

    /// 考虑了地球转动引起的导航系转动角速度。如果不考虑，则w_in为零，R_nm_nm_1为单位阵
    const Eigen::Vector3d phi_in = w_in * delta_t;
    const Eigen::AngleAxisd angle_axisd(phi_in.norm(), phi_in.normalized());
    const Eigen::Matrix3d R_nm_nm_1 = angle_axisd.toRotationMatrix().transpose();

    /// 更新状态：姿态角。计算公式：陀螺仪测量的角速度*时间=delta角度；新角度=旧角度*delta
    Eigen::Matrix3d curr_R; // R_n_m m时刻的旋转
    Eigen::Matrix3d last_R; // C_n_m_1 m-1时刻的旋转
    ComputeOrientation(delta_rotation, R_nm_nm_1, curr_R, last_R);

    /// 更新状态：线速度。计算公式：加速度计测量的加速度*时间=delta速度；新速度=旧速度+delta速度
    Eigen::Vector3d curr_vel; // 当前时刻导航系下的速度
    Eigen::Vector3d last_vel; // 上一时刻导航系下的速度
    ComputeVelocity(last_R, curr_R, last_imu_data, curr_imu_data, last_vel, curr_vel);

    /// 更新状态：位置。计算公式：匀加速直线运动模型；新位置=旧位置+delta位置
    ComputePosition(last_R, curr_R, last_vel, curr_vel, last_imu_data, curr_imu_data);
}

Eigen::Vector3d ErrorStateKalmanFilter::ComputeDeltaRotation(const IMUData &imu_data_0, const IMUData &imu_data_1) {
    const double delta_t = imu_data_1.time - imu_data_0.time;

    CHECK_GT(delta_t, 0.0) << "IMU timestamp error";

    /// 补偿角速度的测量偏置bg。【预期真值 = 测量值 - bg】
    const Eigen::Vector3d &unbias_gyro_0 = ComputeUnbiasGyro(imu_data_0.angle_velocity);
    const Eigen::Vector3d &unbias_gyro_1 = ComputeUnbiasGyro(imu_data_1.angle_velocity);

    /// 陀螺仪测量的是角速度；假设匀加速旋转，则使用前后时刻的【角速度平均值 * 时间】来计算角度变化
    Eigen::Vector3d delta_theta = 0.5 * (unbias_gyro_0 + unbias_gyro_1) * delta_t;

    return delta_theta;
}

Eigen::Vector3d ErrorStateKalmanFilter::ComputeNavigationFrameAngularVelocity() {
    const double latitude = curr_gps_data_.position_lla.y() * kDegree2Radian;
    const double height = curr_gps_data_.position_lla.z();

    constexpr double f = 1.0 / 298.257223563; // 椭球扁率

    constexpr double Re = 6378137.0; // 椭圆长半轴
    constexpr double Rp = (1.0 - f) * Re; // 椭圆短半轴
    const double e = std::sqrt(Re * Re - Rp * Rp) / Re; // 椭圆的偏心率

    const double Rn = Re / std::sqrt(1.0 - e * e * std::sin(latitude) * std::sin(latitude)); // 子午圈主曲率半径
    const double Rm = Re * (1.0 - e * e)
                      / std::pow(1.0 - e * e * std::sin(latitude) * std::sin(latitude), 3.0 / 2.0); // 卯酉圈主曲率半径

    // 由于载体在地球表面运动造成的导航系姿态变化。在导航系下表示
    Eigen::Vector3d w_en_n;
    w_en_n << velocity_[1] / (Rm + height), -velocity_[0] / (Rn + height),
            -velocity_[1] / (Rn + height) * std::tan(latitude);

    Eigen::Vector3d w_ie_n;
    w_ie_n << earth_rotation_speed_ * std::cos(latitude), 0.0, -earth_rotation_speed_ * std::sin(latitude);

    Eigen::Vector3d w_in_n = w_en_n + w_ie_n;

    return w_in_n;
}

void ErrorStateKalmanFilter::ComputeOrientation(const Eigen::Vector3d &angular_delta,
                                                const Eigen::Matrix3d &R_nm_nm_1,
                                                Eigen::Matrix3d &curr_R,
                                                Eigen::Matrix3d &last_R) {
    Eigen::AngleAxisd angle_axisd(angular_delta.norm(), angular_delta.normalized());

    /// [q] 左乘还是右乘？
    /// 考虑这么理解：假设旋转矩阵的后面乘以一个三维点向量进行旋转。deltaR是局部的车体系下的，所以按照“先处理局部，再全局变换”的思路。
    /// 因此全局的旋转R在左边，意味着后乘；局部的delta变换的R在右边（更靠近坐标向量），意味着先乘。
    last_R = pose_.block<3, 3>(0, 0);
    curr_R = R_nm_nm_1.transpose() * pose_.block<3, 3>(0, 0) * angle_axisd.toRotationMatrix();
    pose_.block<3, 3>(0, 0) = curr_R;
}

void ErrorStateKalmanFilter::ComputeVelocity(const Eigen::Matrix3d &R_0, const Eigen::Matrix3d &R_1,
                                             const IMUData &imu_data_0, const IMUData &imu_data_1,
                                             Eigen::Vector3d &last_vel, Eigen::Vector3d &curr_vel) {
    double delta_t = imu_data_1.time - imu_data_0.time;

    CHECK_GT(delta_t, 0.0) << "IMU timestamp error";

    /// imu测量的是imu系下的的线加速度（且包含了重力加速度）
    /// 因此计算分为三步：
    ///     1、补偿线加速度ba。【预期真值 = 测量值 - bg】
    ///     2、将局部加速度（测量时的车体系）转换为全局坐标系下的加速度。此处会分别用到两个时刻下的全局姿态R
    ///     3、补偿重力加速度
    Eigen::Vector3d unbias_accel_0 = R_0 * ComputeUnbiasAccel(imu_data_0.linear_accel) - g_;
    Eigen::Vector3d unbias_accel_1 = R_1 * ComputeUnbiasAccel(imu_data_1.linear_accel) - g_;

    last_vel = velocity_;

    /// 陀螺仪测量的是线加速度；假设匀加速运动，则使用前后时刻的【线加速度平均值 * 时间】来计算速度变化
    // 中值积分
    velocity_ += delta_t * 0.5 * (unbias_accel_0 + unbias_accel_1);

    curr_vel = velocity_;
}

Eigen::Vector3d ErrorStateKalmanFilter::ComputeUnbiasAccel(const Eigen::Vector3d &accel) {
    return accel - accel_bias_;
}

Eigen::Vector3d ErrorStateKalmanFilter::ComputeUnbiasGyro(const Eigen::Vector3d &gyro) {
    return gyro - gyro_bias_;
}

void ErrorStateKalmanFilter::ComputePosition(const Eigen::Matrix3d &R_0, const Eigen::Matrix3d &R_1,
                                             const Eigen::Vector3d &last_vel, const Eigen::Vector3d &curr_vel,
                                             const IMUData &imu_data_0, const IMUData &imu_data_1) {
    /// 与 ComputeVelocity() 中的计算方法一致
    Eigen::Vector3d unbias_accel_0 = R_0 * ComputeUnbiasAccel(imu_data_0.linear_accel) - g_;
    Eigen::Vector3d unbias_accel_1 = R_1 * ComputeUnbiasAccel(imu_data_1.linear_accel) - g_;

    double delta_t = imu_data_1.time - imu_data_0.time;

    /// 匀加速直线运动模型：s = v_0 * t + 0.5 * a * t^2
    pose_.block<3, 1>(0, 3) += 0.5 * delta_t * (curr_vel + last_vel) +
                               0.25 * (unbias_accel_0 + unbias_accel_1) * delta_t * delta_t;
}

void ErrorStateKalmanFilter::ResetState() {
    X_.setZero();
}

void ErrorStateKalmanFilter::EliminateError() {
    pose_.block<3, 1>(0, 3) = pose_.block<3, 1>(0, 3) + X_.block<3, 1>(INDEX_STATE_POSI, 0);

    velocity_ = velocity_ + X_.block<3, 1>(INDEX_STATE_VEL, 0);

    /// 更新姿态角。此处实现是否有问题？
    Eigen::Matrix3d C_nn = SO3Exp(-X_.block<3, 1>(INDEX_STATE_ORI, 0));
    pose_.block<3, 3>(0, 0) = C_nn * pose_.block<3, 3>(0, 0);

    /// My implementation according to <Quaternion kinematics for the error-state Kalman filter>
//    pose_.block<3, 3>(0, 0) = pose_.block<3, 3>(0, 0) * SO3Exp(X_.block<3, 1>(INDEX_STATE_ORI, 0));

    gyro_bias_ = gyro_bias_ + X_.block<3, 1>(INDEX_STATE_GYRO_BIAS, 0);
    accel_bias_ = accel_bias_ + X_.block<3, 1>(INDEX_STATE_ACC_BIAS, 0);
}

Eigen::Matrix4d ErrorStateKalmanFilter::GetPose() const {
    return pose_;
}
