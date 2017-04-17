#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // Measurement function
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // Jacobian matrix
  Hj_ = MatrixXd(3, 4);

  ekf_.x_ = VectorXd(4);
  ekf_.x_ << 0, 0, 0, 0;

  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1000.0, 0.0,
              0.0, 0.0, 0.0, 1000.0;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.Q_ = MatrixXd(4, 4);

  noise_ax = 9;
  noise_ay = 9;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  float x, y, vx, vy;
  double elapsed_time;
  VectorXd z;

  // Process the measurements
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

    // Get the measurements in polar form
    float rho = measurement_pack.raw_measurements_(0);
    float phi = measurement_pack.raw_measurements_(1);
    float rho_dot = measurement_pack.raw_measurements_(2);

    elapsed_time = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;
    
    // Get the x and y components of the position and velocity vectors
    x = rho * cos(phi);
    y = rho * sin(phi);
    vx = rho_dot * cos(phi);
    vy = rho_dot * sin(phi);
    
    z = VectorXd(3);
    z << rho, phi, rho_dot;

  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {

    // Measurements are already in cartesian form
    x = measurement_pack.raw_measurements_(0);
    y = measurement_pack.raw_measurements_(1);

    elapsed_time = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

    z = VectorXd(2);
    z << x, y;

  }

  // If this is our first measurement, initialize the state ekf_.x_ and return
  // No need to predict or update
  if (!is_initialized_) {

    ekf_.x_ << x, y, vx, vy;
    ekf_.P_ << 1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1000.0, 0.0,
               0.0, 0.0, 0.0, 1000.0;

    is_initialized_ = true;
    return;
  }

  // Update the state transition matrix F with elapsed time (dt)
  double dt = elapsed_time;

  ekf_.F_ << 1.0, 0.0, dt, 0.0,
			       0.0, 1.0, 0.0, dt,
			       0.0, 0.0, 1.0, 0.0,
			       0.0, 0.0, 0.0, 1.0;

  // Update the process covariance matrix Q
	ekf_.Q_ << (pow(dt, 4.0) / 4.0) * noise_ax, 0.0, (pow(dt, 3.0) / 2.0) * noise_ax, 0.0,
	           0.0, (pow(dt, 4) / 4.0) * noise_ay, 0.0, (pow(dt, 3.0) / 2.0) * noise_ay,
	           (pow(dt, 3.0) / 2.0) * noise_ax, 0.0, (pow(dt, 2.0) * noise_ax), 0.0,
	           0.0, (pow(dt, 3.0) / 2.0) * noise_ay, 0.0, (pow(dt, 2.0) * noise_ay);


  // Prediction Step

  // In the special case where both x and y == 0,
  // We need to skip processing or we will have a division by zero
  if (x == 0 && y == 0) return;

  ekf_.Predict();

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    MatrixXd Hj = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(z, Hj);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(z);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
