#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  
  // Apply the state transition matrix to our previous state x_
  // and set that as the new state
  x_ = F_ * x_;

  // Apply the state transition matrix to our covariance matrix P_
  // and also add in  process noise
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {

  // Predicted observation and actual observation
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;

  // Matrices required for calculating the Kalman Gain
  // The formula for Kalman Gain is:
  //
  // (Estimation Error) / (Estimation Error + Measurement Error)
  //
  // Which translates to the code below:
  // (P_ * Ht) / (H_ * P_ * Ht + R_)
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	// New estimate = previous estimate + (Kalman Gain * (estimated_z - observed_z))
	x_ = x_ + (K * y);
  
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);

  // New estimation error = (1 - Kalman Gain) * previous estimation error
	P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z, const MatrixXd &Hj) {

  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // Make sure phi is between -pi and +pi
  float phi = atan2(py, px);

  MatrixXd h = MatrixXd(3, 1);
  h << sqrt(pow(px, 2.0) + pow(py, 2.0)),
       phi,
       (px * vx + py * vy) / sqrt(pow(px, 2.0) + pow(py, 2.0));

  VectorXd y = z - h;
	MatrixXd Hjt = Hj.transpose();
	MatrixXd S = Hj * P_ * Hjt + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHjt = P_ * Hjt;
	MatrixXd K = PHjt * Si;
  
	// New estimate = previous estimate + (Kalman Gain * (estimated_z - observed_z))
  x_ = x_ + (K * y);

	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);

  // New estimation error = (1 - Kalman Gain) * previous estimation error
	P_ = (I - K * Hj) * P_;

}
