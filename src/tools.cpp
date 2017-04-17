#include <iostream>
#include <cmath>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                                     const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

  // Sanity checks
  int size_est = estimations.size();
  int size_gnd = ground_truth.size();

  if (size_est == 0 || size_gnd == 0 || size_est != size_gnd) {
      cout << "Tools::CalculateRMSE Error: Estimation or Ground Truth vectors invalid." << endl;
      return rmse;
  }

  //accumulate squared residuals
  VectorXd diff(4);
  for(int i = 0; i < size_est; ++i) {
      diff = estimations[i] - ground_truth[i];
        rmse = rmse.array() + (diff.array() * diff.array());
  }

  //calculate the mean and square root
  rmse = rmse.array() / size_est;
  rmse = rmse.array().sqrt();	

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float g = pow(px, 2) + pow(py, 2);

  //check for division by zero
  if (g == 0) {
      cout << "Division by zero" << endl;
      return Hj;
  }

  //compute the Jacobian matrix
  Hj << px / sqrt(g), py / sqrt(g), 0.0, 0.0,
        -py / g, px / g, 0.0, 0.0,
        (py * (vx * py - vy * px)) / pow(g, 1.5), 
        (px * (vy * px - vx * py)) / pow(g, 1.5),
        px / sqrt(g), py / sqrt(g);

  return Hj;
}
