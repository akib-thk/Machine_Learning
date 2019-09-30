function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;

%non-regular

J_unreg = (1/(2*m)) * sum((h - y).^2);

%regulator
reg = (lambda/(2*m)) * sum(theta(2:end,:).^2);


J = J_unreg + reg;


%theta0 = theta(1,:);
%theta1 = theta(2,:);

% temp0 = theta0 - (alpha/m) * sum(h - y);
% temp1 = theta1 - (alpha/m) * sum((h - y)*(X(:,2))');
% 
% theta0 = temp0;
% theta1 = temp1;

%grad(1) = (1/m) * sum((h - y)' * X(:,1));

%grad(2) = (1/m) * sum((h - y)' * X(:,2)) + (lambda/m) * theta1;

%grad = [grad0 grad1];

theta_reg = theta;
theta_reg(1) = 0;

grad = ((1 / m) * (h - y)' * X) + (lambda / m) * theta_reg';

% =========================================================================

grad = grad(:);

end
