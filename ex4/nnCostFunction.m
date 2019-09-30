function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


K = num_labels;


h1 = X;
a1 = [ones(m,1) X];
z2 = (a1*Theta1');
h2 = sigmoid(z2);

a2 = [ones((size(h2, 1)),1) h2]; %adding bias column to the size of no of row of h2

z3 = (a2*Theta2');
h3 = sigmoid(z3);

a3 = h3;


ymatrix = zeros(m,num_labels);

for i = 1:m
    ymatrix(i,y(i)) = 1; %y vector mapping - find the value of actual y and input that value in that column of y matrix in each row
end


%no regularized
J = (1/m) * sum(sum((-ymatrix.* log(a3) - (1-ymatrix).* log(1-a3))));


%regulating term
regulator = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));


%Full cost
J = J + regulator;




%Part 2

% del3 = y - a3;
% 
% del2 = (Theta2'*del3) .* (h2.*(1-h2)); %use h instead of a becasue a includes the bias unit with valu of 1
% 
% del1 = (Theta1'*del2) .* (h1.*(1-h1)); %h1 is X, it is a1 without the bia unit


for t = 1:m
    
    % Forward to calculate error for sample t
    
    a_1 = X(t,:)'; %making it a column vector with 400 rows
    a_1 = [1; a_1]; %adding bias of 1 -- input layer
           
    z_2 = Theta1 * a_1;
    
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2]; %adding bias --- second layer
    
    z_3 = Theta2 * a_2;
    
    a_3 = sigmoid(z_3); %output layer
    
    
    
    % Error
    yt = ymatrix(t,:)';
    
    delta_3 = a_3 - yt; %output layer compared with real value of y
    
    
    % Propagate error backwards
    
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z_2]);
    delta_2 = delta_2(2:end); %ignoring the bias

    dt2 = delta_3 * a_2'; %capital Delta
    dt1 = delta_2 * a_1';
 
    Theta2_grad = Theta2_grad + dt2;
    Theta1_grad = Theta1_grad + dt1;
end


Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% Add regularization terms
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
