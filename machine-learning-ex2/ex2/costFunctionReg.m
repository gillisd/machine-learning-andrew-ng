function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


J = (1 / m) * sum( -y .* log(sigmoid(X*theta)) - (1 - y) .* log(1 - sigmoid(X*theta)));

grad = sum((sigmoid(X*theta) - y).*X)/m;
reg_inner_sum = 0;

for i = 1:size(theta)
    if i > 1
        reg_inner_sum = reg_inner_sum + theta(i).^2;
        grad(i) = grad(i) + (lambda * theta(i)) / m;
    else
        grad(i) = grad(i);
    end
end
reg_term = reg_inner_sum * lambda / (2*m);
J = J + reg_term;




% =============================================================

end
