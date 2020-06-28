function [Probs]=mcdougle_softmax_func(Q,tau)
    
% Input: Q is estimated payoff values
%     tau = inverse "temperature." High values make
%     probs farther from each other (more greedy), low values less greedy
% Ouput: The bandit to choose (number between 1 and numbandits), and probs

Probs = (exp(Q.*tau)./sum(exp(Q.*tau)))';
