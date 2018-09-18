% Function name - v_fEigAlloc
% Purpose - Waterfilling solution for rate-distortion problem with fixed
% rate
% Input arguments:
%   v_fLambda - eigenvalues
%   s_nM - Number of bits
% Output arguments:
%   v_fLambdaT - Power allocation map
%   s_fDelta - Waterfilling threshold
function [v_fLambdaT, s_fDelta] = v_fEigAlloc(v_fLambda, s_nM)

% Sort eigenvalues in descending manner
v_fTemp = sort(v_fLambda,'descend');
s_fProd = 1 / (s_nM^2);
s_nN = length(v_fLambda);
for kk=1:s_nN
    % Evaluate prod 1..k eig_k
    s_fProd = s_fProd * v_fTemp(kk);
    s_fDelta = s_fProd^(1/kk);
    if  (kk < s_nN) && (s_fDelta<= v_fTemp(kk)) && (s_fDelta>= v_fTemp(kk+1)) 
        break;
    end
end
v_fLambdaT = max(v_fLambda-s_fDelta,0);
