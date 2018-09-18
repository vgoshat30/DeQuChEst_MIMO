% Calculate task ignorant distortion approximation via random codes for
% i.i.d inputs
function [v_fLowBnd]= v_fApproxX2(m_fCx, m_fEstMat, v_nM)
% Output:
%   v_fApprox -  distortion bound for each dicotionary size
% input arguments:
%   m_fCStilde - observed signal covariance
%   v_nM - size of quantizer dictionary

% Get eigenvluaes and eigenmatrix of covariance
[m_fV,m_fDx]= eig(m_fCx);
v_fLambda = diag(m_fDx);
v_fLowBnd =  zeros(size(v_nM));


for kk=1:length(v_nM)
    s_nM = v_nM(kk);
    % Obtain the distibution of the optimal quantizer for given M
    [v_fLambdaT, ~] = v_fEigAlloc(v_fLambda, s_nM);
    % Calculate quantizer output covariance - add small diagonal entries to
    % avoid positive definitness errors
    m_fCz = m_fV*diag(v_fLambdaT)*m_fV.';
    % Obtain lower bound based on rate distortion function
    v_fLowBnd(kk) = trace(m_fEstMat.' * m_fEstMat * (m_fCx -m_fCz));
    
end