% High rate lower bound based on Rodrigues' work
function [v_fLowBnd]= v_fInvGammaLowBnd(v_fAlpha, v_fBeta, s_nNobs, v_nM)
% Output:
%   v_fLowBnd -  distortion bound for each dicotionary size
% input arguments:
%   v_fAlpha,  v_fBeta - inverse gamma parameters
%   s_nNobs - number of observations
%   v_nM - size of quantizer dictionary

s_nK = length(v_fAlpha);

s_nCk = (2 * (gamma(0.5)^ s_nK)) / (s_nK * gamma( s_nK /2));

s_fProd = s_nK / (s_nK + 2);
for kk=1:s_nK
   s_fProd = s_fProd *  (v_fBeta(kk)^2 / (v_fAlpha(kk) -1 +0.5*s_nNobs)) * ...
                (beta((s_nK*v_fAlpha(kk) -2 )/ (s_nK + 2), (s_nK*0.5*s_nNobs +2 )/ (s_nK + 2))^((s_nK + 2)/s_nK)) /...
                beta(v_fAlpha(kk),0.5*s_nNobs);
end

v_fLowBnd = s_fProd * ((s_nCk * v_nM).^(-2/s_nK));
