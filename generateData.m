function generateData(varargin)
    % Training and testing data generation-asymptotic massive MIMO case
    % study.
    %
    % Created April 2018
    % 
    % Creator: Nir Shlezinger
    %   Worked on code: Gosha Tsintsadze
    %                   Matan Shohat
    %
    %
    % Optional Name-Value Pairs:
    %
    %
    %   shlezDataGen('Autosave','on') Displays a folder choosing window
    %                                 before saving the .mat file. Call
    %                                 with 'off' or ommit to automatically
    %                                 save to the corrent directory or to
    %                                 manually chose one.
    %
    %
    %   shlezDataGen('FileName','shlezMat') Choose the file name of the
    %                                       output .mat file. DO NOT add
    %                                       .mat extention.
    %
    %
    %   shlezDataGen('Directory','shlezMat') Specify the directory of the
    %                                        output file.
    %
    %
    %
    %
    %   Specify Data parameters:
    %
    %
    %
    %   shlezDataGen('MaxRate', 5)          Deafult is: 1.5
    %
    %   shlezDataGen('Power', 5)            Deafult is: 4
    %
    %   shlezDataGen('Users', 5)            Deafult is: 4
    %
    %   shlezDataGen('Antennas', 15)        Deafult is: 10
    %
    %   shlezDataGen('Ratio', 4)            Deafult is: 3
    %
    %   shlezDataGen('TrainSetSize', 2^10)  Deafult is: 2^15
    %
    %   shlezDataGen('TestSetSize', 2^15)   Deafult is: 
    %
    %
    %
    %   Modifications History:
    %
    %
    % Updated 10 May 2018
    % Gosha Tsintsadze
    % Matan Shohat
    %
    %   Update description:
    %       Changing the code to use only for generating .mat file with
    %       training and testing X and S datasets (to be used later in
    %       python based NN).
    % 
    % 
    % Updated 11 May 2018
    % Gosha Tsintsadze
    % Matan Shohat
    %
    %   Update description:
    %       Converted to function with optional parameters to better handle
    %       execution from python.
    % 
    % 
    % Updated 18 Sep 2018
    % Gosha Tsintsadze
    % Matan Shohat
    %
    %   Update description:
    %       Adding option to define parameters via function call (as
    %       varargs)
    
    %% Parsing Input
    p = inputParser;
    
    deafultAutosave = 'on';
    deafultFilename = 'shlezingerMat';
    deafultDirectory = pwd;
    
    deafult_s_fPower = 4;
    deafult_s_fNu = 4;
    deafult_s_fNt = 10;
    deafult_s_fRatio = 3;
    
    deafult_s_fT = 2^15;
    deafult_s_fD = 2^10;
    
    deafultMaxRate = 1.5;
    
    addParameter(p,'Autosave',deafultAutosave);
    addParameter(p,'FileName',deafultFilename);
    addParameter(p,'Directory',deafultDirectory);
    
    addParameter(p,'Power',deafult_s_fPower);
    addParameter(p,'Users',deafult_s_fNu);
    addParameter(p,'Antennas',deafult_s_fNt);
    addParameter(p,'Ratio',deafult_s_fRatio);
    
    addParameter(p,'TrainSetSize',deafult_s_fT);
    addParameter(p,'TestSetSize',deafult_s_fD);
    
    addParameter(p,'MaxRate',deafultMaxRate);
    
    parse(p,varargin{:});
    %% Parameters setting
    s_fPower = p.Results.Power;
    s_fNu = p.Results.Users;
    s_fNt = p.Results.Antennas;
    s_fRatio = p.Results.Ratio;

    s_fT = p.Results.TrainSetSize;
    s_fD = p.Results.TestSetSize;
    
    v_fRate = 0.01:0.01:p.Results.MaxRate;
    %% Generate training data and pilot matrix
    s_fK = 2*s_fNt * s_fNu;
    s_fN = s_fK * s_fRatio;
    
    % Pilots matrix
    s_fTau = (s_fNu*s_fRatio);  
    m_fPhi = dftmtx(s_fTau);
    m_fPhi = m_fPhi(:,1:s_fNu);
    m_fSigmaT = eye(s_fTau) + s_fPower*(m_fPhi*m_fPhi');

    % Training  and data - generate channels and observations
    m_fH = (1 / sqrt(2)) * (randn(s_fNu * s_fNt, s_fT + s_fD) + 1j*...
                                        randn(s_fNu * s_fNt, s_fT + s_fD));
    m_fW = (1 / sqrt(2)) * (randn(s_fTau * s_fNt, s_fT + s_fD) + 1j*...
                                       randn(s_fTau * s_fNt, s_fT + s_fD));
    m_fY = sqrt(s_fPower) *(kron(m_fPhi, eye(s_fNt))) * m_fH + m_fW;

    % Convert to real valued training
    trainS = [real(m_fH(:,1:s_fT)); imag(m_fH(:,1:s_fT))].';
    trainX = [real(m_fY(:,1:s_fT)); imag(m_fY(:,1:s_fT))].';
    % Convert to real valued data
    dataS = [real(m_fH(:,s_fT+1:end)); imag(m_fH(:,s_fT+1:end))].';
    dataX = [real(m_fY(:,s_fT+1:end)); imag(m_fY(:,s_fT+1:end))].';
    %% Generate Theoretical bounds
    for kk=1:length(v_fRate)
        s_fRate = v_fRate(kk);
        s_fM = floor(2^(s_fRate * s_fN));

        % No quantization:
        m_fCurves(1, kk) = 0.5 / (1 + s_fPower*s_fTau);

        % Asymptotic optimal task-based quantization:
        m_fCurves(2, kk) = (0.5 / (1 + s_fPower*s_fTau)) * ...
                         (1 + s_fPower*s_fTau * 2^(-2* s_fRatio* s_fRate));

         % Asymptotic optimal task-ignorant:
        m_fCx = 0.5 * [real(m_fSigmaT), -1*imag(m_fSigmaT); ...
                       imag(m_fSigmaT), real(m_fSigmaT)];
        m_fEstMat = sqrt(s_fPower)/(1 + s_fPower*s_fTau) * ...
                                      [real(m_fPhi'), -1*imag(m_fPhi'); ...
                                       imag(m_fPhi'), real(m_fPhi')];
        m_fCurves(3, kk) = (0.5 / (1 + s_fPower*s_fTau)) +  ...
                            1/ (2* s_fNu) * ...
                             v_fApprox(m_fCx, m_fEstMat,...
                             2^(s_fRate* s_fTau * 2));     

        % Hardware-limited upper bound:
        m_fCurves(4, kk) = (0.5 / (1 + s_fPower*s_fTau)) * ...
                            (1 + s_fPower*s_fTau * (pi * 0.5 * sqrt(3))*...
                            (floor(2^(s_fRatio* s_fRate))^(-2)));

    end
    %% Saving the mat file to file
    
    if isequal(p.Results.Autosave,'on')
        shlezMatFile = fullfile(p.Results.Directory, ...
                                [p.Results.FileName '.mat']);
        if cautionSave(shlezMatFile, dataX, dataS, trainX, trainS, ...
                       v_fRate, m_fCurves)
            return;
        end
    elseif isequal(p.Results.Autosave,'off')
        shlezFolder = uigetdir(p.Results.Directory,'Data Output Folder');
        shlezMatFile = fullfile(shlezFolder,[p.Results.FileName '.mat']);
        % If 'Cancel' choosen
        if ~shlezFolder
            return;
        end
        
        if cautionSave(shlezMatFile, dataX, dataS, trainX, trainS, ...
                       v_fRate, m_fCurves)
            return;
        end
    end
end

function canceled = cautionSave(fileDir,dataX, dataS, trainX, trainS, ...
                                v_fRate, m_fCurves)
    % Check if file exists and display a question dialog if so
    % Saving the variables to .mat file
    canceled = false;
    if exist(fileDir,'file')
        answer = questdlg(['File named "' fileDir ...
                           '" already exists.' ...
                           ' Do you want to replace it?'], ...
                          'File Exists', ...
                          'Replace','Cancel','Replace');
        switch answer
            case 'Replace'
                save(fileDir,'dataX', 'dataS', 'trainX', 'trainS', ...
                     'v_fRate', 'm_fCurves');
            case 'Cancel'
                canceled = true;
        end
    else
        save(fileDir,'dataX', 'dataS', 'trainX', 'trainS', ...
                     'v_fRate', 'm_fCurves');
    end
end