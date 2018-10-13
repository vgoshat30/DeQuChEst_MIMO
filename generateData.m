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
    %   generateData('Autosave','on') Displays a folder choosing window
    %                                 before saving the .mat file. Call
    %                                 with 'off' or ommit to automatically
    %                                 save to the corrent directory or to
    %                                 manually chose one.
    %
    %
    %   generateData('FileName','shlezMat') Choose the file name of the
    %                                       output .mat file. DO NOT add
    %                                       .mat extention.
    %
    %
    %   generateData('Directory','shlezMat') Specify the directory of the
    %                                        output file.
    %
    %
    %
    %
    %   Specify Data parameters:
    %
    %
    %
    %   generateData('MaxRate', 5)          Deafult is: 1.5
    %
    %   generateData('TrainPower', 1:5)     Deafult is: 1:10
    %
    %   generateData('TestPower', 3)        Deafult is: 4
    %
    %   generateData('Users', 5)            Deafult is: 4
    %
    %   generateData('Antennas', 15)        Deafult is: 10
    %
    %   generateData('Ratio', 4)            Deafult is: 3
    %
    %   generateData('TrainSetSize', 2^5)  Deafult is: 2^15
    %
    %   generateData('TestSetSize', 2^5)   Deafult is: 2^10
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
    % 
    % 
    % Updated 4 Oct 2018
    % Nir Shlezinger
    %
    %   Update description:
    %       Creating training data based on multiple SNR values.
    
    %% Parsing Input
    p = inputParser;
    
    deafultAutosave = 'on';
    deafultFilename = 'data';
    deafultDirectory = pwd;
    
    deafult_s_fTestPower = 4;
    deafult_s_fTrainPower = 1:10;
    deafult_s_fNu = 4;
    deafult_s_fNt = 10;
    deafult_s_fRatio = 3;
    
    deafult_s_fT = 2^15;
    deafult_s_fD = 2^10;
    
    deafultMaxRate = 1.5;
    
    addParameter(p,'Autosave',deafultAutosave);
    addParameter(p,'FileName',deafultFilename);
    addParameter(p,'Directory',deafultDirectory);
    
    addParameter(p,'TestPower',deafult_s_fTestPower);
    addParameter(p,'TrainPower',deafult_s_fTrainPower);
    addParameter(p,'Users',deafult_s_fNu);
    addParameter(p,'Antennas',deafult_s_fNt);
    addParameter(p,'Ratio',deafult_s_fRatio);
    
    addParameter(p,'TrainSetSize',deafult_s_fT);
    addParameter(p,'TestSetSize',deafult_s_fD);
    
    addParameter(p,'MaxRate',deafultMaxRate);
    
    parse(p,varargin{:});
    %% Parameters setting
    s_fTestPower = p.Results.TestPower;
    v_fTrainPower = p.Results.TrainPower; % Connsiered SNR Range
    s_fNu = p.Results.Users;
    s_fNt = p.Results.Antennas;
    s_fRatio = p.Results.Ratio;

    s_fT_total = p.Results.TrainSetSize;
    s_nPartition = ceil(s_fT_total / length(v_fTrainPower));
    s_fT = s_nPartition * length(v_fTrainPower);
    
    s_fD = p.Results.TestSetSize;
    
    v_fRate = 0.01:0.001:p.Results.MaxRate;
    %% Generate training data and pilot matrix
    s_fK = 2*s_fNt * s_fNu;
    s_fN = s_fK * s_fRatio;
    
    % Pilots matrix
    s_fTau = (s_fNu*s_fRatio);  
    m_fPhi = dftmtx(s_fTau);
    m_fPhi = m_fPhi(:,1:s_fNu);
    m_fSigmaT = eye(s_fTau) + s_fTestPower*(m_fPhi*m_fPhi');

    % Data generate channels and observations
    m_fH = (1 / sqrt(2)) * (randn(s_fNu * s_fNt, s_fD) + 1j*...
                                        randn(s_fNu * s_fNt,  s_fD));
    m_fW = (1 / sqrt(2)) * (randn(s_fTau * s_fNt, s_fD) + 1j*...
                                       randn(s_fTau * s_fNt,  s_fD));
    m_fY = sqrt(s_fTestPower) *(kron(m_fPhi, eye(s_fNt))) * m_fH + m_fW;

       % Convert to real valued data
    dataS = [real(m_fH); imag(m_fH)].';
    dataX = [real(m_fY); imag(m_fY)].';
    
    % Convert to real valued training
    trainS = zeros(s_fT,s_fK);
    trainX = zeros(s_fT,s_fN);
     % Data generate channels and observations
     for ii=1:length(v_fTrainPower)
         s_fTrainPower = v_fTrainPower(ii);
         m_fH = (1 / sqrt(2)) * (randn(s_fNu * s_fNt, s_nPartition) + 1j*...
             randn(s_fNu * s_fNt,  s_nPartition));
         m_fW = (1 / sqrt(2)) * (randn(s_fTau * s_fNt, s_nPartition) + 1j*...
             randn(s_fTau * s_fNt,  s_nPartition));
         m_fY = sqrt(s_fTrainPower) *(kron(m_fPhi, eye(s_fNt))) * m_fH + m_fW;
         % Convert to real valued training
         trainS((1+(ii-1)*s_nPartition): (ii*s_nPartition),:) = [real(m_fH); imag(m_fH)].';
         trainX((1+(ii-1)*s_nPartition): (ii*s_nPartition),:) = [real(m_fY); imag(m_fY)].';
     end
    

 
    %% Generate Theoretical bounds
    for kk=1:length(v_fRate)
        s_fRate = v_fRate(kk);
        s_fM = floor(2^(s_fRate * s_fN));

        % No quantization:
        m_fCurves(1, kk) = 0.5 / (1 + s_fTestPower*s_fTau);

        % Asymptotic optimal task-based quantization:
        m_fCurves(2, kk) = (0.5 / (1 + s_fTestPower*s_fTau)) * ...
                         (1 + s_fTestPower*s_fTau * 2^(-2* s_fRatio* s_fRate));

        % Asymptotic optimal task-ignorant:
        m_fCx = 0.5 * [real(m_fSigmaT), -1*imag(m_fSigmaT); ...
                       imag(m_fSigmaT), real(m_fSigmaT)];
        m_fEstMat = sqrt(s_fTestPower)/(1 + s_fTestPower*s_fTau) * ...
                                      [real(m_fPhi'), -1*imag(m_fPhi'); ...
                                       imag(m_fPhi'), real(m_fPhi')];
        m_fCurves(3, kk) = (0.5 / (1 + s_fTestPower*s_fTau)) +  ...
                            1/ (2* s_fNu) * ...
                             v_fApprox(m_fCx, m_fEstMat,...
                             2^(s_fRate* s_fTau * 2));     

        % Hardware-limited upper bound:
        m_fCurves(4, kk) = (0.5 / (1 + s_fTestPower*s_fTau)) * ...
                            (1 + s_fTestPower*s_fTau * (pi * 0.5 * sqrt(3))*...
                            (floor(2^(s_fRatio* s_fRate))^(-2)));

    end
    %% Saving the mat file to file
    
    if isequal(p.Results.Autosave,'on')
        shlezMatFile = fullfile(p.Results.Directory, ...
                                [p.Results.FileName '.mat']);
        if cautionSave(shlezMatFile, dataX, dataS, trainX, trainS, ...
                       v_fRate, m_fCurves, s_fTestPower, v_fTrainPower, ...
                       s_fNu, s_fNt, s_fRatio, s_fT_total, s_fD)
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
                       v_fRate, m_fCurves, s_fTestPower, v_fTrainPower, ...
                       s_fNu, s_fNt, s_fRatio, s_fT_total, s_fD)
            return;
        end
    end
end

function canceled = cautionSave(fileDir,dataX, dataS, trainX, trainS, ...
                                v_fRate, m_fCurves, s_fTestPower, ...
                                v_fTrainPower, s_fNu, s_fNt, s_fRatio, ...
                                s_fT, s_fD)
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
                     'v_fRate', 'm_fCurves', 's_fTestPower', ...
                     'v_fTrainPower', 's_fNu', 's_fNt', 's_fRatio', ...
                     's_fT', 's_fD');
            case 'Cancel'
                canceled = true;
        end
    else
        save(fileDir,'dataX', 'dataS', 'trainX', 'trainS', ...
                     'v_fRate', 'm_fCurves', 's_fTestPower', ...
                     'v_fTrainPower', 's_fNu', 's_fNt', 's_fRatio', ...
                     's_fT', 's_fD');
    end
end