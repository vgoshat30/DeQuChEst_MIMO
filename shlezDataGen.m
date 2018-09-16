function shlezDataGen(varargin)
    % Training and testiong data generation-asymptotic massive MIMO case
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
    %                                 save to the corrent direction or to
    %                                 manually chosen one.
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
    
    %% Parsing Input
    p = inputParser;
    
    deafultAutosave = 'on';
    deafultFilename = 'shlezingerMat';
    deafultDirectory = pwd;
    
    addParameter(p,'Autosave',deafultAutosave);
    addParameter(p,'FileName',deafultFilename);
    addParameter(p,'Directory',deafultDirectory);
    
    parse(p,varargin{:});
    
    %% Parameters setting
    s_fPower = 4;
    s_fNu = 4;
    s_fNt = 10;
    s_fRatio = 3;

    s_fT = 2^15; % number of training samples
    s_fD = 2^10; % number of data samples

    %% Generate training data and pilot matrix
    % Pilots matrix
    s_fTau = (s_fNu*s_fRatio);  
    m_fPhi = dftmtx(s_fTau);
    m_fPhi = m_fPhi(:,1:s_fNu);
    m_fLMMSE =  (sqrt(s_fPower) / (1 + s_fPower*s_fTau))*...
                                                (kron(m_fPhi',eye(s_fNt)));

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

    %% Saving the mat file to file
    
    if isequal(p.Results.Autosave,'on')
        shlezMatFile = fullfile(p.Results.Directory, ...
                                [p.Results.FileName '.mat']);
        if cautionSave(shlezMatFile,dataX,dataS,trainX,trainS)
            return;
        end
    elseif isequal(p.Results.Autosave,'off')
        shlezFolder = uigetdir(p.Results.Directory,'Data Output Folder');
        shlezMatFile = fullfile(shlezFolder,[p.Results.FileName '.mat']);
        % If 'Cancel' choosen
        if ~shlezFolder
            return;
        end
        
        if cautionSave(shlezMatFile,dataX,dataS,trainX,trainS)
            return;
        end
    end
end

function canceled = cautionSave(fileDir,dataX,dataS,trainX,trainS)
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
                save(fileDir,'dataX','dataS','trainX','trainS');
            case 'Cancel'
                canceled = true;
        end
    else
        save(fileDir,'dataX','dataS','trainX','trainS');
    end
end