% Research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator
% and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the
% authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the
% U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright
% notation herein.

%% EVALDEBUGMAIN
% Purpose: run the reference ICA separator and write per-frame debug
% artifacts that are easy to inspect from Python/Jupyter.

function evalDebugMain()

clc; close all;

frameLenVect = [4];
setList = [1];

separationOpt = 'ICA';
nSources = 2;
nR = 4;
NalphaIndices = 15;
nFrames = 100;

if (exist('OCTAVE_VERSION', 'builtin'))
    pkg load communications
end

inputDirectory = findMixtureDirectory(frameLenVect(1), setList(1), 1);
paramDirectory = findParamDirectory(frameLenVect(1));
outputDirectory = ['..', filesep, 'sepOutput'];
debugDirectory = ['..', filesep, 'debugEval'];

if (~exist(outputDirectory, 'dir'))
    mkdir(outputDirectory);
end

if (~exist(debugDirectory, 'dir'))
    mkdir(debugDirectory);
end

rand('seed', 123456789)

summaryHeaders = {'frameLen', 'alphaIndex', 'setIndex', 'frameSuccessRate', 'ber'};
summaryRows = [];

for ww = 1:length(frameLenVect)
    frameLen = frameLenVect(ww);

    load([paramDirectory, filesep, 'soiParams', num2str(frameLen), '.mat'], 'soiParamsPart')

    for ss = 1:NalphaIndices
        for ii = 1:length(setList)

            currSet = setList(ii);

            [~, ~, ~] = sigSeparator(inputDirectory, outputDirectory, separationOpt, ss, frameLen, currSet, nFrames, nR);

            currParams = soiParamsPart(ss, currSet);
            currParams.nFrames = nFrames;

            [frameSuccessRateM(ww, ss, ii), berM(ww, ss, ii), debugOut] = evaluateSeparation_debug(outputDirectory, ss, frameLen, currSet, nSources, currParams);
            summaryRows = [summaryRows; frameLen, ss, currSet, debugOut.frameSuccessRate, debugOut.ber];
        end
    end
end

writeDebugRunSummary([debugDirectory, filesep, 'debug_run_summary.csv'], summaryHeaders, summaryRows);
save([debugDirectory, filesep, 'debug_results.mat'], 'frameSuccessRateM', 'berM', 'frameLenVect', 'setList', 'summaryRows', 'summaryHeaders', '-v7')

disp(['Saved debug run summary to ', [debugDirectory, filesep, 'debug_run_summary.csv']])

end

function writeDebugRunSummary(csvFilename, headers, summaryMatrix)
fid = fopen(csvFilename, 'w');

if (fid == -1)
    error(['Could not open CSV file ', csvFilename, ' for writing']);
end

for cc = 1:length(headers)
    if (cc < length(headers))
        fprintf(fid, '%s,', headers{cc});
    else
        fprintf(fid, '%s\n', headers{cc});
    end
end

for rr = 1:size(summaryMatrix, 1)
    for cc = 1:size(summaryMatrix, 2)
        if (cc < size(summaryMatrix, 2))
            fprintf(fid, '%.12g,', summaryMatrix(rr, cc));
        else
            fprintf(fid, '%.12g\n', summaryMatrix(rr, cc));
        end
    end
end

fclose(fid);
end

function inputDirectory = findMixtureDirectory(frameLen, setIndex, alphaIndex)
candidateDirs = {
    ['..', filesep, 'mixtureData'], ...
    ['..', filesep, 'rfchallenge_multichannel_starter-main', filesep, 'mixtureData'], ...
    ['..', filesep, 'rfchallenge_multichannel_starter-main', filesep, 'mixtureData', filesep, 'rfChallenge_multisensor_frameLen_4', filesep, 'rfChallenge_multisensor_frameLen_4_setIndex_1']
};

requiredName = ['input_frameLen_', num2str(frameLen), '_setIndex_', num2str(setIndex), '_alphaIndex_', num2str(alphaIndex), '_frame1.iqdata'];

for cc = 1:length(candidateDirs)
    probeFilename = [candidateDirs{cc}, filesep, requiredName];
    if (exist(probeFilename, 'file'))
        inputDirectory = candidateDirs{cc};
        disp(['Using mixtureData directory ', inputDirectory])
        return
    end
end

error(['Could not locate backend mixtureData containing ', requiredName, '. Checked ', formatCandidateDirs(candidateDirs), '.']);
end

function paramDirectory = findParamDirectory(frameLen)
candidateDirs = {
    ['..', filesep, 'soiParamFiles'], ...
    ['..', filesep, 'rfchallenge_multichannel_starter-main', filesep, 'soiParamFiles']
};

requiredName = ['soiParams', num2str(frameLen), '.mat'];

for cc = 1:length(candidateDirs)
    probeFilename = [candidateDirs{cc}, filesep, requiredName];
    if (exist(probeFilename, 'file'))
        paramDirectory = candidateDirs{cc};
        disp(['Using soiParamFiles directory ', paramDirectory])
        return
    end
end

error(['Could not locate backend soiParamFiles containing ', requiredName, '. Checked ', formatCandidateDirs(candidateDirs), '.']);
end

function formatted = formatCandidateDirs(candidateDirs)
formatted = candidateDirs{1};

for cc = 2:length(candidateDirs)
    formatted = [formatted, ', ', candidateDirs{cc}];
end
end
