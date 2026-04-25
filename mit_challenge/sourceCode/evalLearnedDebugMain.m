% Run a learned separator through the MIT debug evaluation path.
% FastICA evalDebugMain remains unchanged; this writes learned artifacts to
% separate output/debug folders so the baseline remains intact.

function evalLearnedDebugMain(NalphaIndices, nFrames, modelName, checkpointPath)

clc; close all;

if (nargin < 1 || isempty(NalphaIndices))
    NalphaIndices = 1;
end
if (nargin < 2 || isempty(nFrames))
    nFrames = 5;
end
if (nargin < 3 || isempty(modelName))
    modelName = 'Hybrid';
end

thisFile = mfilename('fullpath');
thisDir = fileparts(thisFile);
projectRoot = fullfile(thisDir, '..', '..');

if (nargin < 4 || isempty(checkpointPath))
    checkpointPath = defaultCheckpointPath(projectRoot, modelName);
end

if (exist('OCTAVE_VERSION', 'builtin'))
    pkg load communications
end

frameLenVect = [4];
setList = [1];
nSources = 2;
nR = 4;

inputDirectory = ['..', filesep, 'rfchallenge_multichannel_starter-main', filesep, 'mixtureData', filesep, 'rfChallenge_multisensor_frameLen_4', filesep, 'rfChallenge_multisensor_frameLen_4_setIndex_1'];
paramDirectory = ['..', filesep, 'rfchallenge_multichannel_starter-main', filesep, 'soiParamFiles'];
safeModelName = regexprep(modelName, '[^A-Za-z0-9_]', '_');
outputDirectory = ['..', filesep, 'sepOutput_learned_', safeModelName];
debugDirectory = ['..', filesep, 'debugEval_learned_', safeModelName];

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

            learnedSigSeparator(inputDirectory, outputDirectory, 'learned', ss, frameLen, currSet, nFrames, nR, modelName, checkpointPath);

            currParams = soiParamsPart(ss, currSet);
            currParams.nFrames = nFrames;

            [frameSuccessRateM(ww, ss, ii), berM(ww, ss, ii), debugOut] = evaluateSeparation_debug(outputDirectory, ss, frameLen, currSet, nSources, currParams, debugDirectory);
            summaryRows = [summaryRows; frameLen, ss, currSet, debugOut.frameSuccessRate, debugOut.ber];
        end
    end
end

writeDebugRunSummary([debugDirectory, filesep, 'debug_run_summary.csv'], summaryHeaders, summaryRows);
save([debugDirectory, filesep, 'debug_results.mat'], 'frameSuccessRateM', 'berM', 'frameLenVect', 'setList', 'summaryRows', 'summaryHeaders', 'modelName', 'checkpointPath', '-v7')

disp(['Saved learned debug run summary to ', [debugDirectory, filesep, 'debug_run_summary.csv']])

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

function checkpointPath = defaultCheckpointPath(projectRoot, modelName)
safeName = lower(strrep(modelName, '-', '_'));
safeName = strrep(safeName, ' ', '_');

if (strcmp(safeName, 'iqcnn'))
    safeName = 'iq_cnn';
end
if (strcmp(safeName, 'rfhtdemucs'))
    safeName = 'htdemucs';
end

checkpointPath = fullfile(projectRoot, 'pytorch_models', [safeName, '_model.pt']);
end
