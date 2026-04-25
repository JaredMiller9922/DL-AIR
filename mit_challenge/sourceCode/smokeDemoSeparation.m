% Tiny smoke test for the five backend-format demo mixture files.
% This does not run the full evaluation sweep.

function smokeDemoSeparation()

clc;

inputDirectory = 'D:\CS 6955\CS-6955\CS-6953\Demo_2\Data';
outputDirectory = ['..', filesep, 'sepOutput'];
paramDirectory = ['..', filesep, 'rfchallenge_multichannel_starter-main', filesep, 'soiParamFiles'];

frameLen = 128;
setIndex = 1;
alphaIndex = 1;
nFrames = 5;
nR = 4;
nSources = 2;

if (exist('OCTAVE_VERSION', 'builtin'))
    pkg load communications
end

if (~exist(inputDirectory, 'dir'))
    error(['Input directory not found: ', inputDirectory]);
end

if (~exist(outputDirectory, 'dir'))
    mkdir(outputDirectory);
end

inputFiles = dir([inputDirectory, filesep, 'input_frameLen_', num2str(frameLen), '_setIndex_', num2str(setIndex), '_alphaIndex_', num2str(alphaIndex), '_frame*.iqdata']);
disp(['Input files found: ', num2str(length(inputFiles))])

if (length(inputFiles) < nFrames)
    error(['Expected at least ', num2str(nFrames), ' demo input files, found ', num2str(length(inputFiles))]);
end

disp(['Running sigSeparator on ', num2str(nFrames), ' frames'])
sigSeparator(inputDirectory, outputDirectory, 'ICA', alphaIndex, frameLen, setIndex, nFrames, nR);

outputA = dir([outputDirectory, filesep, 'outputA_frameLen_', num2str(frameLen), '_setIndex_', num2str(setIndex), '_alphaIndex_', num2str(alphaIndex), '_frame*.iqdata']);
outputB = dir([outputDirectory, filesep, 'outputB_frameLen_', num2str(frameLen), '_setIndex_', num2str(setIndex), '_alphaIndex_', num2str(alphaIndex), '_frame*.iqdata']);
disp(['outputA files found: ', num2str(length(outputA))])
disp(['outputB files found: ', num2str(length(outputB))])
disp(['Total separated output files found: ', num2str(length(outputA) + length(outputB))])

if (~isempty(outputA))
    disp(['Sample outputA file: ', outputA(1).name])
end

if (~isempty(outputB))
    disp(['Sample outputB file: ', outputB(1).name])
end

soiParamsFilename = [paramDirectory, filesep, 'soiParams', num2str(frameLen), '.mat'];
statsFilename = [paramDirectory, filesep, 'stats', num2str(frameLen), '.mat'];

if (exist(soiParamsFilename, 'file') && exist(statsFilename, 'file'))
    try
        disp(['Attempting evaluateSeparation_debug using ', soiParamsFilename])
        load(soiParamsFilename, 'soiParamsPart')
        currParams = soiParamsPart(alphaIndex, setIndex);
        currParams.nFrames = nFrames;
        [frameSuccessRate, ber, debugOut] = evaluateSeparation_debug(outputDirectory, alphaIndex, frameLen, setIndex, nSources, currParams);
        disp(['Debug evaluation completed. frameSuccessRate=', num2str(frameSuccessRate), ', ber=', num2str(ber)])
        disp(['Debug rows: ', num2str(size(debugOut.frameSummary, 1)), ' frame rows, ', num2str(size(debugOut.outputSummary, 1)), ' output rows'])
    catch evalErr
        disp(['Debug evaluation skipped/failed after separation output generation: ', evalErr.message])
    end
else
    disp(['Debug evaluation skipped. Missing ', soiParamsFilename, ' or ', statsFilename])
end

end
