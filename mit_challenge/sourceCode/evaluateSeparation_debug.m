% Research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator
% and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the
% authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the
% U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright
% notation herein.

%% EVALUATESEPARATION_DEBUG
% Debug-oriented version of evaluateSeparation which preserves per-frame and
% per-output symbol recovery information and writes summary CSV/MAT outputs.
%
% Outputs:
% frameSuccessRate: fraction of frames decoded with zero errors
% ber: overall bit-error-rate after signal separation
% debugOut: struct containing per-frame/per-output recovery artifacts

function [frameSuccessRate, ber, debugOut] = evaluateSeparation_debug(outputDirectory, alphaIndex, frameLen, setIndex, nSources, params, debugDirectory)

%% Input Error Checking
minValAlpha = 1; maxValAlpha = 25; validateInput(alphaIndex, minValAlpha, maxValAlpha, 'alphaIndex');
minValFrameLen = 1; maxValFrameLen = 128; validateInput(frameLen, minValFrameLen, maxValFrameLen, 'frameLen');
minValSetIndex = 1; maxValSetIndex = 20; validateInput(setIndex, minValSetIndex, maxValSetIndex, 'setIndex');

%% Misc Setup and Parameters
nwords = frameLen;
plotFlag = false;

load 'pulseShaping.mat' gRRC

nR = params.nR;
M = params.M;
m = params.m;
n = params.n;
k = params.k;
t = params.t;

nPreambleBits = params.nPreambleBits;
sps = params.sps;
spanSyms = params.spanSyms;
rolloffFactor = params.rolloffFactor;
txSymsPreamble = params.txSymsPreamble;

noisePwr = params.noisePwr;
nFrames = params.nFrames;
nOutputs = 1;

suffixStr = {'A', 'B'};

debugOut = struct();
debugOut.alphaIndex = alphaIndex;
debugOut.frameLen = frameLen;
debugOut.setIndex = setIndex;
debugOut.nSources = nSources;
debugOut.nFrames = nFrames;
debugOut.nR = nR;
debugOut.modOrder = M;
debugOut.nwords = nwords;
debugOut.numErrors = NaN(nFrames, nSources);
debugOut.bestOutputIndex = NaN(nFrames, 1);
debugOut.bestNumErrors = NaN(nFrames, 1);
debugOut.success = zeros(nFrames, 1);
debugOut.rxSymsPayload = cell(nFrames, nSources);
debugOut.rxSymsInt = cell(nFrames, nSources);
debugOut.payloadMeanAbs = NaN(nFrames, nSources);
debugOut.payloadStdAbs = NaN(nFrames, nSources);
debugOut.payloadPhaseStd = NaN(nFrames, nSources);
debugOut.payloadPowerMean = NaN(nFrames, nSources);

frameSummaryHeaders = {'alphaIndex', 'frameLen', 'setIndex', 'frame_number', 'best_output_index', 'best_num_errors', 'success'};
frameSummary = NaN(nFrames, length(frameSummaryHeaders));

outputSummaryHeaders = {'alphaIndex', 'frameLen', 'setIndex', 'frame_number', 'output_index', 'numErrors', 'success', 'payload_mean_abs', 'payload_std_abs', 'payload_phase_std', 'payload_power_mean'};
outputSummary = NaN(nFrames * nSources, length(outputSummaryHeaders));
outputSummaryRow = 1;

for ff = 1:nFrames

    numErrors = NaN(1, nSources);

    for cc = 1:nSources

        currSuffix = suffixStr{cc};
        separatedFilename = [outputDirectory, filesep, 'output', currSuffix, '_frameLen_', num2str(frameLen), '_setIndex_', num2str(setIndex), '_alphaIndex_', num2str(alphaIndex), '_frame', num2str(ff)];
        Y0 = readData_oct(separatedFilename, 1, 'iqdata');

        %% Perform matched filtering with root-raised-cosine filtering
        Y = conv(Y0(1, :), gRRC, 'same');

        %% Downsample to Nyquist
        rxSyms = Y(:, sps:sps:end);

        %% Determine number of preamble symbols
        nSymsPreamble = nPreambleBits ./ log2(M);

        %% Extract the preamble
        rxSymsPreamble = rxSyms(:, 1:nSymsPreamble);

        %% Obtain Channel Estimate Hest by integrating over preamble symbols for each Rx antenna
        hratio = rxSymsPreamble ./ repmat(txSymsPreamble, nOutputs, 1);
        Hest = mean(hratio, 2);

        %% Obtain the payload coded symbols
        rxSymsPayload0 = rxSyms(:, nSymsPreamble + 1:end);

        %% Assume oracle knowledge of noise power
        noiseEst = noisePwr;

        %% MMSE Estimation of Tx Symbols
        rxSymsPayload = Hest' * ((Hest * Hest' + noiseEst .* eye(nOutputs)) \ rxSymsPayload0);

        %% Normalize to unit average power
        rxSymsPayload = rxSymsPayload ./ sqrt(mean(abs(rxSymsPayload) .^ 2));

        if (plotFlag)
            scatterplot(rxSymsPayload)
        end

        %% Integer representation of payload symbols
        rxSymsInt = pskdemod(rxSymsPayload, M, 0, 'gray');

        %% Convert coded symbols from integer to binary representation
        rxCodedBits = de2bi(rxSymsInt, log2(M)).';

        %% Vectorize rxCodedBits to column vector
        rxCodedBitsVect = rxCodedBits(:);

        %% Organize coded bits into codewords
        rxCodedBitsCodewords = reshape(gf(rxCodedBitsVect), nwords, n);

        %% Perform BCH decoding
        rxMsg = bchdeco(rxCodedBitsCodewords, k, t);

        %% Compute the number of errors after decoding
        txMsg = gf(params.trueBits(:, :, ff));
        numErrors(cc) = sum(sum(abs(rxMsg - txMsg.x)));

        payloadAbs = abs(rxSymsPayload);
        debugOut.rxSymsPayload{ff, cc} = rxSymsPayload;
        debugOut.rxSymsInt{ff, cc} = rxSymsInt;
        debugOut.numErrors(ff, cc) = numErrors(cc);
        debugOut.payloadMeanAbs(ff, cc) = mean(payloadAbs);
        debugOut.payloadStdAbs(ff, cc) = std(payloadAbs);
        debugOut.payloadPhaseStd(ff, cc) = std(angle(rxSymsPayload));
        debugOut.payloadPowerMean(ff, cc) = mean(payloadAbs .^ 2);

        outputSummary(outputSummaryRow, :) = [
            alphaIndex, ...
            frameLen, ...
            setIndex, ...
            ff, ...
            cc, ...
            numErrors(cc), ...
            double(numErrors(cc) == 0), ...
            debugOut.payloadMeanAbs(ff, cc), ...
            debugOut.payloadStdAbs(ff, cc), ...
            debugOut.payloadPhaseStd(ff, cc), ...
            debugOut.payloadPowerMean(ff, cc)
        ];
        outputSummaryRow = outputSummaryRow + 1;

    end

    [minErr, bestIdx] = min(numErrors);
    errorCount(ff) = minErr;
    debugOut.bestOutputIndex(ff) = bestIdx;
    debugOut.bestNumErrors(ff) = minErr;
    debugOut.success(ff) = double(minErr == 0);
    frameSummary(ff, :) = [alphaIndex, frameLen, setIndex, ff, bestIdx, minErr, double(minErr == 0)];

    disp(['Number of errors ', num2str(minErr)])
end

ber = mean(errorCount) ./ numel(rxMsg);
frameSuccessRate = length(find(errorCount == 0)) ./ nFrames;

debugOut.frameSuccessRate = frameSuccessRate;
debugOut.ber = ber;
debugOut.frameSummaryHeaders = frameSummaryHeaders;
debugOut.frameSummary = frameSummary;
debugOut.outputSummaryHeaders = outputSummaryHeaders;
debugOut.outputSummary = outputSummary;

if (nargin < 7 || isempty(debugDirectory))
    debugDirectory = getDebugDirectory(outputDirectory);
end
if (~exist(debugDirectory, 'dir'))
    mkdir(debugDirectory);
end

baseFilename = [debugDirectory, filesep, 'frameLen_', num2str(frameLen), '_setIndex_', num2str(setIndex), '_alphaIndex_', num2str(alphaIndex)];
matFilename = [baseFilename, '_debug.mat'];
frameCsvFilename = [baseFilename, '_perFrame.csv'];
outputCsvFilename = [baseFilename, '_perOutput.csv'];

save(matFilename, 'debugOut', '-v7');
writeSummaryCsv(frameCsvFilename, frameSummaryHeaders, frameSummary);
writeSummaryCsv(outputCsvFilename, outputSummaryHeaders, outputSummary);

disp(['Saved debug MAT file to ', matFilename])
disp(['Saved per-frame CSV to ', frameCsvFilename])
disp(['Saved per-output CSV to ', outputCsvFilename])
disp(['Bit Error Rate is ', num2str(ber)])
disp(['Frame Success Rate is ', num2str(frameSuccessRate)])

end

function debugDirectory = getDebugDirectory(outputDirectory)
[parentDirectory, ~, ~] = fileparts(outputDirectory);

if (isempty(parentDirectory))
    parentDirectory = '.';
end

debugDirectory = [parentDirectory, filesep, 'debugEval'];
end

function writeSummaryCsv(csvFilename, headers, summaryMatrix)
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
