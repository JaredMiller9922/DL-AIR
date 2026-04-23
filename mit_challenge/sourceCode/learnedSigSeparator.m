function [Y0, sigIn, W0] = learnedSigSeparator(inputDirectory, outputDirectory, methodStr, alphaIndex, frameLen, setIndex, nFrames, nR, modelName, checkpointPath)

if (nargin < 9 || isempty(modelName))
    modelName = 'Hybrid';
end

thisFile = mfilename('fullpath');
thisDir = fileparts(thisFile);
projectRoot = fullfile(thisDir, '..', '..');
pyScript = fullfile(thisDir, '..', 'infer_mit_separator.py');

if (nargin < 10 || isempty(checkpointPath))
    checkpointPath = defaultCheckpointPath(projectRoot, modelName);
end

pyExe = getenv('PYTHON_EXE');
if (isempty(pyExe))
    pyExe = 'D:\Anaconda\envs\imitation_learning\python.exe';
end

cmd = sprintf('"%s" "%s" --input_dir "%s" --output_dir "%s" --alphaIndex %d --frameLen %d --setIndex %d --nFrames %d --model_name "%s" --checkpoint_path "%s" --n_rx %d', ...
    pyExe, pyScript, inputDirectory, outputDirectory, alphaIndex, frameLen, setIndex, nFrames, modelName, checkpointPath, nR);

disp(['Running learned separator command: ', cmd])
status = system(cmd);
if (status ~= 0)
    error('Python learned separator failed');
end

Y0 = [];
sigIn = [];
W0 = [];

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
