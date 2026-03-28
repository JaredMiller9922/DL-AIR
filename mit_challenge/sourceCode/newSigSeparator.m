function [a,b,c] = newSigSeparator(inputDirectory, outputDirectory, methodStr, alphaIndex, frameLen, setIndex, nFrames, nR)

% --- resolve paths relative to THIS .m file location ---
thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);

pyScript = fullfile(thisDir, '..', 'infer_and_write_sep.py');
ckptPath = fullfile(thisDir, '..', 'hybrid_model.pt');

cmd = sprintf("python '%s' --input_dir '%s' --output_dir '%s' --alphaIndex %d --frameLen %d --setIndex %d --nFrames %d --ckpt '%s'",
    pyScript, inputDirectory, outputDirectory, alphaIndex, frameLen, setIndex, nFrames, ckptPath);

status = system(cmd);
if(status ~= 0)
  error("Python separator failed");
end

a=[]; b=[]; c=[];
end