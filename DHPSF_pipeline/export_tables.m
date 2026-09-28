function export_tables(outputDir)
% Convert Python interchange payloads to native MATLAB tables, then verify.
files = dir(fullfile(outputDir, '*_payload.mat'));
for k = 1:numel(files)
    s = load(fullfile(files(k).folder, files(k).name));
    localizations = array2table(s.localizationMatrix, 'VariableNames', ...
        {'x1','x2','xMean','y1','y2','yMean','angleDegrees','zMicrons','frame_number','track_number'});
    if isfield(s, 'qualityNames')
        qualityNames = cellstr(s.qualityNames);
        fitQuality = array2table(s.fitQualityMatrix, 'VariableNames', qualityNames(:)');
    else
        fitQuality = array2table(s.fitQualityMatrix, 'VariableNames', ...
            {'residualRMS','minLobeSNR','jointEmitterCount','lobeSeparationPixels'});
        fitQuality.zStatus = s.zStatus;
    end
    calibrationNames = {'zMicrons','angleDegrees','beadCount','angleMADDegrees','separationPixels'};
    calibration = array2table(s.calibrationMatrix, 'VariableNames', ...
        calibrationNames(1:size(s.calibrationMatrix, 2)));
    interpolatedNames = {'zMicrons','unwrappedAngleDegrees','separationPixels'};
    interpolatedCalibration = array2table(s.interpolatedCalibration, ...
        'VariableNames', interpolatedNames(1:size(s.interpolatedCalibration, 2)));
    metadata = jsondecode(s.metadataJSON);
    if isfield(metadata, 'endpoint_risk_margin_um')
        zRange = metadata.calibration_supported_z_um;
        margin = metadata.endpoint_risk_margin_um;
        fitQuality.zEndpointRisk = localizations.zMicrons < zRange(1)+margin | ...
            localizations.zMicrons > zRange(2)-margin;
    else
        fitQuality.zEndpointRisk = abs(localizations.zMicrons) > 25;
    end
    localizations.Properties.VariableUnits = {'pixel','pixel','pixel','pixel','pixel','pixel','degree','micron','',''};
    vars = {'localizations','fitQuality','calibration','interpolatedCalibration','metadata'};
    if isfield(s, 'correctedMatrix')
        corrected = array2table(s.correctedMatrix, 'VariableNames', {'xCorrected','yCorrected'});
        vars = [vars, {'corrected'}];
    end
    if isfield(s, 'stabilizedMatrix')
        stabilized = array2table(s.stabilizedMatrix, 'VariableNames', {'xStabilized','yStabilized','zStabilized'});
        drift = array2table(s.driftMatrix, 'VariableNames', {'frame_number','dx','dy','dz'});
        vars = [vars, {'stabilized','drift'}];
    end
    name = strrep(files(k).name, '_payload.mat', '_localizations.mat');
    dest = fullfile(outputDir, name);
    save(dest, vars{:}, '-v7.3');
    check = load(dest);
    assert(istable(check.localizations) && width(check.localizations) == 10);
    assert(height(check.localizations) == size(s.localizationMatrix, 1));
    assert(isequaln(check.localizations{:,:}, s.localizationMatrix));
    fprintf('Verified %s: %d localization rows, native MATLAB table.\n', name, height(localizations));
end
end
