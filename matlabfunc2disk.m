function save_pth = matlabfunc2disk(function_parts, save_pth)
    save_pth = string(save_pth);
    if isempty(function_parts)
        error('No function name provided');
    end
    
    functionName = function_parts{1};  % The first string is the function name 
    % Verify that the function name exists
    if ~exist(functionName, 'file')
        error('Function %s does not exist or is not on the MATLAB path', functionName);
    end
    % Display the function name
    disp(['Function name: ', functionName]);
    for i = 2:length(function_parts)
        disp(['Argument ', num2str(i-1), ': ', function_parts{i}]);
    end

    returned_struct = feval(functionName, function_parts{2:end});
    save(save_pth, 'returned_struct');
end