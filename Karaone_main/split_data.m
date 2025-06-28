%
% [epoched_data] = split_data(inds, data)
%
% input:
%   inds            - A 1xE cell aray, where element e contains the indices for
%                   epoch e.
%   data            - A CxN matrix, where C and N are the number of channels and
%                   samples, respectively.
% output:
%   epoched_data    - A 1xE cell array containing the epoched data specified by
%                   inds.
function [epoched_data] = split_data(inds, data)
    
    num_epochs = size(inds, 2);
    epoched_data = cell(1, num_epochs);
    for i=1:num_epochs
        indices = inds{i};
        epoched_data{i} = data(:, indices(1):indices(2));
    end
end
