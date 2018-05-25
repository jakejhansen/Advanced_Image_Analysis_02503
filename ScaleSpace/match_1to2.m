function [result] = match_1to2(dan, dbn)
matches1to2 = {};
for i = 1:size(dan,2)
    i
    matches = [];
    for j = 1:size(dbn,2)
        matches = [matches, norm(dan(:,i) - dbn(:,j))];
    end
    [Asort, Isort] = sort(matches);
    matches1to2{i} = Isort(1:2);
end

result = matches1to2;
end

