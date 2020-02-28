function plot_keyframe

opt = globals();

color = {'m', 'c', 'b', 'g', 'r'};
leng = {'PoseCNN', 'PoseCNN+ICP', 'Per-Pixel DF', 'Iterative DF', 'Ours'};
aps = zeros(5, 1);
lengs = cell(5, 1);
close all;

% load results
object = load('results_keyframe_plot.mat');
distances = object.distances;
rotation_sym = object.rotation_sym;
rotation_non = object.rotation_non;
translations = object.errors_translation;

index_plot = [1, 2, 3, 4, 5];
[hf, pos] = tight_subplot(2, 2, [.15 .05], [.1 .05], [.06 .03]);
font_size = 5;
legend_font_size = 3;
max_distance = 0.1;
max_rotation = 180;
max_rot_dist = 0.04;


index = 1:size(distances, 1);
index_non = 1:size(rotation_non, 1);
index_sym = 1:size(rotation_sym, 1);

% distances
axes(hf(1))
for i = index_plot
    D = distances(index, i);
    D(D > max_distance) = inf;
    d = sort(D);
    n = numel(d);
    accuracy = cumsum(ones(1, n)) / n;
    if i== 2 || i==4
        plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
    else
        plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
    end
    aps(i) = VOCap(d, accuracy);
    lengs{i} = sprintf('%s (AUC:%.2f)', leng{i}, aps(i)*100);
    hold on;
end
hold off;
h = legend(lengs(index_plot), 'Location', 'southeast');
set(h, 'FontSize', font_size);
h = xlabel('Threshold in Meter');
set(h, 'FontSize', font_size);
h = ylabel('Accuracy');
set(h, 'FontSize', font_size);
h = title('ADD(-S) of All Objects', 'Interpreter', 'none');
set(h, 'FontSize', font_size);
set(gca, 'FontSize', font_size)
set(gca,'XLim',[0 0.1])
set(gca,'YLim',[0 1])
pbaspect([1 0.75 1])

% translation
axes(hf(2))
for i = index_plot
    D = translations(index, i);
    D(D > max_distance) = inf;
    d = sort(D);
    n = numel(d);
    accuracy = cumsum(ones(1, n)) / n;
    if i== 2 || i==4
        plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
    else
        plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
    end
    aps(i) = VOCap(d, accuracy);
    lengs{i} = sprintf('%s (AUC:%.2f)', leng{i}, aps(i)*100);
    hold on;
end
hold off;
h = legend(lengs(index_plot), 'Location', 'southeast');
set(h, 'FontSize', font_size);
h = xlabel('Threshold in Meter');
set(h, 'FontSize', font_size);
h = ylabel('Accuracy');
set(h, 'FontSize', font_size);
h = title('Translation of All Objects', 'Interpreter', 'none');
set(h, 'FontSize', font_size);
set(gca, 'FontSize', font_size)
set(gca,'XLim',[0 0.1])
set(gca,'YLim',[0 1])
pbaspect([1 0.75 1])


% rotation_sym
axes(hf(3))
for i = index_plot
    D = rotation_sym(index_sym, i);
    D(D > max_rot_dist) = inf;
    d = sort(D);
    n = numel(d);
    accuracy = cumsum(ones(1, n)) / n;
    if i== 2 || i==4
        plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
    else
        plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
    end
    aps(i) = VOCap_rotation_sym(d, accuracy);
    lengs{i} = sprintf('%s (AUC:%.2f)', leng{i}, aps(i)*100);
    hold on;
end
hold off;
h = legend(lengs(index_plot), 'Location', 'southeast');
set(h, 'FontSize', font_size);
h = xlabel('Threshold in Meter');
set(h, 'FontSize', font_size);
h = ylabel('Accuracy');
set(h, 'FontSize', font_size);
h = title('Rotaion of Symmetric Objects', 'Interpreter', 'none');
set(h, 'FontSize', font_size);
set(gca, 'FontSize', font_size)
set(gca,'XLim',[0 max_rot_dist])
set(gca,'YLim',[0 1])
pbaspect([1 0.75 1])


% rotation_non
axes(hf(4))
for i = index_plot
    D = rotation_non(index_non, i);
    D(D > max_rotation) = inf;
    d = sort(D);
    n = numel(d);
    accuracy = cumsum(ones(1, n)) / n;
    if i== 2 || i==4
        plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
    else
        plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
    end
    aps(i) = VOCap_rotation_non(d, accuracy);
    lengs{i} = sprintf('%s (AUC:%.2f)', leng{i}, aps(i)*100);
    hold on;
end
hold off;
h = legend(lengs(index_plot), 'Location', 'southeast');
set(h, 'FontSize', font_size);
h = xlabel('Threshold in Degree');
set(h, 'FontSize', font_size);
h = ylabel('Accuracy');
set(h, 'FontSize', font_size);
h = title('Rotation of Asymmetric Objects', 'Interpreter', 'none');
set(h, 'FontSize', font_size);
set(gca, 'FontSize', font_size)
set(gca,'XLim',[0 180])
set(gca,'YLim',[0 1])
pbaspect([1 0.75 1])
    

function ap = VOCap(rec, prec)

index = isfinite(rec);
rec = rec(index);
prec = prec(index)';
mrec=[0 ; rec ; 0.1];
mpre=[0 ; prec ; prec(end)];
for i = 2:numel(mpre)
    mpre(i) = max(mpre(i), mpre(i-1));
end
i = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
ap = sum((mrec(i) - mrec(i-1)) .* mpre(i)) * 10;


function ap = VOCap_rotation_non(rec, prec)

index = isfinite(rec);
rec = rec(index);
prec = prec(index)';
mrec=[0 ; rec ; 180];
mpre=[0 ; prec ; prec(end)];
for i = 2:numel(mpre)
    mpre(i) = max(mpre(i), mpre(i-1));
end
i = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
ap = sum((mrec(i) - mrec(i-1)) .* mpre(i)) / 180;


function ap = VOCap_rotation_sym(rec, prec)

index = isfinite(rec);
rec = rec(index);
prec = prec(index)';
mrec=[0 ; rec ; 0.04];
mpre=[0 ; prec ; prec(end)];
for i = 2:numel(mpre)
    mpre(i) = max(mpre(i), mpre(i-1));
end
i = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
ap = sum((mrec(i) - mrec(i-1)) .* mpre(i)) * 25;


function [ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w)
% tight_subplot creates "subplot" axes with adjustable gaps and margins
%
% [ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w)
%
%   in:  Nh      number of axes in hight (vertical direction)
%        Nw      number of axes in width (horizontaldirection)
%        gap     gaps between the axes in normalized units (0...1)
%                   or [gap_h gap_w] for different gaps in height and width 
%        marg_h  margins in height in normalized units (0...1)
%                   or [lower upper] for different lower and upper margins 
%        marg_w  margins in width in normalized units (0...1)
%                   or [left right] for different left and right margins 
%
%  out:  ha     array of handles of the axes objects
%                   starting from upper left corner, going row-wise as in
%                   subplot
%        pos    positions of the axes objects
%
%  Example: ha = tight_subplot(3,2,[.01 .03],[.1 .01],[.01 .01])
%           for ii = 1:6; axes(ha(ii)); plot(randn(10,ii)); end
%           set(ha(1:4),'XTickLabel',''); set(ha,'YTickLabel','')
% Pekka Kumpulainen 21.5.2012   @tut.fi
% Tampere University of Technology / Automation Science and Engineering
if nargin<3; gap = .02; end
if nargin<4 || isempty(marg_h); marg_h = .05; end
if nargin<5; marg_w = .05; end
if numel(gap)==1; 
    gap = [gap gap];
end
if numel(marg_w)==1; 
    marg_w = [marg_w marg_w];
end
if numel(marg_h)==1; 
    marg_h = [marg_h marg_h];
end
axh = (1-sum(marg_h)-(Nh-1)*gap(1))/Nh; 
axw = (1-sum(marg_w)-(Nw-1)*gap(2))/Nw;
py = 1-marg_h(2)-axh; 
% ha = zeros(Nh*Nw,1);
ii = 0;
for ih = 1:Nh
    px = marg_w(1);
    
    for ix = 1:Nw
        ii = ii+1;
        ha(ii) = axes('Units','normalized', ...
            'Position',[px py axw axh], ...
            'XTickLabel','', ...
            'YTickLabel','');
        px = px+axw+gap(2);
    end
    py = py-axh-gap(1);
end
if nargout > 1
    pos = get(ha,'Position');
end
ha = ha(:);




% % for each class
% for k = 1:numel(classes)
%     index = find(cls_ids == k);
%     if isempty(index)
%         index = 1:size(distances_sys, 1);
%     end
% 
%     % distance symmetry
%     subplot(2, 2, 1);
%     for i = index_plot
%         D = distances_sys(index, i);
%         D(D > max_distance) = inf;
%         d = sort(D);
%         n = numel(d);
%         c = numel(d(d < 0.02));
%         accuracy = cumsum(ones(1, n)) / n;
%         if i== 2 || i==4
%             plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
%         else
%             plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
%         end
%         aps(i) = VOCap(d, accuracy);
%         lengs{i} = sprintf('%s(AUC:%.2f)(<2cm:%.2f)', leng{i}, aps(i)*100, (c/n)*100);
%         hold on;
%     end
%     hold off;
%     h = legend(lengs(index_plot), 'Location', 'southeast');
%     set(h, 'FontSize', font_size);
%     h = xlabel('Average distance threshold in meter (symmetry)');
%     set(h, 'FontSize', font_size);
%     h = ylabel('accuracy');
%     set(h, 'FontSize', font_size);
%     h = title(classes{k}, 'Interpreter', 'none');
%     set(h, 'FontSize', font_size);
%     xt = get(gca, 'XTick');
%     set(gca, 'FontSize', font_size)
% 
%     % distance non-symmetry
%     subplot(2, 2, 2);
%     for i = index_plot
%         D = distances_non(index, i);
%         D(D > max_distance) = inf;
%         d = sort(D);
%         n = numel(d);
%         c = numel(d(d < 0.02));
%         accuracy = cumsum(ones(1, n)) / n;
%         if i== 2 || i==4
%             plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
%         else
%             plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
%         end
%         aps(i) = VOCap(d, accuracy);
%         lengs{i} = sprintf('%s(AUC:%.2f)(<2cm:%.2f)', leng{i}, aps(i)*100, (c/n)*100);
%         hold on;
%     end
%     hold off;
%     h = legend(lengs(index_plot), 'Location', 'southeast');
%     set(h, 'FontSize', font_size);
%     h = xlabel('Average distance threshold in meter (non-symmetry)');
%     set(h, 'FontSize', font_size);
%     h = ylabel('accuracy');
%     set(h, 'FontSize', font_size);
%     h = title(classes{k}, 'Interpreter', 'none');
%     set(h, 'FontSize', font_size);
%     xt = get(gca, 'XTick');
%     set(gca, 'FontSize', font_size)
%     
%     % rotation
%     subplot(2, 2, 3);
%     for i = index_plot
%         D = rotations(index, i);
%         d = sort(D);
%         n = numel(d);
%         accuracy = cumsum(ones(1, n)) / n;
%         if i== 2 || i==4
%             plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
%         else
%             plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
%         end
%         hold on;
%     end
%     hold off;
%     h = legend(leng(index_plot), 'Location', 'southeast');
%     set(h, 'FontSize', font_size);
%     h = xlabel('Rotation angle threshold');
%     set(h, 'FontSize', font_size);
%     h = ylabel('accuracy');
%     set(h, 'FontSize', font_size);
%     h = title(classes{k}, 'Interpreter', 'none');
%     set(h, 'FontSize', font_size);
%     xt = get(gca, 'XTick');
%     set(gca, 'FontSize', font_size)
% 
%     % translation
%     subplot(2, 2, 4);
%     for i = index_plot
%         D = translations(index, i);
%         D(D > max_distance) = inf;
%         d = sort(D);
%         n = numel(d);
%         accuracy = cumsum(ones(1, n)) / n;
%         if i== 2 || i==4
%             plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
%         else
%             plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
%         end
%         hold on;
%     end
%     hold off;
%     h = legend(leng(index_plot), 'Location', 'southeast');
%     set(h, 'FontSize', font_size);
%     h = xlabel('Translation threshold in meter');
%     set(h, 'FontSize', font_size);
%     h = ylabel('accuracy');
%     set(h, 'FontSize', font_size);
%     h = title(classes{k}, 'Interpreter', 'none');
%     set(h, 'FontSize', font_size);
%     xt = get(gca, 'XTick');
%     set(gca, 'FontSize', font_size)
%     
%     filename = sprintf('plots/%s.png', classes{k});
%     hgexport(hf, filename, hgexport('factorystyle'), 'Format', 'png');
% end


% % for each class
% for k = 1:numel(classes)
%     index = find(cls_ids == k);
    
%     if isempty(index) == 0
        

%     else

%         % distances
%         subplot(1, 3, 1);
%         for i = index_plot
%             D = distances(index, i);
%             D(D > max_distance) = inf;
%             d = sort(D);
%             n = numel(d);
%             accuracy = cumsum(ones(1, n)) / n;
%             if i== 2 || i==4
%                 plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
%             else
%                 plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
%             end
%             aps(i) = VOCap(d, accuracy);
%             lengs{i} = sprintf('%s (AUC:%.2f)', leng{i}, aps(i)*100);
%             hold on;
%         end
%         hold off;
%         h = legend(lengs(index_plot), 'Location', 'southeast');
%         set(h, 'FontSize', font_size);
%         h = xlabel('Distance threshold in meter');
%         set(h, 'FontSize', font_size);
%         h = ylabel('accuracy');
%         set(h, 'FontSize', font_size);
%         h = title(classes{k}, 'Interpreter', 'none');
%         set(h, 'FontSize', font_size);
%         set(gca, 'FontSize', font_size)
%         set(gca,'XLim',[0 0.1])
%         set(gca,'YLim',[0 1])
%         pbaspect([1 0.75 1])
% 
%         % translation
%         subplot(1, 3, 2);
%         for i = index_plot
%             D = translations(index, i);
%             D(D > max_distance) = inf;
%             d = sort(D);
%             n = numel(d);
%             accuracy = cumsum(ones(1, n)) / n;
%             if i== 2 || i==4
%                 plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
%             else
%                 plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
%             end
%             aps(i) = VOCap(d, accuracy);
%             lengs{i} = sprintf('%s (AUC:%.2f)', leng{i}, aps(i)*100);
%             hold on;
%         end
%         hold off;
%         h = legend(lengs(index_plot), 'Location', 'southeast');
%         set(h, 'FontSize', font_size);
%         h = xlabel('Translation threshold in meter');
%         set(h, 'FontSize', font_size);
%         h = ylabel('accuracy');
%         set(h, 'FontSize', font_size);
%         h = title(classes{k}, 'Interpreter', 'none');
%         set(h, 'FontSize', font_size);
%         set(gca, 'FontSize', font_size)
%         set(gca,'XLim',[0 0.1])
%         set(gca,'YLim',[0 1])
%         pbaspect([1 0.75 1])
% 
%         if ismember(k, opt.sym_list)
%             % rotation_sym
%             subplot(1, 3, 3);
%             for i = index_plot
%                 D = rotation_sym(index, i);
%                 D(D > max_distance) = inf;
%                 d = sort(D);
%                 n = numel(d);
%                 accuracy = cumsum(ones(1, n)) / n;
%                 if i== 2 || i==4
%                     plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
%                 else
%                     plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
%                 end
%                 aps(i) = VOCap(d, accuracy);
%                 lengs{i} = sprintf('%s (AUC:%.2f)', leng{i}, aps(i)*100);
%                 hold on;
%             end
%             hold off;
%             h = legend(lengs(index_plot), 'Location', 'southeast');
%             set(h, 'FontSize', font_size);
%             h = xlabel('Translation threshold in meter');
%             set(h, 'FontSize', font_size);
%             h = ylabel('accuracy');
%             set(h, 'FontSize', font_size);
%             h = title(classes{k}, 'Interpreter', 'none');
%             set(h, 'FontSize', font_size);
%             set(gca, 'FontSize', font_size)
%             set(gca,'XLim',[0 0.1])
%             set(gca,'YLim',[0 1])
%             pbaspect([1 0.75 1])
%         else 
%             % rotation_non
%             subplot(1, 3, 3);
%             for i = index_plot
%                 D = rotation_non(index, i);
%                 D(D > max_rotation) = inf;
%                 d = sort(D);
%                 n = numel(d);
%                 accuracy = cumsum(ones(1, n)) / n;
%                 if i== 2 || i==4
%                     plot(d, accuracy, color{i}, 'LineStyle', '--', 'LineWidth', 2);
%                 else
%                     plot(d, accuracy, color{i}, 'LineStyle', '-', 'LineWidth', 2);
%                 end
%                 aps(i) = VOCap_rotation(d, accuracy);
%                 lengs{i} = sprintf('%s (AUC:%.2f)', leng{i}, aps(i)*100);
%                 hold on;
%             end
%             hold off;
%             h = legend(lengs(index_plot), 'Location', 'southeast');
%             set(h, 'FontSize', font_size);
%             h = xlabel('Rotation angle threshold');
%             set(h, 'FontSize', font_size);
%             h = ylabel('accuracy');
%             set(h, 'FontSize', font_size);
%             h = title(classes{k}, 'Interpreter', 'none');
%             set(h, 'FontSize', font_size);
%             set(gca, 'FontSize', font_size)
%             set(gca,'XLim',[0 180])
%             set(gca,'YLim',[0 1])
%             pbaspect([1 0.75 1])
%         end

% cls_ids = object.results_cls_id;
