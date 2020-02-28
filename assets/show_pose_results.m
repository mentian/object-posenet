function show_pose_results

opt = globals();

% read class names
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
object_names = C{1};
fclose(fid);
% load CAD models
disp('loading 3D models...');
num_objects = numel(object_names);
models = cell(num_objects, 1);
for i = 1:num_objects
    filename = sprintf('models/%s.mat', object_names{i});
    if exist(filename, 'file')
        object = load(filename);
        obj = object.obj;
    else
        file_obj = fullfile(opt.root, 'models', object_names{i}, 'textured.obj');
        obj = load_obj_file(file_obj);
        save(filename, 'obj');
    end
    models{i} = obj;
end
% load test frames
[seq_ids, frame_ids] = load_dataset_indexes('keyframe.txt');
num = numel(frame_ids);
frames = 1:num;

% for each image
figure('visible', 'off');
for i = frames
    % load PoseCNN results
    filename = sprintf('results_PoseCNN_RSS2018/%06d.mat', i-1);
    result_posecnn = load(filename);
    % load DenseFusion results
    filename = sprintf('Densefusion_wo_refine_result/%04d.mat', i-1);
    result_densefusion = load(filename);
    % load my results
    filename = sprintf('my_result/%04d.mat', i-1);
    result_my = load(filename);
    % number of objects
    cls_indexs = result_posecnn.rois(:, 2);
    n = size(cls_indexs, 1);
    % read image
    filename = fullfile(opt.root, 'data', sprintf('%04d/%06d-color.png', seq_ids(i), frame_ids(i)));
    I = imread(filename);
    disp(filename);

    % PoseCNN
    imshow(I);
    hold on;
    % sort objects according to distances
    poses = result_posecnn.poses;
    distances = poses(:, 7);
    [~, index] = sort(distances, 'descend');
    % draw
    for j = 1:n
        ind = index(j);
        RT = zeros(3, 4);
        RT(1:3, 1:3) = quat2rotm(poses(ind, 1:4));
        RT(:, 4) = poses(ind, 5:7);
        % projection
        objID = cls_indexs(ind);
        x3d = models{objID}.v';
        x2d = project(x3d, opt.intrinsic_matrix_color, RT);
        patch('vertices', x2d, 'faces', models{objID}.f3', ...
            'FaceColor', opt.class_colors(objID+1,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    hold off;
    F = getframe(gca);
    imwrite(F.cdata, sprintf('pose_visual/%04d-%06d-color-posecnn.png', seq_ids(i), frame_ids(i)));

    % DenseFusion
    imshow(I);
    hold on;
    % sort objects according to distances
    poses = result_densefusion.poses;
    distances = poses(:, 7);
    [~, index] = sort(distances, 'descend');
    for j = 1:n
        ind = index(j);
        RT = zeros(3, 4);
        RT(1:3, 1:3) = quat2rotm(poses(ind, 1:4));
        RT(:, 4) = poses(ind, 5:7);
        % projection
        objID = cls_indexs(ind);
        x3d = models{objID}.v';
        x2d = project(x3d, opt.intrinsic_matrix_color, RT);
        % draw
        patch('vertices', x2d, 'faces', models{objID}.f3', ...
            'FaceColor', opt.class_colors(objID+1,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    hold off;
    F = getframe(gca);
    imwrite(F.cdata, sprintf('pose_visual/%04d-%06d-color-densefusion.png', seq_ids(i), frame_ids(i)));

    % Ours
    imshow(I);
    hold on;
    % sort objects according to distances
    poses = result_my.poses;
    distances = poses(:, 7);
    [~, index] = sort(distances, 'descend');
    for j = 1:n
        ind = index(j);
        RT = zeros(3, 4);
        RT(1:3, 1:3) = quat2rotm(poses(ind, 1:4));
        RT(:, 4) = poses(ind, 5:7);
        % projection
        objID = cls_indexs(ind);
        x3d = models{objID}.v';
        x2d = project(x3d, opt.intrinsic_matrix_color, RT);
        % draw
        patch('vertices', x2d, 'faces', models{objID}.f3', ...
            'FaceColor', opt.class_colors(objID+1,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    hold off;
    F = getframe(gca);
    imwrite(F.cdata, sprintf('pose_visual/%04d-%06d-color-ours.png', seq_ids(i), frame_ids(i)));

end