function evaluate_keyframe

opt = globals();

% read class names
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
object_names = C{1};
fclose(fid);

% load model points
num_objects = numel(object_names);
models = cell(num_objects, 1);
for i = 1:num_objects
    filename = fullfile(opt.root, 'models', object_names{i}, 'points.xyz');
    disp(filename);
    models{i} = load(filename);
end

% load the keyframe indexes
fid = fopen('keyframe.txt', 'r');
C = textscan(fid, '%s');
keyframes = C{1};
fclose(fid);

% save results
distances = zeros(100000, 5);
rotation_non = zeros(100000, 5);
rotation_sym = zeros(100000, 5);
errors_translation = zeros(100000, 5);
results_seq_id = zeros(100000, 1);
results_frame_id = zeros(100000, 1);
results_object_id = zeros(100000, 1);
results_cls_id = zeros(100000, 1);

% compute accuracy
success_count = zeros(21, 5);
cls_count = zeros(21, 1);
methods = {'PoseCNN', 'PoseCNN+ICP', 'Per-Pixel DF', 'Iterative DF', 'Ours'};

% for each image
non_count = 0;
sym_count = 0;
count = 0;
for i = 1:numel(keyframes)
    % parse keyframe name
    name = keyframes{i};
    pos = strfind(name, '/');
    seq_id = str2double(name(1:pos-1));
    frame_id = str2double(name(pos+1:end));
    % load PoseCNN result
    filename = sprintf('results_PoseCNN_RSS2018/%06d.mat', i - 1);
    result_posecnn = load(filename);
    % load DenseFusion result
    filename = sprintf('Densefusion_wo_refine_result/%04d.mat', i - 1);
    result_densefusion = load(filename);
    filename = sprintf('Densefusion_iterative_result/%04d.mat', i - 1);
    result_densefusion_refine = load(filename);
    % load my result
    filename = sprintf('my_result/%04d.mat', i - 1);
    result_my = load(filename);
    % load gt poses
    filename = fullfile(opt.root, 'data', sprintf('%04d/%06d-meta.mat', seq_id, frame_id));
    disp(filename);
    gt = load(filename);

    % for each gt poses
    for j = 1:numel(gt.cls_indexes)
        count = count + 1;
        cls_index = gt.cls_indexes(j);
        RT_gt = gt.poses(:, :, j);
        R_gt = zeros(3, 4);
        R_gt(1:3, 1:3) = RT_gt(1:3, 1:3);
        
        cls_count(cls_index) = cls_count(cls_index) + 1;
        if ismember(cls_index, opt.sym_list)
            use_adi = true;
            sym_count = sym_count + 1;
        else
            use_adi = false;
            non_count = non_count + 1;
        end
        threshold = 0.1 * opt.diameters(cls_index);

        results_seq_id(count) = seq_id;
        results_frame_id(count) = frame_id;
        results_object_id(count) = j;
        results_cls_id(count) = cls_index;

        % network result
        roi_index = find(result_posecnn.rois(:, 2) == cls_index);
        if isempty(roi_index) == 0
            RT = zeros(3, 4);
            R = zeros(3, 4);
            % pose from PoseCNN
            RT(1:3, 1:3) = quat2rotm(result_posecnn.poses(roi_index, 1:4));
            RT(:, 4) = result_posecnn.poses(roi_index, 5:7);
            R(1:3, 1:3) = RT(1:3, 1:3);
            errors_translation(count, 1) = te(RT(:, 4), RT_gt(:, 4));
            if use_adi
                distances(count, 1) = adi(RT, RT_gt, models{cls_index}');
                rotation_sym(sym_count, 1) = adi(R, R_gt, models{cls_index}');
            else
                distances(count, 1) = add(RT, RT_gt, models{cls_index}');
                rotation_non(non_count, 1) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
            end
            if distances(count, 1) < threshold
                success_count(cls_index, 1) = success_count(cls_index, 1) + 1;
            end
            % pose from PoseCNN after ICP refinement
            RT(1:3, 1:3) = quat2rotm(result_posecnn.poses_icp(roi_index, 1:4));
            RT(:, 4) = result_posecnn.poses_icp(roi_index, 5:7);
            R(1:3, 1:3) = RT(1:3, 1:3);
            errors_translation(count, 2) = te(RT(:, 4), RT_gt(:, 4));
            if use_adi
                distances(count, 2) = adi(RT, RT_gt, models{cls_index}');
                rotation_sym(sym_count, 2) = adi(R, R_gt, models{cls_index}');
            else
                distances(count, 2) = add(RT, RT_gt, models{cls_index}');
                rotation_non(non_count, 2) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
            end
            if distances(count, 2) < threshold
                success_count(cls_index, 2) = success_count(cls_index, 2) + 1;
            end
            % pose from DenseFusion
            RT(1:3, 1:3) = quat2rotm(result_densefusion.poses(roi_index, 1:4));
            RT(:, 4) = result_densefusion.poses(roi_index, 5:7);
            R(1:3, 1:3) = RT(1:3, 1:3);
            errors_translation(count, 3) = te(RT(:, 4), RT_gt(:, 4));
            if use_adi
                distances(count, 3) = adi(RT, RT_gt, models{cls_index}');
                rotation_sym(sym_count, 3) = adi(R, R_gt, models{cls_index}');
            else
                distances(count, 3) = add(RT, RT_gt, models{cls_index}');
                rotation_non(non_count, 3) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
            end
            if distances(count, 3) < threshold
                success_count(cls_index, 3) = success_count(cls_index, 3) + 1;
            end
            % pose from DenseFusion after refinement
            RT(1:3, 1:3) = quat2rotm(result_densefusion_refine.poses(roi_index, 1:4));
            RT(:, 4) = result_densefusion_refine.poses(roi_index, 5:7);
            R(1:3, 1:3) = RT(1:3, 1:3);
            errors_translation(count, 4) = te(RT(:, 4), RT_gt(:, 4));
            if use_adi
                distances(count, 4) = adi(RT, RT_gt, models{cls_index}');
                rotation_sym(sym_count, 4) = adi(R, R_gt, models{cls_index}');
            else
                distances(count, 4) = add(RT, RT_gt, models{cls_index}');
                rotation_non(non_count, 4) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
            end
            if distances(count, 4) < threshold
                success_count(cls_index, 4) = success_count(cls_index, 4) + 1;
            end
            % pose from our network
            RT(1:3, 1:3) = quat2rotm(result_my.poses(roi_index, 1:4));
            RT(:, 4) = result_my.poses(roi_index, 5:7);
            R(1:3, 1:3) = RT(1:3, 1:3);
            errors_translation(count, 5) = te(RT(:, 4), RT_gt(:, 4));
            if use_adi
                distances(count, 5) = adi(RT, RT_gt, models{cls_index}');
                rotation_sym(sym_count, 5) = adi(R, R_gt, models{cls_index}');
            else
                distances(count, 5) = add(RT, RT_gt, models{cls_index}');
                rotation_non(non_count, 5) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
            end
            if distances(count, 5) < threshold
                success_count(cls_index, 5) = success_count(cls_index, 5) + 1;
            end
        else
            if use_adi
                rotation_sym(sym_count, 1:5) = inf;
            else
                rotation_non(non_count, 1:5) = inf;
            end
            errors_translation(count, 1:5) = inf;
            distances(count, 1:5) = inf;
        end
    end
end
distances = distances(1:count, :);
rotation_sym = rotation_sym(1:sym_count, :);
rotation_non = rotation_non(1:non_count, :);
errors_translation = errors_translation(1:count, :);
save('results_keyframe_plot.mat', 'distances', 'rotation_sym', 'rotation_non', 'errors_translation');

for i = 1:5
    disp(methods{i})
    disp(success_count(:, i) ./ cls_count)
    disp('average')
    disp(mean(success_count(:, i) ./ cls_count))
end
    

function pts_new = transform_pts_Rt(pts, RT)
%     """
%     Applies a rigid transformation to 3D points.
% 
%     :param pts: nx3 ndarray with 3D points.
%     :param R: 3x3 rotation matrix.
%     :param t: 3x1 translation vector.
%     :return: nx3 ndarray with transformed 3D points.
%     """
n = size(pts, 2);
pts_new = RT * [pts; ones(1, n)];

function error = add(RT_est, RT_gt, pts)
%     """
%     Average Distance of Model Points for objects with no indistinguishable views
%     - by Hinterstoisser et al. (ACCV 2012).
% 
%     :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param model: Object model given by a dictionary where item 'pts'
%     is nx3 ndarray with 3D model points.
%     :return: Error of pose_est w.r.t. pose_gt.
%     """
pts_est = transform_pts_Rt(pts, RT_est);
pts_gt = transform_pts_Rt(pts, RT_gt);
diff = pts_est - pts_gt;
error = mean(sqrt(sum(diff.^2, 1)));

function error = adi(RT_est, RT_gt, pts)
%     """
%     Average Distance of Model Points for objects with indistinguishable views
%     - by Hinterstoisser et al. (ACCV 2012).
% 
%     :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param model: Object model given by a dictionary where item 'pts'
%     is nx3 ndarray with 3D model points.
%     :return: Error of pose_est w.r.t. pose_gt.
%     """
pts_est = transform_pts_Rt(pts, RT_est);
pts_gt = transform_pts_Rt(pts, RT_gt);

% Calculate distances to the nearest neighbors from pts_gt to pts_est
MdlKDT = KDTreeSearcher(pts_est');
[~, D] = knnsearch(MdlKDT, pts_gt');
error = mean(D);

function error = re(R_est, R_gt)
%     """
%     Rotational Error.
% 
%     :param R_est: Rotational element of the estimated pose (3x1 vector).
%     :param R_gt: Rotational element of the ground truth pose (3x1 vector).
%     :return: Error of t_est w.r.t. t_gt.
%     """

error_cos = 0.5 * (trace(R_est * inv(R_gt)) - 1.0);
error_cos = min(1.0, max(-1.0, error_cos));
error = acos(error_cos);
error = 180.0 * error / pi;

function error = te(t_est, t_gt)
% """
% Translational Error.
% 
% :param t_est: Translation element of the estimated pose (3x1 vector).
% :param t_gt: Translation element of the ground truth pose (3x1 vector).
% :return: Error of t_est w.r.t. t_gt.
% """
error = norm(t_gt - t_est);