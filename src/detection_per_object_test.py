import os
import json
from src.ReRem_detection_test import ReRemAnomalyDetector

def run_per_object_adaptive_loop(
    model,
    args,
    Config,
    objects,
    overall_output_dir,
    object_anomalies,
    rotation_default,
    masking,
    feature_fuse,
    stop_condition,
):
    results_dir_base = f'{Config.RESULTS_ROOT}/{args.dataset}/{args.model_name}/n_jicheng/K={args.K}'
    features_dir = f"{Config.FEATURE_ROOT}/{args.dataset}/{args.model_name}/"
    os.makedirs(results_dir_base, exist_ok=True)

    final_results = {}

    for object_name in objects:
        current_cnt = 1
        json_path = Config.JSON_STARTS[args.dataset]

        with open(json_path, 'r') as f:
            sort_path_all = json.load(f)
        sample_nums = len(sort_path_all[object_name])
        detector = ReRemAnomalyDetector(
            model=model,
            object_name=object_name,
            data_root=Config.ROOTS[args.dataset],
            features_dir=features_dir,
            feature_fuse=feature_fuse[object_name],
            feature_list=args.feature_list,
            object_anomalies=object_anomalies,
            knn_metric=args.knn_metric,
            knn_neighbors=args.k_neighbors,
            faiss_on_cpu=args.faiss_on_cpu,
            masking=masking[object_name],
            mask_ref_images=args.mask_ref_images,
            rotation=rotation_default[object_name],
            top_p=args.gamma
        )
        while True:
            print(f"\n Iteration: {current_cnt}")
            results_dir = f'{results_dir_base}/cnt={current_cnt}'
            os.makedirs(results_dir, exist_ok=True)
            detector.results = {"all": []}
            first_flag = current_cnt == 1
            n_ref_samples = args.K if first_flag else args.K * current_cnt

            if first_flag:
                cur_sort_path = sort_path_all
                img_ref_samples = [p['path'] for p in cur_sort_path[object_name][:n_ref_samples]]
                detector.init_reference_memory(img_ref_samples)
            else:
                prev_json = f'{results_dir_base}/cnt={current_cnt - 1}/results.json'
                with open(prev_json, 'r') as f:
                    cur_sort_path = json.load(f)

                img_ref_samples = cur_sort_path[object_name]['img_ref_samples']
                img_test_samples = []
                for p in cur_sort_path[object_name]['all']:
                    if p['path'] not in img_ref_samples:
                        img_ref_samples.append(p['path'])
                    if len(img_ref_samples) >= n_ref_samples:
                        break

                detector.results["img_ref_samples"] = img_ref_samples
                detector.add_reference_samples_greedy(img_ref_samples[-args.K:])


            _, results = detector.run_inference(
                overall_output_dir=overall_output_dir,
                save_patch_dists=args.eval_clf,
                save_tiffs=args.eval_segm,
            )

            stop_flag = stop_condition(args.tau, sample_nums, current_cnt, args.K)
            json_output_path = f'{results_dir}/results.json'
            if os.path.exists(json_output_path):
                with open(json_output_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}
            existing_data[object_name] = results
            with open(json_output_path, 'w') as f:
                json.dump(existing_data, f, indent=4)

            if stop_flag:
                _, results = detector.run_inference(
                    overall_output_dir=overall_output_dir,
                    out_samples=img_test_samples,
                    save_patch_dists=args.eval_clf,
                    save_tiffs=args.eval_segm,
                )
                final_results[object_name] = results
                break
            else:
                current_cnt += 1

    return final_results
