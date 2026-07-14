import argparse
import os
from argparse import ArgumentParser
import cv2
import random
import json
from src.utils import get_dataset_info
from src.detection_per_object_test import run_per_object_adaptive_loop
from src.backbones import get_model
from src.post_eval import eval_finished_run
from config import Config
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='MVTec', choices=['VisA', 'MVTec'])
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--feature_list", type=list, default=[6, 9])
    parser.add_argument("--scales", type=list, default=[1, 5])
    parser.add_argument("--resolution", type=int, default=672)
    parser.add_argument("--knn_metric", type=str, default="L2_normalized")
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--faiss_on_cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask_ref_images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval_clf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_segm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_every_time", type=bool, default=False, help='Whether to eval the anomalyscore every time')
    parser.add_argument("--eval_mode", type=bool, default=False)        
    parser.add_argument("--device", default='cuda:0')
    return parser.parse_args()      

def stop_condition(p, sample_num, n, cnt):
    if n * cnt > sample_num * p:
        return True
    else:
        return False

def DPFE_sel(model, objects, object_anomalies, args, Config):
    feature_fuse = {}
    masking = {}
    for object_name in objects:  
        type_anomalies = object_anomalies[object_name]
        type_anomaly = random.choice(type_anomalies)
        #set the path
        data_dir = f"{Config.ROOTS[args.dataset]}/{object_name}/test/{type_anomaly}"
        img_path = random.choice(os.listdir(data_dir))
        img_test_path = f"{data_dir}/{img_path}"

        image_test = cv2.cvtColor(cv2.imread(img_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image_ref_tensor, grid_size1 = model.prepare_image(image_test)
        features = model.extract_features(image_ref_tensor)
        mask_ref = model.Kmeans_forge(features, grid_size1)
        all = len(mask_ref)
        low = sum(mask_ref)
        if low / all > 0.5:
            feature_fuse[object_name] = True
            masking[object_name] = False
        else:
            feature_fuse[object_name] = False
            masking[object_name] = True
        feature_fuse[object_name] = masking[object_name] ^ 1

    return feature_fuse, masking

if __name__ == "__main__":
    args = parse_args()
    random.seed(0)

    objects, object_anomalies = get_dataset_info(args.dataset)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    model = get_model(args.model_name, 'cuda', smaller_edge_size=args.resolution)
    rotation_default = {o: True for o in objects}

    overall_output_dir = os.path.join(
        Config.ROOTS[args.dataset],
        f"results_{args.dataset}",
        args.model_name,
        "n_jicheng",
        f"K={args.K}",
        "final_eval"
    )
    os.makedirs(overall_output_dir, exist_ok=True)
    
    if not args.eval_mode:

        feature_fuse, masking = DPFE_sel(model=model, objects=objects, object_anomalies=object_anomalies, args=args, Config=Config)
        final_results = run_per_object_adaptive_loop(
            model,
            args,
            Config,
            objects,    
            overall_output_dir,
            object_anomalies,
            rotation_default,
            masking,
            feature_fuse,
            stop_condition
        )


        eval_finished_run(
            dataset=args.dataset,
            dataset_base_dir=args.data_root,
            anomaly_maps_dir=os.path.join(overall_output_dir, "Anomaly_maps"),
            output_dir=overall_output_dir,
            seed=0,
            pro_integration_limit=0.3,
            eval_clf=args.eval_clf,
            eval_segm=args.eval_segm,
            # results=final_results
        )   


        with open(os.path.join(overall_output_dir, "final_object_results.json"), "w") as f:
            json.dump(final_results, f, indent=4)
    else:
        eval_finished_run(
            dataset=args.dataset,
            dataset_base_dir=args.data_root,
            anomaly_maps_dir=os.path.join(overall_output_dir, "Anomaly_maps"),
            output_dir=overall_output_dir,
            seed=0,
            pro_integration_limit=0.3,
            eval_clf=args.eval_clf,
            eval_segm=args.eval_segm
        )



