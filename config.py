
class Config:

    DATA = "MVTec"  # 'MVTec' or 'VisA'
    
    ROOTS = {
        "VisA": 'root\\visa',
        "MVTec": 'root\\mvtec',
    }

    FEATURE_ROOT = "./features"

    RESULTS_ROOT = "./results"
    
    JSON_STARTS = {
        "VisA": 'json/VisA/dinov2_vits14/batch-0-shot/results.json',
        "MVTec": 'json/MVTec/dinov2_vits14/batch-0-shot/results.json'
    }
