import argparse
import torch
from collections import OrderedDict
import traceback

def merge(path1, path2, alpha1, sr, f0, info, name, version):
    try:
        def extract(ckpt):
            a = ckpt["model"]
            opt = OrderedDict()
            opt["weight"] = {}
            for key in a.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = a[key]
            return opt

        ckpt1 = torch.load(path1, map_location="cpu")
        ckpt2 = torch.load(path2, map_location="cpu")
        cfg = ckpt1["config"]
        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key][:min_shape0].float())
                    + (1 - alpha1) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()
        opt["config"] = cfg
        opt["sr"] = sr
        opt["f0"] = 1 if f0 else 0
        opt["version"] = version
        opt["info"] = info
        torch.save(opt, f"weights/{name}.pth")
        return "Success."
    except:
        return traceback.format_exc()

def main():
    parser = argparse.ArgumentParser(description="Merge two voice models")
    parser.add_argument("--model_a", required=True, help="Path to model A")
    parser.add_argument("--model_b", required=True, help="Path to model B")
    parser.add_argument("--alpha_a", type=float, default=0.5, help="Weight for model A")
    parser.add_argument("--sr", choices=["32k", "40k", "48k"], default="40k", help="Target sample rate")
    parser.add_argument("--f0", type=lambda x: x.lower() == 'true', default=True, help="Whether the model has pitch guidance")
    parser.add_argument("--info", default="", help="Model information to be placed")
    parser.add_argument("--name", required=True, help="Name for saving the merged model")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2", help="Model version")
    
    args = parser.parse_args()
    
    result = merge(args.model_a, args.model_b, args.alpha_a, args.sr, args.f0, args.info, args.name, args.version)
    print(result)

if __name__ == "__main__":
    main()