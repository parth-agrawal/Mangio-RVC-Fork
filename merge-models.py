import argparse
from infer-web import merge

def main():
    parser = argparse.ArgumentParser(description="Merge two voice models")
    parser.add_argument("--model_a", required=True, help="Path to model A")
    parser.add_argument("--model_b", required=True, help="Path to model B")
    parser.add_argument("--alpha_a", type=float, default=0.5, help="Weight for model A")
    parser.add_argument("--sr", choices=["40k", "48k"], default="40k", help="Target sample rate")
    parser.add_argument("--f0", type=bool, default=True, help="Whether the model has pitch guidance")
    parser.add_argument("--info", default="", help="Model information to be placed")
    parser.add_argument("--name", required=True, help="Name for saving the merged model")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1", help="Model version")
    
    args = parser.parse_args()
    
    result = merge(args.model_a, args.model_b, args.alpha_a, args.sr, args.f0, args.info, args.name, args.version)
    print(result)

if __name__ == "__main__":
    main()