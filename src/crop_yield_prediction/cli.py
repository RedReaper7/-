from __future__ import annotations

import argparse
import json

from .config import get_settings
from .pipeline import predict_live, train_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop yield prediction project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train all candidate models.")

    predict_parser = subparsers.add_parser("predict-live", help="Run a live prediction.")
    predict_parser.add_argument("--crop", required=True)
    predict_parser.add_argument("--latitude", type=float, required=True)
    predict_parser.add_argument("--longitude", type=float, required=True)
    predict_parser.add_argument("--state")
    predict_parser.add_argument("--district")
    predict_parser.add_argument("--season")
    predict_parser.add_argument("--area-ha", type=float, dest="area_ha")
    predict_parser.add_argument("--crop-year", type=int, dest="crop_year")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()

    if args.command == "train":
        result = train_project(settings)
        print(json.dumps(result, indent=2))
        return

    if args.command == "predict-live":
        payload = {
            "crop": args.crop,
            "latitude": args.latitude,
            "longitude": args.longitude,
            "state": args.state,
            "district": args.district,
            "season": args.season,
            "area_ha": args.area_ha,
            "crop_year": args.crop_year,
        }
        result = predict_live(settings, payload)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

