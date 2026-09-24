#!/usr/bin/env python3
"""Resolve the official Ray Train DLC image URI directly — no mirroring needed.

SageMaker's `CreateTrainingJob` cannot pull from `public.ecr.aws`, which made it look like the
Ray Train DLC (published there as `public.ecr.aws/deep-learning-containers/ray:train-ml-cuda-*`)
had to be mirrored into a private ECR repo first. It doesn't: AWS also publishes the identical
image in a private, per-region ECR account under repository `ray` — the same distribution
mechanism every other DLC (`pytorch-training`, `huggingface-training`, ...) already uses, with
the same broad cross-account pull permissions. `CreateTrainingJob` pulls from there directly, no
CodeBuild mirror, no private repo of your own to manage.

The account map below is the exact one the SageMaker SDK ships for that repository (see
`ray-serve.json` in `sagemaker.core.image_uri_config`) — that file is scoped to the `serve-*`
tags, but it's the same repo/account per region, and the `train-ml-cuda-*` tags used here were
confirmed reachable (`ecr:BatchGetImage`) from an unrelated account with zero extra IAM.

Usage:
    from ray_dlc_image import get_ray_train_dlc_image_uri
    image_uri = get_ray_train_dlc_image_uri(sagemaker_session.boto_session.region_name)
"""

RAY_DLC_REPOSITORY = "ray"
RAY_DLC_TAG = "train-ml-cuda-v1.1"

RAY_DLC_ACCOUNTS = {
    "af-south-1": "626614931356",
    "ap-east-1": "871362719292",
    "ap-east-2": "975050140332",
    "ap-northeast-1": "763104351884",
    "ap-northeast-2": "763104351884",
    "ap-northeast-3": "364406365360",
    "ap-south-1": "763104351884",
    "ap-south-2": "772153158452",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ap-southeast-3": "907027046896",
    "ap-southeast-4": "457447274322",
    "ap-southeast-5": "550225433462",
    "ap-southeast-6": "633930458069",
    "ap-southeast-7": "590183813437",
    "ca-central-1": "763104351884",
    "ca-west-1": "204538143572",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
    "eu-central-1": "763104351884",
    "eu-central-2": "380420809688",
    "eu-north-1": "763104351884",
    "eu-south-1": "692866216735",
    "eu-south-2": "503227376785",
    "eu-west-1": "763104351884",
    "eu-west-2": "763104351884",
    "eu-west-3": "763104351884",
    "il-central-1": "780543022126",
    "me-central-1": "914824155844",
    "me-south-1": "217643126080",
    "mx-central-1": "637423239942",
    "sa-east-1": "763104351884",
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-gov-east-1": "446045086412",
    "us-gov-west-1": "442386744353",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}


def get_ray_train_dlc_image_uri(region: str, tag: str = RAY_DLC_TAG) -> str:
    """Return the direct, private-ECR URI for the official Ray Train DLC in `region`."""
    try:
        account = RAY_DLC_ACCOUNTS[region]
    except KeyError:
        raise ValueError(
            f"No known Ray Train DLC account for region {region!r}. "
            "See RAY_DLC_ACCOUNTS in this file."
        ) from None
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{RAY_DLC_REPOSITORY}:{tag}"


if __name__ == "__main__":
    import sys

    region = sys.argv[1] if len(sys.argv) > 1 else "us-west-2"
    print(get_ray_train_dlc_image_uri(region))
