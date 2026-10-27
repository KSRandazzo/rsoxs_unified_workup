import argparse

import httpx
import PyHyperScattering as phs
from tiled.client import from_uri

from rsoxs_unified_workup.config import load_scan_set
from rsoxs_unified_workup.loading import load_scans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_set_id")
    args = parser.parse_args()

    print("Connecting to the BNL Tiled catalog...")
    catalog = from_uri(
        "https://tiled.nsls2.bnl.gov",
        timeout=httpx.Timeout(None),
    )

    loader = phs.load.SST1RSoXSDB(
        corr_mode="none",
        catalog=catalog["rsoxs"]["raw"],
    )

    configuration = load_scan_set(args.scan_set_id)
    scans = load_scans(loader, configuration)

    print("\nSuccessfully loaded:")
    for polarization, scan in scans.items():
        print(
            f"  scan {scan['scan_id']}: "
            f"configured polarization={polarization}°, "
            f"measured polarization="
            f"{scan['data'].attrs['polarization']}°"
        )


if __name__ == "__main__":
    main()
