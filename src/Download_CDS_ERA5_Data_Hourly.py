#!/usr/bin/env python3
"""
Download hourly ERA5 pressure-level and surface-level data from the
Copernicus Climate Data Store (CDS). Looks for .cdsapirc, prompts user
for dates and bounding box.
"""

import os
from datetime import date, timedelta
from pathlib import Path
import cdsapi


# CDS API configuration helpers
def _parse_cds_config(path: Path):
    """Parse .cdsapirc; returns (url, key)."""
    url = None
    key = None

    try:
        text = path.read_text()
    except Exception as exc:
        raise SystemExit(f"Error reading {path}: {exc}")

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("url:"):
            url = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("key:"):
            key = stripped.split(":", 1)[1].strip()

    return url, key


def get_cds_client():
    """Return a configured cdsapi.Client from .cdsapirc or interactive prompt."""

    home_config = Path.home() / ".cdsapirc"
    script_dir_config = Path(__file__).parent / ".cdsapirc"

    if home_config.exists():
        print(f"Found CDS API config at: {home_config}")
        # cdsapi reads ~/.cdsapirc automatically
        return cdsapi.Client()

    if script_dir_config.exists():
        print(f"Found CDS API config at: {script_dir_config}")
        url, key = _parse_cds_config(script_dir_config)
        return cdsapi.Client(url=url, key=key)

    print("No .cdsapirc configuration file was found.")
    print("You have two options to configure the CDS API:\n")
    print("  1) Manually enter your CDS API URL and key")
    print("  2) Provide a path to an existing .cdsapirc file\n")

    choice = input("Enter 1 or 2 (or anything else to abort): ").strip()

    if choice == "1":
        # Option 1: manually enter URL + key
        default_url = "https://cds.climate.copernicus.eu/api/v2"
        url = input(f"Enter CDS API URL [{default_url}]: ").strip()
        if not url:
            url = default_url

        key = input("Enter CDS API key (format USER_ID:API_KEY): ").strip()
        if not key:
            raise SystemExit("No API key entered. Exiting.")

        print("Using manually entered CDS API credentials.")
        return cdsapi.Client(url=url, key=key)

    elif choice == "2":
        # Option 2: provide path to .cdsapirc
        path_str = input("Enter the full path to your .cdsapirc file: ").strip()
        cfg_path = Path(path_str).expanduser()

        if not cfg_path.exists():
            raise SystemExit(f"Config file not found at: {cfg_path}. Exiting.")

        url, key = _parse_cds_config(cfg_path)
        if not url or not key:
            raise SystemExit("Could not read 'url' and 'key' from the provided .cdsapirc. Exiting.")

        print(f"Using CDS API config from: {cfg_path}")
        return cdsapi.Client(url=url, key=key)

    else:
        raise SystemExit("No valid option selected. Exiting.")


# ---------------------------------------------------------------------
# Helper functions for prompting user input
# ---------------------------------------------------------------------
def prompt_int(prompt_text, default):
    """Prompt for an integer with a default."""
    while True:
        raw = input(f"{prompt_text} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")


def prompt_float(prompt_text, default):
    """Prompt for a float with a default."""
    while True:
        raw = input(f"{prompt_text} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number (decimals allowed).")


# ---------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------
def download_era5_data(start_year, start_month, start_day,
                       end_year, end_month, end_day,
                       north_latitude, south_latitude, west_longitude, east_longitude,
                       https_proxy=None):
    """Download hourly ERA5 data for the given date range and bounding box."""

    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
        os.environ["https_proxy"] = https_proxy

    c = get_cds_client()

    root_dir = Path(__file__).parent.parent

    pressure_dir = root_dir / "data/external/ERA5/pressure"
    surface_dir = root_dir / "data/external/ERA5/surface"

    pressure_dir.mkdir(parents=True, exist_ok=True)
    surface_dir.mkdir(parents=True, exist_ok=True)

    times = [
        "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00", "21:00", "22:00", "23:00",
    ]

    idate = date(start_year, start_month, start_day)
    edate = date(end_year, end_month, end_day)

    while idate <= edate:
        stryear = f"{idate.year:04d}"
        strmonth = f"{idate.month:02d}"
        strday = f"{idate.day:02d}"
        yyyymmdd = f"{stryear}{strmonth}{strday}"

        for hour in times:
            hhmm = hour.replace(":", "")
            print(f"Requesting data for {yyyymmdd} {hour}")

            pressure_filename = pressure_dir / f"preslev_{yyyymmdd}_{hhmm}.grib"
            surface_filename = surface_dir / f"surface_{yyyymmdd}_{hhmm}.grib"

            c.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "format": "grib",
                    "year": stryear,
                    "month": strmonth,
                    "day": strday,
                    "time": [hour],
                    "variable": [
                        "divergence", "fraction_of_cloud_cover", "geopotential",
                        "ozone_mass_mixing_ratio", "potential_vorticity", "relative_humidity",
                        "specific_cloud_ice_water_content", "specific_cloud_liquid_water_content",
                        "specific_humidity", "specific_rain_water_content", "specific_snow_water_content",
                        "temperature", "u_component_of_wind", "v_component_of_wind",
                        "vertical_velocity", "vorticity",
                    ],
                    "pressure_level": [
                        "10", "20", "30", "50", "70", "100", "125", "150", "175", "200",
                        "225", "250", "300", "350", "400", "450", "500", "550", "600",
                        "650", "700", "750", "775", "800", "825", "850", "875", "900",
                        "925", "950", "975", "1000",
                    ],
                    "area": [
                        north_latitude, west_longitude, south_latitude, east_longitude
                    ],
                    "grid": [0.25, 0.25],
                },
                str(pressure_filename),
            )

            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "grib",
                    "year": stryear,
                    "month": strmonth,
                    "day": strday,
                    "time": [hour],
                    "variable": [
                        "10m_u_component_of_wind", "10m_v_component_of_wind",
                        "2m_dewpoint_temperature", "2m_temperature",
                        "land_sea_mask", "mean_sea_level_pressure",
                        "sea_ice_cover", "sea_surface_temperature",
                        "skin_temperature", "snow_density", "snow_depth",
                        "soil_temperature_level_1", "soil_temperature_level_2",
                        "soil_temperature_level_3", "soil_temperature_level_4",
                        "surface_pressure",
                        "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
                        "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4",
                    ],
                    "area": [
                        north_latitude, west_longitude, south_latitude, east_longitude
                    ],
                    "grid": [0.25, 0.25],
                },
                str(surface_filename),
            )

        idate += timedelta(days=1)

    print("Hourly ERA5 downloads complete.")
    print(f"Pressure files saved in: {pressure_dir}")
    print(f"Surface files saved in:  {surface_dir}")


# ---------------------------------------------------------------------
# Main entry point (interactive)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=== ERA5 Download Configuration ===")
    print("Press Enter to accept the default value shown in [brackets].\n")

    # SSLS dataset ranges from 2019-08-25 00:01:06.039000-04:00 to 2019-09-13 23:59:51.985000-04:00

    # ----- Default values (you can change these) -----
    default_start_year = 2019
    default_start_month = 8
    default_start_day = 25

    default_end_year = 2019
    default_end_month = 9
    default_end_day = 13
    
    # Savannah, GA bounding box
    default_north_lat = 32.172647
    default_south_lat = 31.92835
    default_west_lon = -81.213645
    default_east_lon = -80.850964
    # ------------------------------------------------

    # Ask the user for date range (with defaults)
    print("Start date:")
    start_year = prompt_int("  Start year", default_start_year)
    start_month = prompt_int("  Start month", default_start_month)
    start_day = prompt_int("  Start day", default_start_day)

    print("\nEnd date:")
    end_year = prompt_int("  End year", default_end_year)
    end_month = prompt_int("  End month", default_end_month)
    end_day = prompt_int("  End day", default_end_day)

    # Ask the user for spatial bounds (with defaults)
    print("\nGeographic bounding box (latitudes/longitudes):")
    print("Note: North > South; longitudes in degrees (West/East, can be negative).")

    north_latitude = prompt_float("  North latitude", default_north_lat)
    south_latitude = prompt_float("  South latitude", default_south_lat)
    west_longitude = prompt_float("  West longitude", default_west_lon)
    east_longitude = prompt_float("  East longitude", default_east_lon)

    # Optional: ask for HTTPS proxy (or leave as None)
    print("\nHTTPS proxy (optional).")
    print("Example: http://user:password@proxy-server:port")
    proxy_input = input("  Enter HTTPS proxy (or leave blank for none): ").strip()
    https_proxy = proxy_input if proxy_input else None

    # Echo back configuration
    print("\n=== Summary of your choices ===")
    print(f"  Start date: {start_year:04d}-{start_month:02d}-{start_day:02d}")
    print(f"  End date:   {end_year:04d}-{end_month:02d}-{end_day:02d}")
    print(f"  North lat:  {north_latitude}")
    print(f"  South lat:  {south_latitude}")
    print(f"  West lon:   {west_longitude}")
    print(f"  East lon:   {east_longitude}")
    print(f"  HTTPS proxy: {https_proxy if https_proxy else 'None'}")
    print("================================\n")

    # Call the download_era5_data function with user-provided values
    download_era5_data(
        start_year=start_year,
        start_month=start_month,
        start_day=start_day,
        end_year=end_year,
        end_month=end_month,
        end_day=end_day,
        north_latitude=north_latitude,
        south_latitude=south_latitude,
        west_longitude=west_longitude,
        east_longitude=east_longitude,
        https_proxy=https_proxy,
    )

