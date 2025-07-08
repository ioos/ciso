import xarray as xr

url = "http://test.opendap.org:8080/opendap/ugrid/NECOFS_GOM3_FORECAST.nc"
ds = xr.open_dataset(url)

variables = [
    "salinity",
    "fvcom_mesh",
    "x",
    "y",
    "lonc",
    "latc",
    "nv",
    "yc",
    "xc",
    "h",
    "zeta",
]

ds["nv"].attrs.update({"cf_role": "face_node_connectivity"})
ds = ds.isel(time=[-1])[variables]

ds.to_netcdf("NECOFS_GOM3_FORECAST.nc")
