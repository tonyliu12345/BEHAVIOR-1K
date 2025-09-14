"""
Helper script to download BEHAVIOR-1K dataset and assets.
Improved version that can import obj file and articulated file (glb, gltf).
"""

import os
import pathlib
import subprocess
from typing import Literal, Optional
import click
import shutil
import tempfile
import omnigibson.lazy as lazy
from OmniGibson.omnigibson.utils.asset_utils import encrypt_file, get_dataset_path
import omnigibson as og

from omnigibson.utils.asset_conversion_utils import (
    import_og_asset_from_urdf,
    generate_urdf_for_mesh,
)


@click.command()
@click.option(
    "--asset-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Absolute path to asset file to import. This can be a raw visual mesh (for single-bodied, static objects), e.g. .obj, .glb, etc., or a more complex (such as articulated) objects defined in .urdf format.",
)
@click.option("--category", required=True, type=click.STRING, help="Category name to assign to the imported asset")
@click.option(
    "--model",
    required=True,
    type=click.STRING,
    help="Model name to assign to the imported asset. This must be unique within the dataset.",
)
@click.option(
    "--collision-method",
    type=click.Choice(["coacd", "convex", "none"]),
    default="coacd",
    help="Method to generate the collision mesh. 'coacd' generates a set of convex decompositions, while 'convex' generates a single convex hull. 'none' will not generate any explicit mesh",
)
@click.option(
    "--hull-count",
    type=int,
    default=32,
    help="Maximum number of convex hulls to decompose individual visual meshes into. Only relevant if --collision-method=coacd",
)
@click.option("--up-axis", type=click.Choice(["z", "y"]), default="z", help="Up axis for the mesh.")
@click.option("--headless", is_flag=True, help="Run the script in headless mode.")
@click.option("--scale", type=int, default=1, help="User choice scale, will be overwritten if check_scale and rescale")
@click.option("--check_scale", is_flag=True, help="Check meshes scale based on heuristic")
@click.option("--rescale", is_flag=True, help="Rescale meshes based on heuristic if check_scale ")
@click.option("--overwrite", is_flag=True, help="Overwrite any pre-existing files")
@click.option(
    "--extra-metadata-json",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="If specified, path to additional metadata json file to include in the imported asset's metadata.",
)
def import_custom_object(
    asset_path: str,
    category: str,
    model: str,
    collision_method: Literal["coacd", "convex", "none"],
    hull_count: int,
    up_axis: Literal["z", "y"],
    headless: bool,
    scale: int,
    check_scale: bool,
    rescale: bool,
    overwrite: bool,
    extra_metadata_json: Optional[str] = None,
):
    """
    Imports a custom-defined object asset into an OmniGibson-compatible USD format and saves the imported asset
    files to the selected dataset directory.
    """

    assert len(model) == 6 and model.isalpha(), "Model name must be 6 characters long and contain only letters."
    collision_method = None if collision_method == "none" else collision_method

    # Resolve the asset path here
    asset_path = pathlib.Path(asset_path).absolute()

    # If we're not a URDF, import the mesh directly first
    temp_dir = tempfile.mkdtemp()

    try:
        if asset_path.suffix != ".urdf":
            # Try to generate URDF, may raise ValueError if too many submeshes
            urdf_path = generate_urdf_for_mesh(
                asset_path,
                temp_dir,
                category,
                model,
                collision_method,
                hull_count,
                up_axis,
                scale=scale,
                check_scale=check_scale,
                rescale=rescale,
                overwrite=True,
            )
            if urdf_path is not None:
                click.echo("URDF generation complete!")
                collision_method = None
            else:
                # Clean up temp directories before exiting
                click.echo("Error during URDF generation")
                raise click.Abort()
        else:
            urdf_path = asset_path
            collision_method = collision_method

        # Convert to USD
        _, usd_path, prim = import_og_asset_from_urdf(
            dataset_root=get_dataset_path("simaffordance"),
            category=category,
            model=model,
            urdf_path=str(urdf_path),
            collision_method=collision_method,
            hull_count=hull_count,
            overwrite=overwrite,
            use_usda=False,
            extra_metadata_json=extra_metadata_json,
        )

        # Archive the output USD files to USDZ
        usd_path2 = pathlib.Path(usd_path)
        model_dir = usd_path2.parent.parent
        moved_usd_path = model_dir / usd_path2.name
        usdz_path = moved_usd_path.with_suffix(".usdz")

        # Here we perform a trick where we first copy the USD file to a new location and then
        # update its asset paths to be relative to the new location. This allows the usdz archiver
        # to use relative paths instead of absolute ones.
        shutil.copy(usd_path, moved_usd_path)
        stage = lazy.pxr.Usd.Stage.Open(str(moved_usd_path))

        # Now update the asset paths
        materials_dir = (model_dir / "material").resolve()
        def _update_path(asset_path):
            # Compute the path that this path refers to
            asset_full_path = (usd_path2.parent / asset_path).resolve()

            # Check if that's a subpath of the materials dir. If not, return unchanged
            try:
                asset_full_path.relative_to(materials_dir)
            except ValueError:
                return asset_path
            
            # Return it relative to the new USD location
            return str(asset_full_path.relative_to(model_dir.resolve()))

        lazy.pxr.UsdUtils.ModifyAssetPaths(stage.GetRootLayer(), _update_path)
        stage.Save()
        del stage

        # Convert USD to USDZ
        print(f"Converting {moved_usd_path} to {usdz_path}")
        lazy.pxr.UsdUtils.CreateNewUsdzPackage(str(moved_usd_path), str(usdz_path))

        # Remove the MDL files from inside the USDZ. We want to load these from the
        # OmniGibson installation instead of from the USDZ directly, since they have
        # dependencies.
        path_7za = pathlib.Path(__file__).parents[4] / "asset_pipeline" / "7zzs"
        subprocess.run([str(path_7za), "d", str(usdz_path), "*.mdl"], check=True)

        os.remove(usd_path)
        os.remove(moved_usd_path)

    finally:
        # Clean up temp directories before exiting
        shutil.rmtree(temp_dir)

    # Visualize if not headless
    if not headless:
        click.echo("The asset has been successfully imported. You can view it and make changes and save if you'd like.")
        while True:
            og.sim.render()

    breakpoint()


if __name__ == "__main__":
    import_custom_object()
