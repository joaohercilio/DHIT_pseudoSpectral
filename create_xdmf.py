import h5py
import numpy as np
from pathlib import Path

filename = "fields.h5"
output = "fields.xdmf"

with h5py.File(filename, "r") as h5:
    timesteps = sorted(h5["u/3D"].keys(), key=int)
    shape = h5[f"u/3D/{timesteps[0]}"].shape
    _, nx, ny, nz = shape

    coords = [h5[f"u/mesh/x{i}"][:] for i in range(3)]
    origin = [float(c[0]) for c in coords]
    spacing = [float(c[1] - c[0]) for c in coords]

with open(output, "w") as out:
    out.write('<?xml version="1.0" ?>\n')
    out.write('<Xdmf Version="3.0"><Domain>\n')
    out.write('<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n')

    for t in timesteps:
        out.write(f'<Grid Name="{t}" GridType="Uniform">\n')
        out.write(f'<Time Value="{t}"/>\n')
        out.write(f'<Topology TopologyType="3DCoRectMesh" Dimensions="{nz} {ny} {nx}"/>\n')
        out.write('<Geometry GeometryType="ORIGIN_DXDYDZ">\n')
        out.write('<DataItem Dimensions="3" Format="XML">')
        out.write(f'{origin[2]} {origin[1]} {origin[0]}</DataItem>\n')
        out.write('<DataItem Dimensions="3" Format="XML">')
        out.write(f'{spacing[2]} {spacing[1]} {spacing[0]}</DataItem>\n')
        out.write('</Geometry>\n')

        for i, name in enumerate(("u", "v", "w")):
            out.write(f'<Attribute Name="{name}" AttributeType="Scalar" Center="Node">\n')
            out.write(f'<DataItem ItemType="HyperSlab" Dimensions="{nz} {ny} {nx}">\n')
            out.write('<DataItem Dimensions="3 4" Format="XML">\n')
            out.write(f'{i} 0 0 0\n1 1 1 1\n1 {nx} {ny} {nz}\n')
            out.write('</DataItem>\n')
            out.write(f'<DataItem Dimensions="3 {nx} {ny} {nz}" NumberType="Float" Precision="8" Format="HDF">')
            out.write(f'{filename}:/u/3D/{t}</DataItem>\n')
            out.write('</DataItem></Attribute>\n')

        out.write('</Grid>\n')

    out.write('</Grid></Domain></Xdmf>\n')
