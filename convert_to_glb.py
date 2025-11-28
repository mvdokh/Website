#!/usr/bin/env python3
"""
Convert STL to GLB format for web rendering.
GLB is the binary version of glTF - the standard for web 3D.

Usage: python convert_to_glb.py Mouse_Brain.stl -o brain.glb
"""

import struct
import json
import base64
import argparse


def parse_ascii_stl(filename):
    """Parse ASCII STL file."""
    vertices = []
    indices = []
    vertex_map = {}
    
    print(f"Loading {filename}...")
    
    with open(filename, 'r') as f:
        current_face = []
        for line in f:
            line = line.strip()
            if line.startswith('vertex'):
                parts = line.split()
                v = (float(parts[1]), float(parts[2]), float(parts[3]))
                
                # Round to reduce duplicates
                v = (round(v[0], 4), round(v[1], 4), round(v[2], 4))
                
                if v not in vertex_map:
                    vertex_map[v] = len(vertices)
                    vertices.append(v)
                
                current_face.append(vertex_map[v])
                
            elif line.startswith('endfacet'):
                if len(current_face) == 3:
                    indices.extend(current_face)
                current_face = []
    
    print(f"Loaded: {len(vertices)} vertices, {len(indices)//3} faces")
    return vertices, indices


def compute_normals(vertices, indices):
    """Compute smooth vertex normals by averaging face normals."""
    normals = [[0, 0, 0] for _ in vertices]
    
    # Accumulate face normals to vertices
    for i in range(0, len(indices), 3):
        i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        
        # Edge vectors
        e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
        
        # Cross product (face normal)
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        
        # Add to each vertex
        for idx in [i0, i1, i2]:
            normals[idx][0] += nx
            normals[idx][1] += ny
            normals[idx][2] += nz
    
    # Normalize
    for n in normals:
        length = (n[0]**2 + n[1]**2 + n[2]**2) ** 0.5
        if length > 0:
            n[0] /= length
            n[1] /= length
            n[2] /= length
    
    return normals


def compute_bounds(vertices):
    """Compute bounding box and center."""
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    min_z = min(v[2] for v in vertices)
    max_z = max(v[2] for v in vertices)
    
    center = [
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
        (min_z + max_z) / 2
    ]
    
    return {
        'min': [min_x, min_y, min_z],
        'max': [max_x, max_y, max_z],
        'center': center
    }


def create_glb(vertices, indices, normals, output_file):
    """Create a GLB (binary glTF) file."""
    
    # Center the mesh
    bounds = compute_bounds(vertices)
    cx, cy, cz = bounds['center']
    
    # Create binary buffers
    # Vertices (centered)
    vertex_data = b''
    for v in vertices:
        vertex_data += struct.pack('<fff', v[0] - cx, v[1] - cy, v[2] - cz)
    
    # Normals
    normal_data = b''
    for n in normals:
        normal_data += struct.pack('<fff', n[0], n[1], n[2])
    
    # Indices (as unsigned int for large meshes)
    index_data = b''
    for idx in indices:
        index_data += struct.pack('<I', idx)
    
    # Combine all binary data
    # Pad to 4-byte alignment
    def pad4(data):
        remainder = len(data) % 4
        if remainder:
            data += b'\x00' * (4 - remainder)
        return data
    
    vertex_offset = 0
    vertex_length = len(vertex_data)
    
    normal_offset = vertex_length
    normal_length = len(normal_data)
    
    index_offset = vertex_length + normal_length
    index_length = len(index_data)
    
    buffer_data = vertex_data + normal_data + index_data
    buffer_data = pad4(buffer_data)
    
    # Compute bounds for accessor
    min_pos = [
        min(v[0] - cx for v in vertices),
        min(v[1] - cy for v in vertices),
        min(v[2] - cz for v in vertices)
    ]
    max_pos = [
        max(v[0] - cx for v in vertices),
        max(v[1] - cy for v in vertices),
        max(v[2] - cz for v in vertices)
    ]
    
    # Create glTF JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "STL to GLB converter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Brain"}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1
                },
                "indices": 2,
                "material": 0
            }],
            "name": "Brain"
        }],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.9, 0.85, 0.85, 1.0],  # Pinkish white
                "metallicFactor": 0.0,
                "roughnessFactor": 0.7
            },
            "name": "BrainMaterial"
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3",
                "min": min_pos,
                "max": max_pos
            },
            {
                "bufferView": 1,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(indices),
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": vertex_offset,
                "byteLength": vertex_length,
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": normal_offset,
                "byteLength": normal_length,
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": index_length,
                "target": 34963  # ELEMENT_ARRAY_BUFFER
            }
        ],
        "buffers": [{
            "byteLength": len(buffer_data)
        }]
    }
    
    # Convert JSON to bytes
    json_str = json.dumps(gltf, separators=(',', ':'))
    json_data = json_str.encode('utf-8')
    json_data = pad4(json_data)
    
    # Create GLB structure
    # Header: magic (4) + version (4) + length (4) = 12 bytes
    # JSON chunk: length (4) + type (4) + data
    # BIN chunk: length (4) + type (4) + data
    
    json_chunk_type = 0x4E4F534A  # "JSON"
    bin_chunk_type = 0x004E4942   # "BIN\0"
    
    total_length = 12 + 8 + len(json_data) + 8 + len(buffer_data)
    
    with open(output_file, 'wb') as f:
        # GLB header
        f.write(struct.pack('<I', 0x46546C67))  # magic: "glTF"
        f.write(struct.pack('<I', 2))           # version
        f.write(struct.pack('<I', total_length))
        
        # JSON chunk
        f.write(struct.pack('<I', len(json_data)))
        f.write(struct.pack('<I', json_chunk_type))
        f.write(json_data)
        
        # BIN chunk
        f.write(struct.pack('<I', len(buffer_data)))
        f.write(struct.pack('<I', bin_chunk_type))
        f.write(buffer_data)
    
    print(f"Created {output_file}")
    return bounds


def main():
    parser = argparse.ArgumentParser(description='Convert STL to GLB')
    parser.add_argument('input', help='Input STL file')
    parser.add_argument('--output', '-o', default='brain.glb', help='Output GLB file')
    args = parser.parse_args()
    
    # Load STL
    vertices, indices = parse_ascii_stl(args.input)
    
    # Compute smooth normals
    print("Computing normals...")
    normals = compute_normals(vertices, indices)
    
    # Create GLB
    print("Creating GLB...")
    bounds = create_glb(vertices, indices, normals, args.output)
    
    # Report
    import os
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nDone! {args.output} ({size_kb:.1f} KB)")
    print(f"Vertices: {len(vertices)}, Faces: {len(indices)//3}")
    print(f"Center: {bounds['center']}")


if __name__ == '__main__':
    main()

