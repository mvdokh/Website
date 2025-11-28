#!/usr/bin/env python3
"""
STL to optimized JSON converter for web rendering.
Uses mesh decimation to reduce polygon count while preserving shape.

Usage: python process_stl.py Mouse_Brain.stl --target 5000
"""

import json
import argparse
import sys

def parse_ascii_stl(filename):
    """Parse ASCII STL file and return vertices and faces."""
    vertices = []
    faces = []
    vertex_map = {}  # To deduplicate vertices
    
    print(f"Loading {filename}...")
    
    with open(filename, 'r') as f:
        current_face = []
        for line in f:
            line = line.strip()
            if line.startswith('vertex'):
                parts = line.split()
                v = (float(parts[1]), float(parts[2]), float(parts[3]))
                
                # Deduplicate vertices
                if v not in vertex_map:
                    vertex_map[v] = len(vertices)
                    vertices.append(v)
                
                current_face.append(vertex_map[v])
                
            elif line.startswith('endfacet'):
                if len(current_face) == 3:
                    faces.append(current_face)
                current_face = []
    
    print(f"Loaded: {len(vertices)} unique vertices, {len(faces)} faces")
    return vertices, faces


def compute_face_normals(vertices, faces):
    """Compute normal for each face."""
    normals = []
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Edge vectors
        e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
        
        # Cross product
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        
        # Normalize
        length = (nx*nx + ny*ny + nz*nz) ** 0.5
        if length > 0:
            normals.append((nx/length, ny/length, nz/length))
        else:
            normals.append((0, 0, 1))
    
    return normals


def compute_quadric_error(vertices, faces):
    """
    Simple quadric error metric for each vertex.
    Used to prioritize which vertices to keep during decimation.
    """
    import math
    
    # For each vertex, compute sum of squared distances to adjacent face planes
    vertex_errors = [0.0] * len(vertices)
    vertex_face_count = [0] * len(vertices)
    
    for face in faces:
        for vi in face:
            vertex_face_count[vi] += 1
    
    return vertex_face_count


def simple_decimate(vertices, faces, target_faces):
    """
    Improved decimation that keeps connected regions together.
    Uses spatial binning to ensure uniform coverage.
    """
    if len(faces) <= target_faces:
        return vertices, faces
    
    print(f"Decimating from {len(faces)} to {target_faces} faces...")
    
    # Compute face centers and normals
    face_data = []
    for i, face in enumerate(faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Center
        cx = (v0[0] + v1[0] + v2[0]) / 3
        cy = (v0[1] + v1[1] + v2[1]) / 3
        cz = (v0[2] + v1[2] + v2[2]) / 3
        
        # Normal and area
        e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        area = (nx*nx + ny*ny + nz*nz) ** 0.5 / 2
        
        face_data.append({
            'idx': i,
            'center': (cx, cy, cz),
            'area': area,
            'normal': (nx, ny, nz)
        })
    
    # Spatial binning - divide space into grid cells
    all_x = [f['center'][0] for f in face_data]
    all_y = [f['center'][1] for f in face_data]
    all_z = [f['center'][2] for f in face_data]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    min_z, max_z = min(all_z), max(all_z)
    
    # Create bins
    num_bins = 20
    bin_size_x = (max_x - min_x) / num_bins + 0.001
    bin_size_y = (max_y - min_y) / num_bins + 0.001
    bin_size_z = (max_z - min_z) / num_bins + 0.001
    
    bins = {}
    for fd in face_data:
        bx = int((fd['center'][0] - min_x) / bin_size_x)
        by = int((fd['center'][1] - min_y) / bin_size_y)
        bz = int((fd['center'][2] - min_z) / bin_size_z)
        key = (bx, by, bz)
        if key not in bins:
            bins[key] = []
        bins[key].append(fd)
    
    # From each bin, select faces proportionally
    kept_indices = set()
    total_faces = len(faces)
    
    for key, bin_faces in bins.items():
        # Sort by area (largest first)
        bin_faces.sort(key=lambda x: x['area'], reverse=True)
        
        # How many to keep from this bin
        proportion = len(bin_faces) / total_faces
        keep_count = max(1, int(target_faces * proportion * 1.1))
        
        # Keep the largest faces from this bin
        for i, fd in enumerate(bin_faces):
            if i < keep_count:
                kept_indices.add(fd['idx'])
    
    # If we have too few, add more from largest faces globally
    if len(kept_indices) < target_faces:
        face_data.sort(key=lambda x: x['area'], reverse=True)
        for fd in face_data:
            if len(kept_indices) >= target_faces:
                break
            kept_indices.add(fd['idx'])
    
    # If we have too many, remove smallest
    if len(kept_indices) > target_faces:
        kept_list = list(kept_indices)
        kept_with_area = [(idx, face_data[idx]['area']) for idx in kept_list]
        kept_with_area.sort(key=lambda x: x[1], reverse=True)
        kept_indices = set(x[0] for x in kept_with_area[:target_faces])
    
    # Build new mesh
    new_faces = [faces[i] for i in sorted(kept_indices)]
    
    used_vertices = set()
    for face in new_faces:
        used_vertices.update(face)
    
    old_to_new = {}
    new_vertices = []
    for old_idx in sorted(used_vertices):
        old_to_new[old_idx] = len(new_vertices)
        new_vertices.append(vertices[old_idx])
    
    remapped_faces = [[old_to_new[v] for v in face] for face in new_faces]
    
    print(f"Result: {len(new_vertices)} vertices, {len(remapped_faces)} faces")
    return new_vertices, remapped_faces


def try_trimesh_decimate(filename, target_faces):
    """
    Try to use trimesh for proper mesh decimation.
    Falls back to simple decimation if trimesh is not available.
    """
    try:
        import trimesh
        print("Using trimesh for high-quality decimation...")
        
        mesh = trimesh.load(filename)
        print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        if len(mesh.faces) > target_faces:
            # Use quadric decimation for best quality
            simplified = mesh.simplify_quadric_decimation(target_faces)
            print(f"Decimated to: {len(simplified.vertices)} vertices, {len(simplified.faces)} faces")
            
            vertices = simplified.vertices.tolist()
            faces = simplified.faces.tolist()
            normals = simplified.face_normals.tolist()
            
            return vertices, faces, normals
        else:
            return mesh.vertices.tolist(), mesh.faces.tolist(), mesh.face_normals.tolist()
            
    except ImportError:
        print("trimesh not installed. Using simple decimation.")
        print("For better results: pip install trimesh")
        return None


def main():
    parser = argparse.ArgumentParser(description='Convert STL to optimized JSON for web')
    parser.add_argument('input', help='Input STL file')
    parser.add_argument('--output', '-o', default='brain_mesh.json', help='Output JSON file')
    parser.add_argument('--target', '-t', type=int, default=8000, help='Target number of faces')
    args = parser.parse_args()
    
    # Try trimesh first (better quality)
    result = try_trimesh_decimate(args.input, args.target)
    
    if result:
        vertices, faces, normals = result
    else:
        # Fallback to simple decimation
        vertices, faces = parse_ascii_stl(args.input)
        vertices, faces = simple_decimate(vertices, faces, args.target)
        normals = compute_face_normals(vertices, faces)
    
    # Compute bounding box for centering
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
    
    # Round vertices to reduce file size
    vertices = [[round(v[0], 4), round(v[1], 4), round(v[2], 4)] for v in vertices]
    normals = [[round(n[0], 4), round(n[1], 4), round(n[2], 4)] for n in normals]
    center = [round(c, 4) for c in center]
    
    # Create output JSON
    output = {
        'vertices': vertices,
        'faces': faces,
        'normals': normals,
        'center': center,
        'bounds': {
            'min': [round(min_x, 4), round(min_y, 4), round(min_z, 4)],
            'max': [round(max_x, 4), round(max_y, 4), round(max_z, 4)]
        }
    }
    
    # Write JSON
    with open(args.output, 'w') as f:
        json.dump(output, f)
    
    # Report file size
    import os
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nSaved to {args.output} ({size_kb:.1f} KB)")
    print(f"Final mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Center: {center}")

if __name__ == '__main__':
    main()

