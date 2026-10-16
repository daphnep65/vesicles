import pyvista
import trimesh
import numpy as np
import configparser


# ============================================================
#  Type / angle helpers  (unchanged from the plane+dome model)
# ============================================================

def HeadOrTail(n):
    j = (n + 1) % 6 + 1
    if j == 1 or j == 2:
        return 1
    else:
        return 2


def HeadOrTailBilayer(n):
    j = (n + 1) % 3
    if j == 1:
        return 1
    else:
        return 2


def clamp_fraction(value, name):
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def make_negative_curved_lipid_mask(n_lipids, negative_curved_fraction):
    """Deterministically pick round(fraction * n_lipids) bolalipid blocks for the
    negative-curvature (type 3) head, evenly spread through the list."""
    n_lipids = int(n_lipids)
    fraction = clamp_fraction(negative_curved_fraction, "negative_curved_fraction")
    mask = np.zeros(n_lipids, dtype=bool)
    n_negative = int(np.round(fraction * n_lipids))
    if n_negative <= 0:
        return mask
    if n_negative >= n_lipids:
        mask[:] = True
        return mask
    indices = np.floor((np.arange(n_negative) + 0.5) * n_lipids / n_negative).astype(int)
    mask[indices] = True
    return mask


def MembraneAtomType(atom_index, negative_curved_lipid_mask=None):
    """Heads -> 1, tails -> 2.  For selected bolalipid blocks ONE end head -> 3
    (alternating which end by block parity), exactly as in the plane model."""
    base_type = HeadOrTail(atom_index)
    if base_type != 1 or negative_curved_lipid_mask is None:
        return base_type
    lipid_index = atom_index // 3
    local_index = atom_index % 6
    if lipid_index < len(negative_curved_lipid_mask) and negative_curved_lipid_mask[lipid_index]:
        type3_head_index = 0 if (lipid_index % 2 == 0) else 5
        if local_index == type3_head_index:
            return 3
    return 1


def HeadOrTailAngle(n, fraction):
    j = (n + 1) % 4 + 1
    if j == 1 or j == 2:
        return 1
    else:
        k = (n // 4) % 10
        coloured = int(10 * fraction)
        if k < coloured:
            return 2   # minority - km_soft
        else:
            return 3   # km_stiff


# ============================================================
#  Bolalipid along a surface normal  (for defintion of coords)
# ============================================================

def BolalipidCoordinates(COM, normal, sigma):
    nrm = np.linalg.norm(normal)
    normal = np.array([0.0, 0.0, 1.0]) if nrm == 0 else normal / nrm
    bola = np.zeros((6, 3))
    bola[0, :] = COM + sigma * (0.5 + 2) * normal
    bola[1, :] = COM + sigma * (0.5 + 1) * normal
    bola[2, :] = COM + sigma * (0.5) * normal
    bola[3, :] = COM - sigma * (0.5) * normal
    bola[4, :] = COM - sigma * (0.5 + 1) * normal
    bola[5, :] = COM - sigma * (0.5 + 2) * normal
    return bola


# ============================================================
#  Vesicle definition using bolalipids
# ============================================================

def _pv_to_trimesh(mesh):
    mesh = mesh.extract_surface().triangulate()
    mesh.compute_normals(cell_normals=True, auto_orient_normals=True, inplace=True)
    faces = mesh.faces.reshape((mesh.n_faces_strict, 4))[:, 1:]
    tm = trimesh.Trimesh(mesh.points, faces, process=False)
    #tm.fix_normals()
    return tm


def sample_surface_with_keep(tmesh, n_target, seed_base, keep_fn=None, max_attempts=800):
    """Sample exactly n_target points on tmesh, optionally rejecting via keep_fn(points)->bool mask."""
    if n_target <= 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)
    pts_list, face_list, remaining, attempt = [], [], n_target, 0
    while remaining > 0:
        batch = int(np.ceil(remaining * 2.0 + 50))
        pts, fidx = trimesh.sample.sample_surface(tmesh, count=batch, seed=seed_base + attempt)
        if keep_fn is not None:
            m = keep_fn(pts)
            pts, fidx = pts[m], fidx[m]
        if pts.shape[0] > 0:
            take = min(remaining, pts.shape[0])
            pts_list.append(pts[:take])
            face_list.append(fidx[:take])
            remaining -= take
        attempt += 1
        if attempt > max_attempts and remaining > 0:
            raise RuntimeError("Sampling failed; relax junction_gap or lower N_membrane_lipids.")
    return np.vstack(pts_list), np.hstack(face_list)


def place_bolalipids(points, faces, face_normals, sigma):
    out = np.zeros((len(faces) * 6, 3))
    for i, fi in enumerate(faces):
        out[i * 6:(i + 1) * 6, :] = BolalipidCoordinates(points[i], face_normals[fi], sigma)
    return out


def VesicleCoordinates(R_ves, sigma, N_positions, center=(0, 0, 0), sphere_res=160):
    """Build membrane bolalipid coordinates on vesicle sphere (radius R_ves)"""
    
    vesicle = pyvista.Sphere(radius=R_ves, center=center,
                         theta_resolution=sphere_res, phi_resolution=sphere_res)
    
    tv = _pv_to_trimesh(vesicle)

    pv, fv = sample_surface_with_keep(tv, int(N_positions), 1)

    coords = np.vstack([
        place_bolalipids(pv, fv, tv.face_normals, sigma)])

    return coords


# ============================================================
#  MAIN: write LAMMPS data file
# ============================================================

def main():
    cfg = configparser.ConfigParser()
    cfg.read('cfg.ini')

    negative_curved_fraction = float(cfg['_'].get('fraction_negative_curve'))
    N_positions_A = int(cfg['_'].get('N_membrane_lipids_A')) # amount of bolalipids, for amount of lipids *2
    N_positions_B = int(cfg['_'].get('N_membrane_lipids_B')) # amount of bolalipids, for amount of lipids *2


    # ---- vesicle + box geometry ----
    R_ves_A     = float(cfg['_'].get('R_vesicle_A', '30'))
    R_ves_B     = float(cfg['_'].get('R_vesicle_B', '30'))
    sigma = float(cfg['_'].get('sigma', '1.0'))
    box_buffer = float(cfg['_'].get('box_buffer', '8.0'))
    inter_vesicle = float(cfg['_'].get('R_intervesicle')) # Initial distance between the bilayer centers of the vesicles
    
    middlepoint_x = R_ves_A - R_ves_B
    center_A = (middlepoint_x -inter_vesicle/2 - R_ves_A, 0, 0)
    center_B = (middlepoint_x + inter_vesicle/2 + R_ves_B, 0, 0)

    # ---- membrane coordinates (constant N_positions) ----
    coordinates_lipids_A = VesicleCoordinates(R_ves_A, sigma, N_positions_A, center_A)
    coordinates_lipids_B = VesicleCoordinates(R_ves_B, sigma, N_positions_B, center_B)
    # len(coordinates_lipids) == 6 * N_positions
    
    # For now: both vesicles have the same NIC fraction and rhd
    N_reduced_lipids_positions_A = int(np.floor(negative_curved_fraction * N_positions_A))
    N_normal_lipids_positions_A = N_positions_A - N_reduced_lipids_positions_A
    N_reduced_lipids_positions_B = int(np.floor(negative_curved_fraction * N_positions_B))
    N_normal_lipids_positions_B = N_positions_B - N_reduced_lipids_positions_B

    Natoms_normal_A  = N_normal_lipids_positions_A * 6
    Nbonds_normal_A  = N_normal_lipids_positions_A * 4
    Nangles_normal_A = N_normal_lipids_positions_A * 2

    Natoms_normal_B  = N_normal_lipids_positions_B * 6
    Nbonds_normal_B  = N_normal_lipids_positions_B * 4
    Nangles_normal_B = N_normal_lipids_positions_B * 2
    
    Natoms_reduced_A  = N_reduced_lipids_positions_A * 6
    Nbonds_reduced_A  = N_reduced_lipids_positions_A * 4
    Nangles_reduced_A = N_reduced_lipids_positions_A * 2

    Natoms_reduced_B  = N_reduced_lipids_positions_B * 6
    Nbonds_reduced_B  = N_reduced_lipids_positions_B * 4
    Nangles_reduced_B = N_reduced_lipids_positions_B * 2

    negative_curved_lipid_mask_A = make_negative_curved_lipid_mask(
        (2*N_positions_A), negative_curved_fraction
    ) # Double amount of positions, as bolalipids have 2 heads
    
    negative_curved_lipid_mask_B = make_negative_curved_lipid_mask(
        (2*N_positions_B), negative_curved_fraction)

    Natoms_A = Natoms_normal_A + Natoms_reduced_A
    Nbonds_A = Nbonds_normal_A + Nbonds_reduced_A
    Nangles_A = Nangles_normal_A + Nangles_reduced_A

    Natoms_B = Natoms_normal_B + Natoms_reduced_B
    Nbonds_B = Nbonds_normal_B + Nbonds_reduced_B
    Nangles_B = Nangles_normal_B + Nangles_reduced_B

    # ---- periodic box big enough for the whole system ----
    xbox = R_ves_A + R_ves_B + inter_vesicle + box_buffer
    ybox = zbox = max(R_ves_A, R_ves_B) + box_buffer
    
    print(f"[geom] x_box={xbox:.3f}    y_box={ybox:.3f}    z_box={zbox:.3f}")
    print(f"[Vesicle A] N_lipids={(2*N_positions_A)}    N_normal_lipids={(2*N_normal_lipids_positions_A)}   N_NIC_lipids={(2*N_reduced_lipids_positions_A)}")
    print(f"[Vesicle B] N_lipids={(2*N_positions_B)}    N_normal_lipids={(2*N_normal_lipids_positions_B)}   N_NIC_lipids={(2*N_reduced_lipids_positions_B)}")

    outfile = 'vesicle.data'
    with open(outfile, 'w') as f:
        f.write('LAMMPS data file\n\n')
        f.write(f'{Natoms_A+Natoms_B} atoms\n')
        f.write(f'{Nbonds_A+Nbonds_B} bonds\n')
        f.write(f'{Nangles_A+Nangles_B} angles\n\n')
        f.write('3 atom types\n')
        f.write('1 bond types\n')
        f.write('1 angle types\n')
        f.write('\n')
        f.write(f'{-xbox} {xbox} xlo xhi\n')
        f.write(f'{-ybox} {ybox} ylo yhi\n')
        f.write(f'{-zbox} {zbox} zlo zhi\n\n')
        f.write('Masses\n\n')
        f.write('1 1\n2 1\n3 1\n\n')
        f.write('Atoms # angle\n\n')

        # --- bilayer membrane atoms ---
        for i in range(Natoms_A):
            atom_id = i + 1
            mol_id = i // 3 + 1
            global_atom_index = i
            atype = MembraneAtomType(i, negative_curved_lipid_mask_A)
            x, y, z = coordinates_lipids_A[global_atom_index]
            f.write(f"{atom_id} {mol_id} {atype} {x} {y} {z}\n")
            
        for i in range(Natoms_B):
            atom_id = Natoms_A + i + 1
            mol_id = Natoms_A // 3 + i // 3 + 1
            global_atom_index = i
            atype = MembraneAtomType(i, negative_curved_lipid_mask_B)
            x, y, z = coordinates_lipids_B[global_atom_index]
            f.write(f"{atom_id} {mol_id} {atype} {x} {y} {z}\n")

        # --- bonds ---
        f.write('\nBonds\n\n')
        for i in range(Nbonds_A + Nbonds_B):
            f.write(f"{i+1} 1 {i+1+i//2} {i+2+i//2}\n")

        # --- angles ---
        f.write('\nAngles\n\n')
        for i in range(Nangles_A + Nangles_B):
            f.write(f"{i+1} 1 {i+1 + (i//1)*2} {i+2 + (i//1)*2} {i+3 + (i//1)*2}\n")


if __name__ == "__main__":
    main()