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
    negative-curvature (type 4) head, evenly spread through the list."""
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
    """Heads -> 1, tails -> 2.  For selected bolalipid blocks ONE end head -> 4
    (alternating which end by block parity), exactly as in the plane model."""
    base_type = HeadOrTail(atom_index)
    if base_type != 1 or negative_curved_lipid_mask is None:
        return base_type
    lipid_index = atom_index // 6
    local_index = atom_index % 6
    if lipid_index < len(negative_curved_lipid_mask) and negative_curved_lipid_mask[lipid_index]:
        type4_head_index = 0 if (lipid_index % 2 == 0) else 5
        if local_index == type4_head_index:
            return 4
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
#  Bolalipid along a surface normal  (unchanged)
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
#  Vesicle (sphere with hole) + lens (two caps)
# ============================================================

def _pv_to_trimesh(mesh):
    mesh = mesh.extract_surface().triangulate()
    mesh.compute_normals(cell_normals=True, auto_orient_normals=True, inplace=True)
    faces = mesh.faces.reshape((mesh.n_faces_strict, 4))[:, 1:]
    tm = trimesh.Trimesh(mesh.points, faces, process=False)
    tm.fix_normals()
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


def VesicleCoordinates(R_ves, alpha, theta_lens, sigma, N_positions,
                       junction_gap, sphere_res=160):
    """
    Build membrane bolalipid coordinates on:
      - vesicle sphere (radius R_ves) with the polar cap above the junction removed
      - top lens cap    (bulges OUT, +z)
      - bottom lens cap  (bulges IN,  -z)
    sharing the junction circle at z = R_ves cos(alpha), radius a = R_ves sin(alpha).

    Lens caps both have opening angle theta_lens, so R_lens = a / sin(theta_lens).
    Counts split by surface area; total placed = N_positions exactly.
    """
    z_j = R_ves * np.cos(alpha)
    a = R_ves * np.sin(alpha)
    if not (0.0 < theta_lens < np.pi):
        raise ValueError("theta_lens must be in (0, pi).")
    R_lens = a / np.sin(theta_lens)

    # cap sphere centres (symmetric biconvex lens about z = z_j)
    z_ct = z_j - R_lens * np.cos(theta_lens)   # top cap centre  -> apex at z_j + h
    z_cb = z_j + R_lens * np.cos(theta_lens)   # bottom cap centre-> apex at z_j - h
    c_top = np.array([0.0, 0.0, z_ct])
    c_bot = np.array([0.0, 0.0, z_cb])

    # --- meshes ---
    sph = pyvista.Sphere(radius=R_ves, center=(0, 0, 0),
                         theta_resolution=sphere_res, phi_resolution=sphere_res)
    vesicle = sph.clip(normal="z", origin=(0, 0, z_j), invert=True)    # keep z <= z_j

    st = pyvista.Sphere(radius=R_lens, center=(0, 0, z_ct),
                        theta_resolution=sphere_res, phi_resolution=sphere_res)
    top_cap = st.clip(normal="z", origin=(0, 0, z_j), invert=False)    # keep z >= z_j

    sb = pyvista.Sphere(radius=R_lens, center=(0, 0, z_cb),
                        theta_resolution=sphere_res, phi_resolution=sphere_res)
    bot_cap = sb.clip(normal="z", origin=(0, 0, z_j), invert=True)     # keep z <= z_j

    tv, tt, tb = _pv_to_trimesh(vesicle), _pv_to_trimesh(top_cap), _pv_to_trimesh(bot_cap)

    # --- area-based lipid counts (exact total) ---
    A_ves = 2.0 * np.pi * R_ves ** 2 * (1.0 + np.cos(alpha))           # sphere minus polar cap
    A_cap = 2.0 * np.pi * R_lens ** 2 * (1.0 - np.cos(theta_lens))
    A_tot = A_ves + 2.0 * A_cap
    N_ves = int(round(N_positions * A_ves / A_tot))
    N_top = int(round(N_positions * A_cap / A_tot))
    N_bot = int(N_positions) - N_ves - N_top                          # remainder -> exact total

    # --- thin junction exclusion band so seam lipids don't collide ---
    keep_ves = lambda p: p[:, 2] <= z_j - junction_gap
    keep_top = lambda p: p[:, 2] >= z_j + junction_gap
    keep_bot = lambda p: p[:, 2] <= z_j - junction_gap

    pv, fv = sample_surface_with_keep(tv, N_ves, 1,     keep_ves)
    pt, ft = sample_surface_with_keep(tt, N_top, 10001, keep_top)
    pb, fb = sample_surface_with_keep(tb, N_bot, 20001, keep_bot)

    coords = np.vstack([
        place_bolalipids(pv, fv, tv.face_normals, sigma),
        place_bolalipids(pt, ft, tt.face_normals, sigma),
        place_bolalipids(pb, fb, tb.face_normals, sigma),
    ])

    geom = dict(z_j=z_j, a=a, R_lens=R_lens, theta_lens=theta_lens,
                c_top=c_top, c_bot=c_bot,
                apex_top=z_ct + R_lens, apex_bot=z_cb - R_lens,
                N_ves=N_ves, N_top=N_top, N_bot=N_bot)
    return coords, geom


# ============================================================
#  Droplet particles inside the lens
# ============================================================

def LensLattice(c_top, c_bot, R_lens, spacing, inset):
    """Simple cubic lattice inside the biconvex lens = intersection of the two cap balls,
    shrunk by `inset` so droplet beads stay clear of the cap membranes."""
    Rin = R_lens - inset
    if Rin <= 0:
        return np.zeros((0, 3))
    Rin2 = Rin * Rin
    zt = c_top[2] + R_lens
    zb = c_bot[2] - R_lens
    xs = np.arange(-R_lens, R_lens + 1e-9, spacing)
    ys = np.arange(-R_lens, R_lens + 1e-9, spacing)
    zs = np.arange(zb, zt + 1e-9, spacing)
    coords = []
    for x in xs:
        for y in ys:
            for z in zs:
                p = np.array([x, y, z])
                if np.sum((p - c_top) ** 2) <= Rin2 and np.sum((p - c_bot) ** 2) <= Rin2:
                    coords.append(p)
    return np.array(coords) if coords else np.zeros((0, 3))


def remove_overlaps(droplet_coords, lipid_coords, cutoff):
    if droplet_coords.shape[0] == 0 or lipid_coords.shape[0] == 0:
        return droplet_coords
    cutoff2 = cutoff * cutoff
    kept = [p for p in droplet_coords if np.all(np.sum((lipid_coords - p) ** 2, axis=1) > cutoff2)]
    return np.array(kept) if kept else np.zeros((0, 3))


# ============================================================
#  MAIN: write LAMMPS data file
# ============================================================

def main():
    cfg = configparser.ConfigParser()
    cfg.read('cfg.ini')

    fraction_bilayer = float(cfg['_']['Bilayer_fraction'])
    fraction_soft_bolalipids = float(cfg['_']['soft_bolalipid_fraction'])
    negative_curved_fraction = float(cfg['_']['negative_curve_fraction'])

    N_positions = int(cfg['_'].get('N_membrane_lipids', '6000'))

    # ---- vesicle + lens geometry ----
    R_ves     = float(cfg['_'].get('R_vesicle', '30'))
    alpha     = float(cfg['_'].get('alpha_junction', '0.6'))   # polar angle of junction (rad)
    theta_lens = float(cfg['_'].get('theta_lens', '1.0471975512'))  # cap opening angle (rad)

    sigma = float(cfg['_'].get('sigma', '1.0'))
    junction_gap = float(cfg['_'].get('junction_gap', '1.0'))

    lattice_spacing = float(cfg['_'].get('lattice_spacing', '1.0'))
    type3_clearance = float(cfg['_'].get('type3_clearance', str(1.5 * sigma)))
    chain_half_length = 2.5 * sigma
    box_buffer = float(cfg['_'].get('box_buffer', '8.0'))

    # ---- membrane coordinates (constant N_positions) ----
    coordinates_lipids, geom = VesicleCoordinates(
        R_ves, alpha, theta_lens, sigma, N_positions, junction_gap
    )
    # len(coordinates_lipids) == 6 * N_positions

    # ---- bola / bilayer split (use ALL points; exact total) ----
    fraction_bilayer_heads = fraction_bilayer / (2 - fraction_bilayer)
    N_bolalipids_positions = int(np.floor((1 - fraction_bilayer_heads) * N_positions))
    N_bilayer_lipids_positions = int(N_positions) - N_bolalipids_positions

    Natoms_bola  = N_bolalipids_positions * 6
    Nbonds_bola  = N_bolalipids_positions * 5
    Nangles_bola = N_bolalipids_positions * 4

    Natoms_bila  = N_bilayer_lipids_positions * 6
    Nbonds_bila  = N_bilayer_lipids_positions * 4
    Nangles_bila = N_bilayer_lipids_positions * 2

    negative_curved_lipid_mask = make_negative_curved_lipid_mask(
        N_bolalipids_positions, negative_curved_fraction
    )

    # ---- droplet (type 3) inside the lens ----
    inset = chain_half_length + type3_clearance
    raw_drop = LensLattice(geom['c_top'], geom['c_bot'], geom['R_lens'],
                           lattice_spacing, inset)
    droplet_coords = remove_overlaps(raw_drop, coordinates_lipids, type3_clearance)
    N_drop = droplet_coords.shape[0]

    Natoms_membrane = Natoms_bola + Natoms_bila
    Natoms = Natoms_membrane + N_drop
    Nbonds = Nbonds_bola + Nbonds_bila
    Nangles = Nangles_bola + Nangles_bila

    # ---- periodic box big enough for the whole vesicle + protruding cap ----
    L = R_ves + box_buffer
    xbox = ybox = L
    zhi = max(R_ves, geom['apex_top']) + box_buffer
    zlo = -(R_ves + box_buffer)

    print(f"[geom] z_j={geom['z_j']:.3f}  a={geom['a']:.3f}  R_lens={geom['R_lens']:.3f}")
    print(f"[geom] lens apex z in [{geom['apex_bot']:.3f}, {geom['apex_top']:.3f}]")
    print(f"[counts] N_ves={geom['N_ves']} N_top={geom['N_top']} N_bot={geom['N_bot']} "
          f"droplet={N_drop}")

    outfile = 'plane.data'
    with open(outfile, 'w') as f:
        f.write('LAMMPS data file\n\n')
        f.write(f'{Natoms} atoms\n')
        f.write(f'{Nbonds} bonds\n')
        f.write(f'{Nangles} angles\n\n')
        f.write('4 atom types\n')
        f.write('1 bond types\n')
        if fraction_bilayer == 1 and fraction_soft_bolalipids == 0:
            f.write('1 angle types\n')
        else:
            f.write('3 angle types\n')
        f.write('\n')
        f.write(f'{-xbox} {xbox} xlo xhi\n')
        f.write(f'{-ybox} {ybox} ylo yhi\n')
        f.write(f'{zlo} {zhi} zlo zhi\n\n')
        f.write('Masses\n\n')
        f.write('1 1\n2 1\n3 1\n4 1\n\n')
        f.write('Atoms # angle\n\n')

        # --- bola membrane atoms ---
        for i in range(Natoms_bola):
            atom_id = i + 1
            mol_id = i // 6 + 1
            atype = MembraneAtomType(i, negative_curved_lipid_mask)
            x, y, z = coordinates_lipids[i]
            f.write(f"{atom_id} {mol_id} {atype} {x} {y} {z}\n")

        # --- bilayer membrane atoms ---
        for i in range(Natoms_bila):
            atom_id = Natoms_bola + i + 1
            mol_id = i // 3 + Natoms_bola // 6 + 1
            global_atom_index = Natoms_bola + i
            atype = HeadOrTailBilayer(i)
            x, y, z = coordinates_lipids[global_atom_index]
            f.write(f"{atom_id} {mol_id} {atype} {x} {y} {z}\n")

        # --- droplet (type 3) atoms inside the lens ---
        last_bola_mol = Natoms_bola // 6
        last_bila_mol = Natoms_bila // 3
        first_drop_mol = last_bola_mol + last_bila_mol + 1
        for i in range(N_drop):
            atom_id = Natoms_membrane + i + 1
            mol_id = first_drop_mol + i
            x, y, z = droplet_coords[i]
            f.write(f"{atom_id} {mol_id} 3 {x} {y} {z}\n")

        # --- bonds (unchanged) ---
        f.write('\nBonds\n\n')
        for i in range(Nbonds_bola):
            f.write(f"{i+1} 1 {i+1 + i//5} {i+2 + i//5}\n")
        for i in range(Nbonds_bila):
            f.write(
                f"{i+Nbonds_bola+1} 1 "
                f"{i+Nbonds_bola+Nbonds_bola//5+1+i//2} "
                f"{i+Nbonds_bola+Nbonds_bola//5+2+i//2}\n"
            )

        # --- angles (unchanged) ---
        f.write('\nAngles\n\n')
        for i in range(Nangles_bola):
            f.write(f"{i+1} {HeadOrTailAngle(i, fraction_soft_bolalipids)} "
                    f"{i+1+(i//4)*2} {i+2+(i//4)*2} {i+3+(i//4)*2}\n")
        for i in range(Nangles_bila):
            base = i + Nangles_bola + (Nangles_bola // 4) * 2
            f.write(f"{i+Nangles_bola+1} 1 "
                    f"{base+1 + (i//1)*2} {base+2 + (i//1)*2} {base+3 + (i//1)*2}\n")


if __name__ == "__main__":
    main()