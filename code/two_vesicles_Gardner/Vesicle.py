import pyvista
import trimesh
import numpy as np
import configparser


# ============================================================
#  Type / angle helpers
# ============================================================

def HeadOrTail(n):
    j = (n + 1) % 6 + 1
    if j == 1 or j == 2:
        return 1
    else:
        return 2


def HeadOrTailBilayer(n):
    """Bilayer position = 6 beads along the normal = two 3-bead lipids.
    Outer-leaflet lipid: beads 0,1,2 -> head, tail, tail  (head at +2.5 sigma)
    Inner-leaflet lipid: beads 3,4,5 -> tail, tail, head  (head at -2.5 sigma)
    """
    local = n % 6
    if local == 0 or local == 5:
        return 1
    return 2


def clamp_fraction(value, name):
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def evenly_spaced_mask(n_items, n_pick, offset=0.5):
    """Deterministically pick n_pick of n_items, evenly spread.
    `offset` in [0,1) shifts the picks within each spacing interval."""
    n_items, n_pick = int(n_items), int(n_pick)
    mask = np.zeros(n_items, dtype=bool)
    if n_pick <= 0 or n_items <= 0:
        return mask
    if n_pick >= n_items:
        mask[:] = True
        return mask
    indices = np.floor((np.arange(n_pick) + offset) * n_items / n_pick).astype(int)
    mask[np.clip(indices, 0, n_items - 1)] = True
    return mask


def split_nic_counts(n_bola, n_bila_positions, negative_curved_fraction):
    """negative_curved_fraction = fraction of ALL lipid molecules (bolalipids + bilayer
    lipids) that carry a type-4 head.  Returns (n_bola_nic, n_bila_outer, n_bila_inner).
    The total is split between the two kinds in proportion to their molecule counts,
    and the per-leaflet totals (bola + bilayer) differ by at most one."""
    fraction = clamp_fraction(negative_curved_fraction, "negative_curved_fraction")
    n_bila_mol = 2 * int(n_bila_positions)
    n_mol = int(n_bola) + n_bila_mol
    if n_mol == 0:
        return 0, 0, 0
    n_total = int(np.round(fraction * n_mol))
    n_bola_nic = int(np.round(n_total * n_bola / n_mol))
    n_bila_nic = n_total - n_bola_nic
    # bolalipids put ceil(n/2) on the outer leaflet, so give the bilayer's extra one to inner
    n_bila_outer = n_bila_nic // 2
    n_bila_inner = n_bila_nic - n_bila_outer
    return n_bola_nic, n_bila_outer, n_bila_inner


def make_bola_type4_end(n_bola, n_nic):
    """For each bolalipid: -1 (no type-4 head), 0 (outer head is type 4) or 5 (inner head).
    The end alternates by RANK among the selected lipids, so the outer/inner split is
    exactly ceil(n/2) / floor(n/2) regardless of the selection spacing."""
    mask = evenly_spaced_mask(n_bola, n_nic)
    ends = np.full(int(n_bola), -1, dtype=int)
    selected = np.flatnonzero(mask)
    ends[selected[0::2]] = 0
    ends[selected[1::2]] = 5
    return ends


def MembraneAtomType(atom_index, bola_type4_end=None, nic_type=4):
    """Bolalipid atoms: heads -> 1, tails -> 2, chosen end head -> nic_type (4, or 3 with no droplet)."""
    base_type = HeadOrTail(atom_index)
    if base_type != 1 or bola_type4_end is None:
        return base_type
    lipid_index = atom_index // 6
    local_index = atom_index % 6
    if lipid_index < len(bola_type4_end) and bola_type4_end[lipid_index] == local_index:
        return nic_type
    return 1


def BilayerAtomType(bila_atom_index, outer_mask, inner_mask, nic_type=4):
    """Bilayer atoms: heads -> 1, tails -> 2; head -> nic_type if that lipid is selected.
    outer_mask / inner_mask are per bilayer position (one per leaflet)."""
    base_type = HeadOrTailBilayer(bila_atom_index)
    if base_type != 1:
        return base_type
    pos = bila_atom_index // 6
    local = bila_atom_index % 6
    if local == 0 and outer_mask[pos]:
        return nic_type
    if local == 5 and inner_mask[pos]:
        return nic_type
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
#  Bolalipid along a surface normal  
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
                       junction_gap, sphere_res=160, center=(0, 0, 0)):
    """
    Build membrane coordinates on:
      - vesicle sphere (radius R_ves) with the polar cap above the junction removed
      - top lens cap    (bulges OUT, +z)
      - bottom lens cap  (bulges IN,  -z)
    sharing the junction circle at z = R_ves cos(alpha), radius a = R_ves sin(alpha).
    """
    # ---- alpha_junction = 0  ->  plain closed vesicle, no lens / no droplet ----
    if alpha <= 0.0:
        sph = pyvista.Sphere(radius=R_ves, center=center,
                             theta_resolution=sphere_res, phi_resolution=sphere_res)
        tv = _pv_to_trimesh(sph)
        pv, fv = sample_surface_with_keep(tv, int(N_positions), 1)
        coords = place_bolalipids(pv, fv, tv.face_normals, sigma)
        pole = np.array([0.0, 0.0, R_ves])
        geom = dict(z_j=R_ves, a=0.0, R_lens=0.0, theta_lens=theta_lens,
                    c_top=pole, c_bot=pole, apex_top=R_ves, apex_bot=R_ves,
                    N_ves=int(N_positions), N_top=0, N_bot=0)
        return coords, geom

    z_j = R_ves * np.cos(alpha)
    a = R_ves * np.sin(alpha)
    if not (0.0 < theta_lens < np.pi):
        raise ValueError("theta_lens must be in (0, pi).")
    R_lens = a / np.sin(theta_lens)

    z_ct = z_j - R_lens * np.cos(theta_lens)
    z_cb = z_j + R_lens * np.cos(theta_lens)
    c_top = np.array([0.0, 0.0, z_ct])
    c_bot = np.array([0.0, 0.0, z_cb])

    sph = pyvista.Sphere(radius=R_ves, center=(0, 0, 0),
                         theta_resolution=sphere_res, phi_resolution=sphere_res)
    vesicle = sph.clip(normal="z", origin=(0, 0, z_j), invert=True)

    st = pyvista.Sphere(radius=R_lens, center=(0, 0, z_ct),
                        theta_resolution=sphere_res, phi_resolution=sphere_res)
    top_cap = st.clip(normal="z", origin=(0, 0, z_j), invert=False)

    sb = pyvista.Sphere(radius=R_lens, center=(0, 0, z_cb),
                        theta_resolution=sphere_res, phi_resolution=sphere_res)
    bot_cap = sb.clip(normal="z", origin=(0, 0, z_j), invert=True)

    tv, tt, tb = _pv_to_trimesh(vesicle), _pv_to_trimesh(top_cap), _pv_to_trimesh(bot_cap)

    A_ves = 2.0 * np.pi * R_ves ** 2 * (1.0 + np.cos(alpha))
    A_cap = 2.0 * np.pi * R_lens ** 2 * (1.0 - np.cos(theta_lens))
    A_tot = A_ves + 2.0 * A_cap
    N_ves = int(round(N_positions * A_ves / A_tot))
    N_top = int(round(N_positions * A_cap / A_tot))
    N_bot = int(N_positions) - N_ves - N_top

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

    fraction_bilayer = float(cfg['_'].get('Bilayer_fraction', '1.0'))
    fraction_soft_bolalipids = float(cfg['_'].get('soft_bolalipid_fraction', '0.0'))
    # fraction of ALL lipid molecules (bolalipids + bilayer lipids) with a type-4 head
    negative_curved_fraction = float(cfg['_'].get('negative_curve_fraction', '0.0'))

    N_positions = int(cfg['_'].get('N_membrane_lipids', '4000'))
    R_ves     = float(cfg['_'].get('R_vesicle', '20'))
    
    single_vesicle = cfg['_'].getboolean('single_vesicle', True)
    N_positions_B = int(cfg['_'].get('N_membrane_lipids_B', '4000'))
    R_ves_B     = float(cfg['_'].get('R_vesicle_B', '20'))
    R_intervesicle = float(cfg['_'].get('R_intervesicle', '6'))
    
    alpha     = float(cfg['_'].get('alpha_junction', '0.0'))
    theta_lens = float(cfg['_'].get('theta_lens', '0.1'))

    sigma = float(cfg['_'].get('sigma', '1.0'))
    junction_gap = float(cfg['_'].get('junction_gap', '1.0'))

    lattice_spacing = float(cfg['_'].get('lattice_spacing', '1.0'))
    type3_clearance = float(cfg['_'].get('type3_clearance', str(1.5 * sigma)))
    chain_half_length = 2.5 * sigma
    box_buffer = float(cfg['_'].get('box_buffer', '8.0'))
    
    
    if single_vesicle:
        coordinates_lipids, geom = VesicleCoordinates(
            R_ves, alpha, theta_lens, sigma, N_positions, junction_gap, center=(0, 0, 0)
        )
    else:
        middlepoint_x = R_ves_B - R_ves
        coordinates_lipids_A, geom_A = VesicleCoordinates(
            R_ves, alpha, theta_lens, sigma, N_positions, junction_gap, center=(middlepoint_x + R_intervesicle/2 + R_ves, 0, 0)
        )
        coordinates_lipids_B, geom_B = VesicleCoordinates(
            R_ves_B, alpha, theta_lens, sigma, N_positions_B, junction_gap,
            center=(middlepoint_x - R_intervesicle/2 - R_ves_B, 0, 0)
        )
        coordinates_lipids = np.vstack([coordinates_lipids_A, coordinates_lipids_B])
        geom = {
            'z_j': max(geom_A['z_j'], geom_B['z_j']),
            'a': max(geom_A['a'], geom_B['a']),
            'R_lens': max(geom_A['R_lens'], geom_B['R_lens']),
            'theta_lens': max(geom_A['theta_lens'], geom_B['theta_lens']),
            'c_top': np.array([0.0, 0.0, max(geom_A['c_top'][2], geom_B['c_top'][2])]),
            'c_bot': np.array([0.0, 0.0, min(geom_A['c_bot'][2], geom_B['c_bot'][2])]),
            'apex_top': max(geom_A['apex_top'], geom_B['apex_top']),
            'apex_bot': min(geom_A['apex_bot'], geom_B['apex_bot']),
            'N_ves': geom_A['N_ves'] + geom_B['N_ves'],
            'N_top': geom_A['N_top'] + geom_B['N_top'],
            'N_bot': geom_A['N_bot'] + geom_B['N_bot'],
        }

    fraction_bilayer_heads = fraction_bilayer / (2 - fraction_bilayer)
    N_bolalipids_positions = int(np.floor((1 - fraction_bilayer_heads) * N_positions))
    N_bilayer_lipids_positions = int(N_positions) - N_bolalipids_positions

    Natoms_bola  = N_bolalipids_positions * 6
    Nbonds_bola  = N_bolalipids_positions * 5
    Nangles_bola = N_bolalipids_positions * 4

    Natoms_bila  = N_bilayer_lipids_positions * 6
    Nbonds_bila  = N_bilayer_lipids_positions * 4
    Nangles_bila = N_bilayer_lipids_positions * 2
    
    if single_vesicle:
        Natoms_bila_B, Nbonds_bila_B, Nangles_bila_B = 0, 0, 0
    else:
        Natoms_bila_B  = N_bilayer_lipids_positions * 6
        Nbonds_bila_B  = N_bilayer_lipids_positions * 4
        Nangles_bila_B = N_bilayer_lipids_positions * 2

    # ---- NIC (type 4) heads, balanced between leaflets ----
    n_bola_nic, n_bila_outer, n_bila_inner = split_nic_counts(
        N_bolalipids_positions, N_bilayer_lipids_positions, negative_curved_fraction)
    bola_type4_end = make_bola_type4_end(N_bolalipids_positions, n_bola_nic)
    # offsets interleave the two leaflet picks so outer and inner NICs
    # don't stack on the same bilayer positions
    bila_outer_mask = evenly_spaced_mask(N_bilayer_lipids_positions, n_bila_outer, offset=0.25)
    bila_inner_mask = evenly_spaced_mask(N_bilayer_lipids_positions, n_bila_inner, offset=0.75)

    n_out = int(np.sum(bola_type4_end == 0) + bila_outer_mask.sum())
    n_in = int(np.sum(bola_type4_end == 5) + bila_inner_mask.sum())
    n_mol = N_bolalipids_positions + 2 * N_bilayer_lipids_positions

    inset = chain_half_length + type3_clearance
    raw_drop = LensLattice(geom['c_top'], geom['c_bot'], geom['R_lens'],
                           lattice_spacing, inset)
    droplet_coords = remove_overlaps(raw_drop, coordinates_lipids, type3_clearance)
    N_drop = droplet_coords.shape[0]

    # No lens (alpha_junction = 0) -> no type-3 droplet beads, so NIC heads take
    # type 3 and the file has 3 atom types; otherwise NIC heads are type 4.
    no_lens = alpha <= 0.0
    nic_type = 3 if no_lens else 4
    n_atom_types = 3 if no_lens else 4

    Natoms_membrane = Natoms_bola + Natoms_bila + Natoms_bila_B
    Natoms = Natoms_membrane + N_drop
    Nbonds = Nbonds_bola + Nbonds_bila + Nbonds_bila_B
    Nangles = Nangles_bola + Nangles_bila + Nangles_bila_B

    if single_vesicle:
        L = R_ves + box_buffer
        xbox = ybox = L
        zhi = max(R_ves, geom['apex_top']) + box_buffer
        zlo = -(R_ves + box_buffer)
    else:
        L = max(R_ves, R_ves_B) + box_buffer
        ybox = zbox = L
        xbox = R_ves + R_ves_B + R_intervesicle + box_buffer/2
        
    if not no_lens:
        print(f"[geom] z_j={geom['z_j']:.3f}  a={geom['a']:.3f}  R_lens={geom['R_lens']:.3f}")
        print(f"[geom] lens apex z in [{geom['apex_bot']:.3f}, {geom['apex_top']:.3f}]")
        print(f"[counts] N_ves={geom['N_ves']} N_top={geom['N_top']} N_bot={geom['N_bot']} "
          f"droplet={N_drop}")
        print(f"[NIC] {n_out + n_in} of {n_mol} lipids ({(n_out + n_in) / max(n_mol, 1):.4f}); "
          f"outer={n_out} inner={n_in}  "
          f"[bola {n_bola_nic}, bilayer {int(bila_outer_mask.sum() + bila_inner_mask.sum())}]")
    elif single_vesicle:
        print(f"[geom] single vesicle, no lens")
        print(f"[counts] N_ves={geom['N_ves']} droplet={N_drop}")
        print(f"[NIC] {n_out + n_in} of {n_mol} lipids ({(n_out + n_in) / max(n_mol, 1):.4f}); "
          f"outer={n_out} inner={n_in}  "
          f"[bola {n_bola_nic}, bilayer {int(bila_outer_mask.sum() + bila_inner_mask.sum())}]")
    elif not single_vesicle:
        print(f"[geom] two vesicles, no lens")
        print(f"[geom] x_box={xbox:.3f}    y_box={ybox:.3f}    z_box={zbox:.3f}")
        print(f"[Vesicle A] radius: {R_ves} sigma   NIC: {n_out + n_in} of {n_mol} lipids ({(n_out + n_in) / max(n_mol, 1):.4f}); ")
        print(f"[Vesicle B] radius: {R_ves_B} sigma   NIC: {n_out + n_in} of {n_mol} lipids ({(n_out + n_in) / max(n_mol, 1):.4f}); ")
        
        

    outfile = 'vesicles.data'
    with open(outfile, 'w') as f:
        f.write('LAMMPS data file\n\n')
        f.write(f'{Natoms} atoms\n')
        f.write(f'{Nbonds} bonds\n')
        f.write(f'{Nangles} angles\n\n')
        f.write(f'{n_atom_types} atom types\n')
        f.write('1 bond types\n')
        if fraction_bilayer == 1 and fraction_soft_bolalipids == 0:
            f.write('1 angle types\n')
        else:
            f.write('3 angle types\n')
        f.write('\n')
        f.write(f'{-xbox} {xbox} xlo xhi\n')
        f.write(f'{-ybox} {ybox} ylo yhi\n')
        if single_vesicle:            
            f.write(f'{zlo} {zhi} zlo zhi\n\n')
        else:
            f.write(f'{-zbox} {zbox} zlo zhi\n\n')
        f.write('Masses\n\n')
        for t in range(1, n_atom_types + 1):
            f.write(f'{t} 1\n')
        f.write('\n')
        f.write('Atoms # angle\n\n')

        # --- bola membrane atoms ---
        for i in range(Natoms_bola):
            atom_id = i + 1
            mol_id = i // 6 + 1
            atype = MembraneAtomType(i, bola_type4_end, nic_type)
            x, y, z = coordinates_lipids[i]
            f.write(f"{atom_id} {mol_id} {atype} {x} {y} {z}\n")

        # --- bilayer membrane atoms ---
        for i in range(Natoms_bila):
            atom_id = Natoms_bola + i + 1
            mol_id = i // 3 + Natoms_bola // 6 + 1
            global_atom_index = Natoms_bola + i
            atype = BilayerAtomType(i, bila_outer_mask, bila_inner_mask, nic_type)
            x, y, z = coordinates_lipids[global_atom_index]
            f.write(f"{atom_id} {mol_id} {atype} {x} {y} {z}\n")

        # --- second vesicle bilayer membrane atoms ---
        for i in range(Natoms_bila_B):
            atom_id = Natoms_bola + Natoms_bila + i + 1
            mol_id = i // 3 + Natoms_bola // 6 + Natoms_bila // 3 + 1
            global_atom_index = Natoms_bola + Natoms_bila + i
            atype = BilayerAtomType(i, bila_outer_mask, bila_inner_mask, nic_type)
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
        for i in range(Nbonds_bila_B):
            f.write(
                f"{i+Nbonds_bola+Nbonds_bila+1} 1 "
                f"{i+Nbonds_bola+Nbonds_bila+Nbonds_bila//2+1+i//2} "
                f"{i+Nbonds_bola+Nbonds_bila+Nbonds_bila//2+2+i//2}\n"
            )

        # --- angles (unchanged) ---
        f.write('\nAngles\n\n')
        for i in range(Nangles_bola):
            f.write(f"{i+1} {HeadOrTailAngle(i, fraction_soft_bolalipids)} "
                    f"{i+1+(i//4)*2} {i+2+(i//4)*2} {i+3+(i//4)*2}\n")
        for i in range(Nangles_bila):
            base = i + Natoms_bola
            f.write(f"{i+Nangles_bola+1} 1 "
                    f"{base+1 + (i//1)*2} {base+2 + (i//1)*2} {base+3 + (i//1)*2}\n")
        for i in range(Nangles_bila_B):
            base = i + Natoms_bola + Natoms_bila
            f.write(f"{i+Nangles_bola+Nangles_bila+1} 1 "
                    f"{base+1 + (i//1)*2} {base+2 + (i//1)*2} {base+3 + (i//1)*2}\n")


if __name__ == "__main__":
    main()