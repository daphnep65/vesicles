import numpy as np
import trimesh
import configparser


# ----------------- Helpers for types / angles -----------------

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


def HeadOrTailAngle(n):
    j = (n + 1) % 4 + 1
    if j == 1 or j == 2:
        return 1
    else:
        return 2


# ----------------- Geometry: a bolalipid along a normal -----------------

def BolalipidCoordinates(COM, normal, sigma):
    # Make sure normal is unit length
    nrm = np.linalg.norm(normal)
    if nrm == 0:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / nrm

    bolalipid = np.zeros((6, 3))

    bolalipid[0, :] = COM + sigma * (0.5 + 2) * normal
    bolalipid[1, :] = COM + sigma * (0.5 + 1) * normal
    bolalipid[2, :] = COM + sigma * (0.5) * normal
    bolalipid[3, :] = COM - sigma * (0.5) * normal
    bolalipid[4, :] = COM - sigma * (0.5 + 1) * normal
    bolalipid[5, :] = COM - sigma * (0.5 + 2) * normal

    return bolalipid


# ----------------- Sampling: planar membrane -----------------

def PlanarMembraneCoordinates(length,
                              width,
                              sigma,
                              N_lipids,
                              seed=1):
    """
    Create one planar membrane centered at z = 0.

    Each sampled point is a membrane midplane COM.
    The lipid is placed along the +z normal.
    """

    rng = np.random.default_rng(seed)

    xs = rng.uniform(-length / 2, length / 2, N_lipids)
    ys = rng.uniform(-width / 2, width / 2, N_lipids)
    zs = np.zeros(N_lipids)

    coordinates = np.column_stack([xs, ys, zs])

    normal = np.array([0.0, 0.0, 1.0])

    coordinates_lipids = np.zeros((N_lipids * 6, 3))

    for i in range(N_lipids):
        point = BolalipidCoordinates(coordinates[i], normal, sigma)
        coordinates_lipids[i * 6:(i + 1) * 6, :] = point

    return coordinates_lipids


# ----------------- Sampling: spherical vesicle -----------------

def VesicleCoordinates(radius,
                       center,
                       sigma,
                       N_lipids,
                       seed=1001,
                       sphere_subdivisions=4):
    """
    Create one spherical vesicle.

    radius is the vesicle mid-surface radius.
    center is the vesicle center.
    Lipids are placed along the local outward radial normal.
    """

    sphere = trimesh.creation.icosphere(
        subdivisions=sphere_subdivisions,
        radius=radius
    )

    points, faces = trimesh.sample.sample_surface(
        sphere,
        count=N_lipids,
        seed=seed
    )

    points = points + center

    coordinates_lipids = np.zeros((N_lipids * 6, 3))

    for i in range(N_lipids):
        normal = points[i] - center
        point = BolalipidCoordinates(points[i], normal, sigma)
        coordinates_lipids[i * 6:(i + 1) * 6, :] = point

    return coordinates_lipids


# ----------------- Combined membrane + vesicle coordinates -----------------

def PlaneAndVesicleCoordinates(length,
                               width,
                               vesicle_radius,
                               vesicle_gap,
                               sigma,
                               N_plane_lipids,
                               N_vesicle_lipids,
                               vesicle_x=0.0,
                               vesicle_y=0.0):
    """
    Build:
      - planar membrane centered at z = 0
      - spherical vesicle above the membrane

    vesicle_gap is the vertical distance between:
      - the highest bead of the planar membrane
      - the lowest outer bead of the vesicle

    Since each bolalipid extends 2.5*sigma from its COM,
    the vesicle center is placed at:

        z_center = vesicle_radius + vesicle_gap + 5*sigma

    because:
      plane top bead       = +2.5*sigma
      vesicle bottom bead  = z_center - vesicle_radius - 2.5*sigma

    Therefore:
      vesicle bottom bead - plane top bead = vesicle_gap
    """

    chain_half_length = 2.5 * sigma

    z_center = vesicle_radius + vesicle_gap + 2.0 * chain_half_length

    vesicle_center = np.array([
        vesicle_x,
        vesicle_y,
        z_center
    ])

    plane_coords = PlanarMembraneCoordinates(
        length=length,
        width=width,
        sigma=sigma,
        N_lipids=N_plane_lipids,
        seed=1
    )

    vesicle_coords = VesicleCoordinates(
        radius=vesicle_radius,
        center=vesicle_center,
        sigma=sigma,
        N_lipids=N_vesicle_lipids,
        seed=1001
    )

    coordinates_lipids = np.vstack([
        plane_coords,
        vesicle_coords
    ])

    return coordinates_lipids, vesicle_center


# ----------------- MAIN: write LAMMPS data file -----------------

def main():
    # read in parameters
    cfg = configparser.ConfigParser()
    cfg.read('cfg.ini')

    fraction_bilayer = float(cfg['_'].get('Bilayer_fraction', '0.0'))

    # ---- Geometry parameters ----
    length = float(cfg['_'].get('Plane_length', '88'))
    width  = float(cfg['_'].get('Plane_width',  '88'))

    sigma = float(cfg['_'].get('sigma', '1.0'))

    vesicle_radius = float(cfg['_'].get('Vesicle_radius', '15.0'))
    vesicle_gap = float(cfg['_'].get('Vesicle_gap', '5.0'))

    vesicle_x = float(cfg['_'].get('Vesicle_x', '0.0'))
    vesicle_y = float(cfg['_'].get('Vesicle_y', '0.0'))

    # ---- Lipid numbers ----
    N_plane_lipids = int(cfg['_'].get('N_plane_lipids', '6000'))
    N_vesicle_lipids = int(cfg['_'].get('N_vesicle_lipids', '3000'))

    N_positions = N_plane_lipids + N_vesicle_lipids

    # ---- Generate coordinates ----
    coordinates_lipids, vesicle_center = PlaneAndVesicleCoordinates(
        length=length,
        width=width,
        vesicle_radius=vesicle_radius,
        vesicle_gap=vesicle_gap,
        sigma=sigma,
        N_plane_lipids=N_plane_lipids,
        N_vesicle_lipids=N_vesicle_lipids,
        vesicle_x=vesicle_x,
        vesicle_y=vesicle_y
    )

    # ---- Convert bilayer fraction in same style as your code ----
    fraction_bilayer_heads = fraction_bilayer / (2 - fraction_bilayer)

    N_bolalipids_positions = int(np.floor((1 - fraction_bilayer_heads) * N_positions))
    N_bilayer_lipids_positions = int(np.floor(fraction_bilayer_heads * N_positions))

    Natoms_bola = int(N_bolalipids_positions * 6)
    Nbonds_bola = int(N_bolalipids_positions * 5)
    Nangles_bola = int(N_bolalipids_positions * 4)

    Natoms_bila = int(N_bilayer_lipids_positions * 6)
    Nbonds_bila = int(N_bilayer_lipids_positions * 4)
    Nangles_bila = int(N_bilayer_lipids_positions * 2)

    Natoms = Natoms_bola + Natoms_bila
    Nbonds = Nbonds_bola + Nbonds_bila
    Nangles = Nangles_bola + Nangles_bila

    # ---- Simulation box ----
    chain_half_length = 2.5 * sigma

    xbox = length / 2
    ybox = width / 2

    zlo = -10.0
    zhi = vesicle_center[2] + vesicle_radius + chain_half_length + 10.0

    # Make sure the box also contains the vesicle horizontally
    xbox = max(xbox, abs(vesicle_x) + vesicle_radius + chain_half_length + 10.0)
    ybox = max(ybox, abs(vesicle_y) + vesicle_radius + chain_half_length + 10.0)

    # ---- Write LAMMPS data file ----
    outfile = 'plane_vesicle.data'

    with open(outfile, 'w') as f:
        f.write('LAMMPS data file\n\n')
        f.write(f'{Natoms} atoms\n')
        f.write(f'{Nbonds} bonds\n')
        f.write(f'{Nangles} angles\n\n')

        f.write('2 atom types\n')
        f.write('1 bond types\n')

        if fraction_bilayer == 1:
            f.write('1 angle types\n')
        else:
            f.write('2 angle types\n')

        f.write('\n')
        f.write(f'{-xbox} {xbox} xlo xhi\n')
        f.write(f'{-ybox} {ybox} ylo yhi\n')
        f.write(f'{zlo} {zhi} zlo zhi\n\n')

        f.write('Masses\n\n')
        f.write('1 1\n')
        f.write('2 1\n\n')

        f.write('Atoms # angle\n\n')

        # --- Bolalipid atoms ---
        for i in range(Natoms_bola):
            atom_id = i + 1
            mol_id = i // 6 + 1
            atype = HeadOrTail(i)
            x, y, z = coordinates_lipids[i]
            f.write(f"{atom_id} {mol_id} {atype} {x} {y} {z}\n")

        # --- Bilayer atoms ---
        for i in range(Natoms_bila):
            atom_id = Natoms_bola + i + 1
            mol_id = i // 3 + Natoms_bola // 6 + 1
            atype = HeadOrTail(i + Natoms_bola)
            x, y, z = coordinates_lipids[Natoms_bola + i]
            f.write(f"{atom_id} {mol_id} {atype} {x} {y} {z}\n")

        # --- Bonds ---
        f.write('\nBonds\n\n')

        for i in range(Nbonds_bola):
            f.write(
                f"{i + 1} 1 "
                f"{i + 1 + i // 5} "
                f"{i + 2 + i // 5}\n"
            )

        for i in range(Nbonds_bila):
            f.write(
                f"{i + Nbonds_bola + 1} 1 "
                f"{i + Nbonds_bola + Nbonds_bola // 5 + 1 + i // 2} "
                f"{i + Nbonds_bola + Nbonds_bola // 5 + 2 + i // 2}\n"
            )

        # --- Angles ---
        f.write('\nAngles\n\n')

        for i in range(Nangles_bola):
            f.write(
                f"{i + 1} {HeadOrTailAngle(i)} "
                f"{i + 1 + (i // 4) * 2} "
                f"{i + 2 + (i // 4) * 2} "
                f"{i + 3 + (i // 4) * 2}\n"
            )

        for i in range(Nangles_bila):
            base = i + Nangles_bola + (Nangles_bola // 4) * 2
            f.write(
                f"{i + Nangles_bola + 1} 1 "
                f"{base + 1 + (i // 1) * 2} "
                f"{base + 2 + (i // 1) * 2} "
                f"{base + 3 + (i // 1) * 2}\n"
            )

    print(f"Wrote {outfile}")
    print(f"Plane lipids: {N_plane_lipids}")
    print(f"Vesicle lipids: {N_vesicle_lipids}")
    print(f"Vesicle radius: {vesicle_radius}")
    print(f"Vesicle gap: {vesicle_gap}")
    print(f"Vesicle center: {vesicle_center}")


if __name__ == "__main__":
    main()