use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use std::collections::{HashMap, VecDeque};

const GRID_W: i32 = 20;
const GRID_H: i32 = 15;
const START: GridPos = GridPos { x: 5, y: 5 };

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    fn delta(self) -> (i32, i32) {
        match self {
            Direction::North => (0, 1),
            Direction::South => (0, -1),
            Direction::East => (1, 0),
            Direction::West => (-1, 0),
        }
    }

    // Yaw that makes Bevy's default forward (-Z) align with this world direction.
    fn yaw(self) -> f32 {
        use std::f32::consts::{FRAC_PI_2, PI};
        match self {
            Direction::South => 0.0,
            Direction::North => PI,
            Direction::East => -FRAC_PI_2,
            Direction::West => FRAC_PI_2,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum NavQualifier {
    Closest,
}

#[derive(Copy, Clone, Debug)]
enum Target {
    Energy,
    Base,
}

#[derive(Copy, Clone, Debug)]
enum Action {
    Move(Direction),
    Wait,
    Pickup,
    Drop,
    NavigateTo(NavQualifier, Target),
}

#[derive(Component, Copy, Clone, Debug, PartialEq, Eq)]
struct GridPos {
    x: i32,
    y: i32,
}

#[repr(usize)]
#[derive(Copy, Clone, Debug)]
enum Terrain {
    Grass = 0,
    Dirt = 1,
    Stone = 2,
    Water = 3,
    Wall = 4,
}

impl Terrain {
    const ALL: [Terrain; 5] = [
        Terrain::Grass,
        Terrain::Dirt,
        Terrain::Stone,
        Terrain::Water,
        Terrain::Wall,
    ];

    fn passable(self) -> bool {
        matches!(self, Terrain::Grass | Terrain::Dirt | Terrain::Stone)
    }

    // Map an elevation value in [0, 1] to a terrain band. Bands are tuned for
    // value-noise's central-bias distribution, not uniform — the extreme tails
    // (water, wall) are intentionally narrow.
    fn from_elevation(n: f32) -> Terrain {
        if n < 0.30 {
            Terrain::Water
        } else if n < 0.42 {
            Terrain::Dirt
        } else if n < 0.70 {
            Terrain::Grass
        } else if n < 0.84 {
            Terrain::Stone
        } else {
            Terrain::Wall
        }
    }

    fn color(self) -> Color {
        match self {
            Terrain::Grass => Color::srgb(0.20, 0.36, 0.22),
            Terrain::Dirt => Color::srgb(0.36, 0.26, 0.18),
            Terrain::Stone => Color::srgb(0.45, 0.46, 0.50),
            Terrain::Water => Color::srgb(0.10, 0.28, 0.55),
            Terrain::Wall => Color::srgb(0.16, 0.14, 0.12),
        }
    }
}

#[derive(Resource)]
struct World {
    tiles: Vec<Terrain>,
}

impl World {
    fn idx(x: i32, y: i32) -> usize {
        (y * GRID_W + x) as usize
    }

    fn get(&self, x: i32, y: i32) -> Option<Terrain> {
        if !(0..GRID_W).contains(&x) || !(0..GRID_H).contains(&y) {
            return None;
        }
        Some(self.tiles[Self::idx(x, y)])
    }

    fn generate() -> Self {
        let mut tiles = vec![Terrain::Grass; (GRID_W * GRID_H) as usize];
        for y in 0..GRID_H {
            for x in 0..GRID_W {
                tiles[Self::idx(x, y)] = Terrain::from_elevation(value_noise(x, y));
            }
        }
        // Guarantee worker's starting cell is walkable.
        tiles[Self::idx(START.x, START.y)] = Terrain::Grass;
        let mut world = Self { tiles };
        world.ensure_connected();
        world
    }

    // Carve dirt paths until every passable cell is reachable from START.
    // Each pass flood-fills from START, picks any stranded passable cell,
    // and L-carves toward the nearest reachable cell — barrier tiles along
    // the way become Dirt; passable tiles are left alone.
    fn ensure_connected(&mut self) {
        loop {
            let reachable = self.flood_from(START.x, START.y);
            let stranded = (0..GRID_H)
                .flat_map(|y| (0..GRID_W).map(move |x| (x, y)))
                .find(|&(x, y)| {
                    self.tiles[Self::idx(x, y)].passable() && !reachable[Self::idx(x, y)]
                });
            let Some((sx, sy)) = stranded else { return };

            let (tx, ty) = (0..GRID_H)
                .flat_map(|y| (0..GRID_W).map(move |x| (x, y)))
                .filter(|&(x, y)| reachable[Self::idx(x, y)])
                .min_by_key(|&(x, y)| (x - sx).abs() + (y - sy).abs())
                .expect("START is always reachable");
            self.carve(sx, sy, tx, ty);
        }
    }

    fn flood_from(&self, sx: i32, sy: i32) -> Vec<bool> {
        let mut visited = vec![false; (GRID_W * GRID_H) as usize];
        let mut stack = vec![(sx, sy)];
        while let Some((x, y)) = stack.pop() {
            if !(0..GRID_W).contains(&x) || !(0..GRID_H).contains(&y) {
                continue;
            }
            let i = Self::idx(x, y);
            if visited[i] || !self.tiles[i].passable() {
                continue;
            }
            visited[i] = true;
            stack.extend_from_slice(&[(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]);
        }
        visited
    }

    fn carve(&mut self, mut x: i32, mut y: i32, tx: i32, ty: i32) {
        while (x, y) != (tx, ty) {
            let i = Self::idx(x, y);
            if !self.tiles[i].passable() {
                self.tiles[i] = Terrain::Dirt;
            }
            if x != tx {
                x += (tx - x).signum();
            } else {
                y += (ty - y).signum();
            }
        }
    }
}

// Deterministic 2D hash so terrain is identical across runs without an RNG dep.
fn tile_hash(x: i32, y: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(73_856_093) ^ (y as u32).wrapping_mul(19_349_663);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

// Bilinear-interpolated value noise. tile_hash is sampled on a coarse lattice
// (one sample every NOISE_CELL grid cells), then interpolated with smoothstep
// easing so neighboring tiles get similar values and terrain forms clusters.
// Returns a value in [0, 1].
fn value_noise(x: i32, y: i32) -> f32 {
    const NOISE_CELL: f32 = 4.5;
    let fx = x as f32 / NOISE_CELL;
    let fy = y as f32 / NOISE_CELL;
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let tx = smoothstep(fx - x0 as f32);
    let ty = smoothstep(fy - y0 as f32);
    let v00 = lattice_value(x0, y0);
    let v10 = lattice_value(x0 + 1, y0);
    let v01 = lattice_value(x0, y0 + 1);
    let v11 = lattice_value(x0 + 1, y0 + 1);
    let a = v00 * (1.0 - tx) + v10 * tx;
    let b = v01 * (1.0 - tx) + v11 * tx;
    a * (1.0 - ty) + b * ty
}

fn lattice_value(ix: i32, iy: i32) -> f32 {
    tile_hash(ix, iy) as f32 / u32::MAX as f32
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

#[derive(Component, Default)]
struct Program {
    instructions: Vec<Action>,
    pc: usize,
}

#[derive(Component)]
struct Worker;

#[derive(Component)]
struct EnergyNode;

#[derive(Component, Default)]
struct Inventory {
    energy: u32,
}

#[derive(Component, Default)]
struct Base {
    stored: u32,
}

// Cached navigation plan. While `plan` is non-empty, NavigateTo holds the pc
// and consumes one step per tick. Plan is recomputed lazily whenever it's
// empty at the start of a NavigateTo execution.
#[derive(Component, Default)]
struct NavState {
    plan: VecDeque<Direction>,
}

// Visual interpolation. prev is the world position at the previous fixed tick;
// current is the world position at the latest fixed tick. The Update system
// lerps Transform between them using Time<Fixed>::overstep_fraction().
#[derive(Component, Copy, Clone)]
struct MoveAnim {
    prev: Vec3,
    current: Vec3,
}

// Yaw (radians) of the worker around the Y axis. Default forward is -Z, so
// yaw=0 faces South (-Z world), and rotation is CCW looking from +Y.
#[derive(Component, Copy, Clone)]
struct Facing {
    prev_yaw: f32,
    current_yaw: f32,
}

#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

impl OrbitCamera {
    fn transform(&self) -> Transform {
        let cos_p = self.pitch.cos();
        let pos = self.focus
            + Vec3::new(
                self.distance * self.yaw.sin() * cos_p,
                self.distance * self.pitch.sin(),
                self.distance * self.yaw.cos() * cos_p,
            );
        Transform::from_translation(pos).looking_at(self.focus, Vec3::Y)
    }
}

#[derive(Resource, Default)]
struct Tick(u64);

#[derive(Resource)]
struct Editor {
    source: String,
    status: String,
}

const DEFAULT_PROGRAM: &str = "\
# Gather energy and deliver it home.
# Comments start with '#'. One instruction per line.
navigate_to(closest, energy)
pickup
navigate_to(closest, base)
drop
";

impl Default for Editor {
    fn default() -> Self {
        Self {
            source: DEFAULT_PROGRAM.into(),
            status: "Compile a program to load it onto the worker.".into(),
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Programming Game".into(),
                resolution: (1100u32, 750u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .insert_resource(ClearColor(Color::srgb(0.05, 0.06, 0.09)))
        .insert_resource(Tick::default())
        .insert_resource(Editor::default())
        .insert_resource(Time::<Fixed>::from_hz(4.0))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (orbit_camera_input, interpolate_transforms, spin_energy),
        )
        .add_systems(EguiPrimaryContextPass, editor_ui)
        .add_systems(
            FixedUpdate,
            (
                advance_tick,
                snapshot_anim_state,
                step_workers,
                sync_anim_current,
            )
                .chain(),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Angled top-down camera looking at grid center.
    let orbit = OrbitCamera {
        focus: Vec3::ZERO,
        distance: 22.0,
        yaw: 0.0,
        pitch: 0.85,
    };
    commands.spawn((
        Camera3d::default(),
        orbit.transform(),
        orbit,
        AmbientLight {
            color: Color::srgb(0.75, 0.78, 0.92),
            brightness: 250.0,
            ..default()
        },
    ));

    // Sun.
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(6.0, 14.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Terrain: generate world, pre-create one material per terrain type, then
    // spawn tiles. Walls use a taller mesh; water sits slightly recessed.
    let world = World::generate();
    let slab_mesh = meshes.add(Cuboid::new(0.96, 0.1, 0.96));
    let wall_mesh = meshes.add(Cuboid::new(0.96, 0.7, 0.96));
    let term_mats: [Handle<StandardMaterial>; 5] = std::array::from_fn(|i| {
        let t = Terrain::ALL[i];
        materials.add(StandardMaterial {
            base_color: t.color(),
            perceptual_roughness: if matches!(t, Terrain::Water) { 0.3 } else { 0.9 },
            metallic: if matches!(t, Terrain::Stone) { 0.05 } else { 0.0 },
            ..default()
        })
    });

    for gx in 0..GRID_W {
        for gy in 0..GRID_H {
            let terrain = world.get(gx, gy).unwrap();
            let (x, z) = grid_to_world(gx, gy);
            let (mesh, y) = match terrain {
                Terrain::Wall => (wall_mesh.clone(), 0.35),
                Terrain::Water => (slab_mesh.clone(), -0.05),
                _ => (slab_mesh.clone(), 0.0),
            };
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(term_mats[terrain as usize].clone()),
                Transform::from_xyz(x, y, z),
            ));
        }
    }

    // Scatter energy nodes on grass tiles (skip the worker's start cell).
    let energy_mesh = meshes.add(Cuboid::new(0.32, 0.32, 0.32));
    let energy_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.85, 0.15),
        emissive: LinearRgba::new(0.9, 0.65, 0.1, 1.0),
        perceptual_roughness: 0.3,
        metallic: 0.2,
        ..default()
    });
    for gy in 0..GRID_H {
        for gx in 0..GRID_W {
            if (gx, gy) == (START.x, START.y) {
                continue;
            }
            if !matches!(world.get(gx, gy), Some(Terrain::Grass)) {
                continue;
            }
            // Salts are the two halves of 0x9E3779B9 (golden-ratio mix constant
            // used by xorshift hashes). Reusing tile_hash with a different
            // seed decorrelates resource placement from terrain choice.
            if tile_hash(gx ^ 0x9e37, gy ^ 0x79b9) % 100 < 8 {
                let (x, z) = grid_to_world(gx, gy);
                commands.spawn((
                    EnergyNode,
                    GridPos { x: gx, y: gy },
                    Mesh3d(energy_mesh.clone()),
                    MeshMaterial3d(energy_mat.clone()),
                    Transform::from_xyz(x, 0.35, z)
                        .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_4)),
                ));
            }
        }
    }

    commands.insert_resource(world);

    // Base — sits at the worker's start cell so home == origin.
    let (wx, wz) = grid_to_world(START.x, START.y);
    commands.spawn((
        Base::default(),
        START,
        Mesh3d(meshes.add(Cuboid::new(0.85, 0.14, 0.85))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.55, 0.75),
            emissive: LinearRgba::new(0.05, 0.18, 0.30, 1.0),
            perceptual_roughness: 0.35,
            metallic: 0.4,
            ..default()
        })),
        Transform::from_xyz(wx, 0.07, wz),
    ));

    // Worker.
    let start_world = Vec3::new(wx, 0.45, wz);
    let initial = parse_program(DEFAULT_PROGRAM).unwrap_or_default();
    let initial_yaw = initial
        .iter()
        .find_map(|a| match a {
            Action::Move(d) => Some(d.yaw()),
            _ => None,
        })
        .unwrap_or(0.0);
    let nose_mesh = meshes.add(Cuboid::new(0.22, 0.22, 0.22));
    let nose_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.15, 0.10, 0.05),
        perceptual_roughness: 0.6,
        ..default()
    });
    commands
        .spawn((
            Worker,
            START,
            Inventory::default(),
            NavState::default(),
            Program {
                instructions: initial,
                pc: 0,
            },
            MoveAnim {
                prev: start_world,
                current: start_world,
            },
            Facing {
                prev_yaw: initial_yaw,
                current_yaw: initial_yaw,
            },
            Mesh3d(meshes.add(Cuboid::new(0.7, 0.7, 0.7))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.95, 0.75, 0.25),
                perceptual_roughness: 0.4,
                metallic: 0.1,
                ..default()
            })),
            Transform {
                translation: start_world,
                rotation: Quat::from_rotation_y(initial_yaw),
                ..default()
            },
        ))
        .with_children(|p| {
            // Nose marker — sits just in front of the worker (local -Z face).
            p.spawn((
                Mesh3d(nose_mesh),
                MeshMaterial3d(nose_mat),
                Transform::from_xyz(0.0, 0.05, -0.45),
            ));
        });
}

// Shortest-arc lerp between two yaw angles (radians). Wraps the difference
// into [-π, π] so the rotation never takes the long way around.
fn lerp_yaw(prev: f32, current: f32, t: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut diff = current - prev;
    while diff > PI {
        diff -= TAU;
    }
    while diff < -PI {
        diff += TAU;
    }
    prev + diff * t
}

// Try to walk one cell. Returns the new position if the destination is in
// bounds and passable.
fn try_walk(world: &World, pos: GridPos, dir: Direction) -> Option<GridPos> {
    let (dx, dy) = dir.delta();
    let nx = pos.x + dx;
    let ny = pos.y + dy;
    if matches!(world.get(nx, ny), Some(t) if t.passable()) {
        Some(GridPos { x: nx, y: ny })
    } else {
        None
    }
}

// Face the given direction and try to walk one cell. Centralizes the canonical
// "one tick of motion" used by both Action::Move and the per-tick step of
// Action::NavigateTo, so future motion verbs can't drift apart.
fn step_in_direction(world: &World, pos: &mut GridPos, facing: &mut Facing, dir: Direction) {
    facing.current_yaw = dir.yaw();
    if let Some(new_pos) = try_walk(world, *pos, dir) {
        *pos = new_pos;
    }
}

// BFS from `start` over passable terrain. Returns the path to the first cell
// satisfying `is_target` — which is also the closest such cell, since BFS on
// unit-cost grids expands in order of distance. Returns None if no reachable
// target exists; returns Some(empty) if the start cell itself is a target.
fn find_path(
    world: &World,
    start: GridPos,
    is_target: impl Fn(GridPos) -> bool,
) -> Option<VecDeque<Direction>> {
    if is_target(start) {
        return Some(VecDeque::new());
    }
    // None at the start cell means "no parent"; Some((prev, dir)) elsewhere
    // means "I was reached from prev by stepping `dir`."
    type Parents = HashMap<(i32, i32), Option<((i32, i32), Direction)>>;
    let mut came_from: Parents = HashMap::new();
    let mut queue: VecDeque<(i32, i32)> = VecDeque::new();
    queue.push_back((start.x, start.y));
    came_from.insert((start.x, start.y), None);

    const DIRS: [Direction; 4] = [
        Direction::North,
        Direction::South,
        Direction::East,
        Direction::West,
    ];

    while let Some((cx, cy)) = queue.pop_front() {
        for dir in DIRS {
            let (dx, dy) = dir.delta();
            let nx = cx + dx;
            let ny = cy + dy;
            if came_from.contains_key(&(nx, ny)) {
                continue;
            }
            let Some(t) = world.get(nx, ny) else { continue };
            if !t.passable() {
                continue;
            }
            came_from.insert((nx, ny), Some(((cx, cy), dir)));
            if is_target(GridPos { x: nx, y: ny }) {
                let mut path = VecDeque::new();
                let mut cur = (nx, ny);
                // flatten() collapses both "missing entry" and "start cell" to
                // None, ending the walk at the root without a sentinel value.
                while let Some(((px, py), d)) = came_from.get(&cur).copied().flatten() {
                    path.push_front(d);
                    cur = (px, py);
                }
                return Some(path);
            }
            queue.push_back((nx, ny));
        }
    }
    None
}

// Grid cell (gx, gy) maps to world (x, z). Y is up in 3D and reserved for height.
fn grid_to_world(gx: i32, gy: i32) -> (f32, f32) {
    let ox = -(GRID_W as f32) / 2.0 + 0.5;
    let oz = -(GRID_H as f32) / 2.0 + 0.5;
    (ox + gx as f32, oz + gy as f32)
}

fn parse_program(src: &str) -> Result<Vec<Action>, String> {
    let mut out = Vec::new();
    for (i, line) in src.lines().enumerate() {
        let code = line.split('#').next().unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        // Normalize paren/comma syntax to whitespace so navigate_to(closest,
        // energy) and navigate_to closest energy both parse the same way.
        let normalized = code
            .replace(['(', ')', ','], " ")
            .to_ascii_lowercase();
        let words: Vec<&str> = normalized.split_whitespace().collect();
        let action = match words.as_slice() {
            ["n"] | ["north"] | ["up"] => Action::Move(Direction::North),
            ["s"] | ["south"] | ["down"] => Action::Move(Direction::South),
            ["e"] | ["east"] | ["right"] => Action::Move(Direction::East),
            ["w"] | ["west"] | ["left"] => Action::Move(Direction::West),
            ["wait"] | ["noop"] => Action::Wait,
            ["pickup"] | ["grab"] | ["take"] => Action::Pickup,
            ["drop"] | ["deposit"] | ["deliver"] => Action::Drop,
            [nav, q, t] if matches!(*nav, "navigate_to" | "goto" | "nav") => {
                let qualifier = match *q {
                    "closest" => NavQualifier::Closest,
                    other => {
                        return Err(format!(
                            "line {}: unknown nav qualifier '{}'",
                            i + 1,
                            other
                        ))
                    }
                };
                let target = match *t {
                    "energy" => Target::Energy,
                    "base" => Target::Base,
                    other => {
                        return Err(format!("line {}: unknown nav target '{}'", i + 1, other))
                    }
                };
                Action::NavigateTo(qualifier, target)
            }
            _ => return Err(format!("line {}: unknown instruction '{}'", i + 1, code)),
        };
        out.push(action);
    }
    if out.is_empty() {
        return Err("program is empty".into());
    }
    Ok(out)
}

fn advance_tick(mut tick: ResMut<Tick>) {
    tick.0 = tick.0.wrapping_add(1);
}

// Bevy's Query<Data, Filter> signatures are inherently nested; a type alias
// per system body would be more noise than the inline form.
#[allow(clippy::type_complexity)]
fn step_workers(
    world: Res<World>,
    mut commands: Commands,
    energy_q: Query<(Entity, &GridPos), (With<EnergyNode>, Without<Worker>, Without<Base>)>,
    mut base_q: Query<(&GridPos, &mut Base), (Without<Worker>, Without<EnergyNode>)>,
    mut workers: Query<
        (
            &mut GridPos,
            &mut Program,
            &mut Facing,
            &mut Inventory,
            &mut NavState,
        ),
        (With<Worker>, Without<EnergyNode>, Without<Base>),
    >,
) {
    for (mut pos, mut prog, mut facing, mut inv, mut nav) in &mut workers {
        if prog.instructions.is_empty() {
            continue;
        }
        let action = prog.instructions[prog.pc];
        let mut advance_pc = true;
        match action {
            Action::Move(dir) => {
                // Turn-and-walk; if the destination is blocked the worker
                // still turns toward the obstacle.
                step_in_direction(&world, &mut pos, &mut facing, dir);
            }
            Action::Pickup => {
                for (ent, epos) in &energy_q {
                    if *epos == *pos {
                        commands.entity(ent).despawn();
                        inv.energy += 1;
                        break;
                    }
                }
            }
            Action::Drop => {
                if inv.energy > 0 {
                    for (bpos, mut base) in &mut base_q {
                        if *bpos == *pos {
                            base.stored += inv.energy;
                            inv.energy = 0;
                            break;
                        }
                    }
                }
            }
            Action::Wait => {}
            Action::NavigateTo(_qualifier, target) => {
                // Lazy plan: only recompute when we have no cached steps.
                if nav.plan.is_empty() {
                    let targets: Vec<GridPos> = match target {
                        Target::Energy => energy_q.iter().map(|(_, p)| *p).collect(),
                        Target::Base => base_q.iter().map(|(p, _)| *p).collect(),
                    };
                    // find_path with an empty target set returns None, which
                    // unwrap_or_default collapses to an empty plan — same as
                    // "no path found", so no separate empty-guard is needed.
                    nav.plan = find_path(&world, *pos, |p| targets.contains(&p))
                        .unwrap_or_default();
                }
                if let Some(dir) = nav.plan.pop_front() {
                    step_in_direction(&world, &mut pos, &mut facing, dir);
                }
                // Hold pc on this instruction until the plan is fully drained.
                advance_pc = nav.plan.is_empty();
            }
        }
        if advance_pc {
            prog.pc = (prog.pc + 1) % prog.instructions.len();
        }
    }
}

fn orbit_camera_input(
    mut contexts: EguiContexts,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut q: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let blocked_by_ui = contexts
        .ctx_mut()
        .map(|c| c.is_pointer_over_area() || c.wants_pointer_input())
        .unwrap_or(false);

    let mut delta = Vec2::ZERO;
    for ev in motion.read() {
        delta += ev.delta;
    }
    let mut scroll = 0.0_f32;
    for ev in wheel.read() {
        scroll += match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y * 0.05,
        };
    }

    if blocked_by_ui {
        return;
    }

    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let rmb = buttons.pressed(MouseButton::Right);
    let mmb = buttons.pressed(MouseButton::Middle);

    for (mut orbit, mut tf) in &mut q {
        let mut changed = false;

        if delta != Vec2::ZERO {
            let panning = mmb || (rmb && shift);
            let orbiting = rmb && !shift;

            if panning {
                let right = Vec3::from(tf.right());
                let up = Vec3::from(tf.up());
                let scale = orbit.distance * 0.0015;
                orbit.focus += -right * delta.x * scale + up * delta.y * scale;
                changed = true;
            } else if orbiting {
                orbit.yaw -= delta.x * 0.005;
                // clamp pitch to avoid flipping or going underground
                orbit.pitch = (orbit.pitch - delta.y * 0.005).clamp(0.1, 1.5);
                changed = true;
            }
        }

        if scroll != 0.0 {
            orbit.distance = (orbit.distance * (1.0 - scroll * 0.1)).clamp(3.0, 80.0);
            changed = true;
        }

        if changed {
            *tf = orbit.transform();
        }
    }
}

// Visual flourish — slow Y rotation so energy nodes are visible at a glance.
fn spin_energy(time: Res<Time>, mut q: Query<&mut Transform, With<EnergyNode>>) {
    let dt = time.delta_secs();
    for mut tf in &mut q {
        tf.rotate_y(dt * 1.5);
    }
}

// Runs in FixedUpdate BEFORE step_workers. Snapshots the previous-frame state
// of every interpolated component, so step_workers only has to write "current"
// values. Doing this unconditionally is what prevents the visual "pop-back"
// when a tick doesn't change position or facing (e.g. Pickup, Drop, Wait).
fn snapshot_anim_state(
    mut anim_q: Query<&mut MoveAnim>,
    mut facing_q: Query<&mut Facing>,
) {
    for mut a in &mut anim_q {
        a.prev = a.current;
    }
    for mut f in &mut facing_q {
        f.prev_yaw = f.current_yaw;
    }
}

// Runs in FixedUpdate AFTER step_workers. Re-derives anim.current from the
// (possibly updated) GridPos. Runs unconditionally — when GridPos didn't
// change, current ends up equal to prev and interpolation is a no-op.
fn sync_anim_current(mut q: Query<(&GridPos, &mut MoveAnim)>) {
    for (pos, mut anim) in &mut q {
        let (x, z) = grid_to_world(pos.x, pos.y);
        anim.current = Vec3::new(x, anim.current.y, z);
    }
}

// Runs in Update every frame. Interpolates Transform between prev and current
// using the fraction of time elapsed in the current fixed-tick step. The
// visual lags the simulation by up to one tick but is perfectly smooth.
fn interpolate_transforms(
    fixed_time: Res<Time<Fixed>>,
    mut q: Query<(&MoveAnim, Option<&Facing>, &mut Transform)>,
) {
    let t = fixed_time.overstep_fraction();
    for (anim, facing, mut tf) in &mut q {
        tf.translation = anim.prev.lerp(anim.current, t);
        if let Some(f) = facing {
            let yaw = lerp_yaw(f.prev_yaw, f.current_yaw, t);
            tf.rotation = Quat::from_rotation_y(yaw);
        }
    }
}

fn editor_ui(
    mut contexts: EguiContexts,
    mut editor: ResMut<Editor>,
    mut q: Query<(&mut Program, &Inventory), With<Worker>>,
    bases: Query<&Base>,
    tick: Res<Tick>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::SidePanel::left("editor")
        .default_width(260.0)
        .show(ctx, |ui| {
            ui.heading("Worker Program");
            ui.label("One instruction per line. Tokens:");
            ui.monospace("N S E W Wait Pickup Drop");
            ui.monospace("navigate_to(closest, energy|base)");
            ui.label("'#' starts a comment.");
            ui.separator();

            ui.add(
                egui::TextEdit::multiline(&mut editor.source)
                    .desired_rows(16)
                    .desired_width(f32::INFINITY)
                    .font(egui::TextStyle::Monospace),
            );

            if ui.button("Compile & Load").clicked() {
                match parse_program(&editor.source) {
                    Ok(instrs) => {
                        let n = instrs.len();
                        for (mut prog, _) in &mut q {
                            prog.instructions = instrs.clone();
                            prog.pc = 0;
                        }
                        editor.status = format!("Loaded {n} instruction(s).");
                    }
                    Err(e) => {
                        editor.status = format!("Error: {e}");
                    }
                }
            }

            ui.separator();
            ui.label(&editor.status);
            ui.separator();
            ui.label(format!("tick: {}", tick.0));
            if let Ok((prog, inv)) = q.single() {
                ui.label(format!(
                    "pc: {} / {}",
                    prog.pc,
                    prog.instructions.len()
                ));
                ui.label(format!("energy: {}", inv.energy));
            }
            if let Ok(base) = bases.single() {
                ui.label(format!("delivered: {}", base.stored));
            }
            ui.separator();
            ui.label("Camera:");
            ui.monospace("RMB drag  — orbit");
            ui.monospace("Shift+RMB — pan");
            ui.monospace("MMB drag  — pan");
            ui.monospace("Scroll    — zoom");
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_grass_world() -> World {
        World {
            tiles: vec![Terrain::Grass; (GRID_W * GRID_H) as usize],
        }
    }

    fn place(world: &mut World, x: i32, y: i32, t: Terrain) {
        world.tiles[World::idx(x, y)] = t;
    }

    // Standalone BFS so connectivity assertions aren't tautological with
    // World::flood_from — if both broke the same way, the test would still pass.
    fn bfs_reachable(world: &World, sx: i32, sy: i32) -> Vec<bool> {
        let mut seen = vec![false; (GRID_W * GRID_H) as usize];
        let mut queue = VecDeque::from([(sx, sy)]);
        while let Some((x, y)) = queue.pop_front() {
            if !(0..GRID_W).contains(&x) || !(0..GRID_H).contains(&y) {
                continue;
            }
            let i = World::idx(x, y);
            if seen[i] || !world.tiles[i].passable() {
                continue;
            }
            seen[i] = true;
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                queue.push_back((x + dx, y + dy));
            }
        }
        seen
    }

    // --- find_path -----------------------------------------------------------

    #[test]
    fn find_path_returns_empty_when_start_is_goal() {
        let w = flat_grass_world();
        let p = find_path(&w, GridPos { x: 5, y: 5 }, |p| p == GridPos { x: 5, y: 5 });
        assert_eq!(p, Some(VecDeque::new()));
    }

    #[test]
    fn find_path_walks_a_straight_line_on_open_grass() {
        let w = flat_grass_world();
        let p = find_path(&w, GridPos { x: 5, y: 5 }, |p| p == GridPos { x: 8, y: 5 }).unwrap();
        assert_eq!(p, VecDeque::from(vec![Direction::East; 3]));
    }

    #[test]
    fn find_path_detours_around_a_wall() {
        let mut w = flat_grass_world();
        // Single-cell wall directly east of (5,5); detour must go north or south.
        place(&mut w, 6, 5, Terrain::Wall);
        let p = find_path(&w, GridPos { x: 5, y: 5 }, |p| p == GridPos { x: 7, y: 5 }).unwrap();
        // Manhattan distance is 2; the detour adds 2 extra steps.
        assert_eq!(p.len(), 4);
    }

    #[test]
    fn find_path_returns_none_when_target_is_walled_in() {
        let mut w = flat_grass_world();
        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            place(&mut w, 7 + dx, 5 + dy, Terrain::Wall);
        }
        let p = find_path(&w, GridPos { x: 5, y: 5 }, |p| p == GridPos { x: 7, y: 5 });
        assert_eq!(p, None);
    }

    #[test]
    fn find_path_returns_none_when_no_cell_matches() {
        let w = flat_grass_world();
        let p = find_path(&w, GridPos { x: 5, y: 5 }, |_| false);
        assert_eq!(p, None);
    }

    #[test]
    fn find_path_picks_the_closest_of_multiple_targets() {
        let w = flat_grass_world();
        let near = GridPos { x: 5, y: 6 };
        let far = GridPos { x: 5, y: 12 };
        let p = find_path(&w, GridPos { x: 5, y: 5 }, |p| p == near || p == far).unwrap();
        // BFS expands by distance, so the 1-step target wins over the 7-step one.
        assert_eq!(p, VecDeque::from(vec![Direction::North]));
    }

    // --- try_walk ------------------------------------------------------------

    #[test]
    fn try_walk_moves_into_grass() {
        let w = flat_grass_world();
        assert_eq!(
            try_walk(&w, GridPos { x: 5, y: 5 }, Direction::East),
            Some(GridPos { x: 6, y: 5 })
        );
    }

    #[test]
    fn try_walk_is_blocked_by_walls() {
        let mut w = flat_grass_world();
        place(&mut w, 6, 5, Terrain::Wall);
        assert_eq!(try_walk(&w, GridPos { x: 5, y: 5 }, Direction::East), None);
    }

    #[test]
    fn try_walk_is_blocked_by_water() {
        let mut w = flat_grass_world();
        place(&mut w, 6, 5, Terrain::Water);
        assert_eq!(try_walk(&w, GridPos { x: 5, y: 5 }, Direction::East), None);
    }

    #[test]
    fn try_walk_is_blocked_at_grid_boundary() {
        let w = flat_grass_world();
        assert_eq!(
            try_walk(&w, GridPos { x: GRID_W - 1, y: 5 }, Direction::East),
            None
        );
    }

    // --- snapshot + sync (interpolation pop-back regression) -----------------

    fn run_snapshot_sync_once(world: &mut bevy::ecs::world::World) {
        let mut schedule = Schedule::default();
        schedule.add_systems((snapshot_anim_state, sync_anim_current).chain());
        schedule.run(world);
    }

    #[test]
    fn interpolation_does_not_pop_back_when_grid_pos_unchanged() {
        let mut world = bevy::ecs::world::World::new();
        let (x, z) = grid_to_world(1, 2);
        let initial = Vec3::new(x, 0.45, z);
        // Seed with a stale prev (simulating "worker just finished a move last tick").
        let entity = world
            .spawn((
                GridPos { x: 1, y: 2 },
                MoveAnim {
                    prev: Vec3::new(99.0, 0.45, 99.0),
                    current: initial,
                },
            ))
            .id();

        run_snapshot_sync_once(&mut world);

        let anim = world.get::<MoveAnim>(entity).unwrap();
        // After the snapshot+sync cycle on an unchanged GridPos, prev and current
        // must coincide so the next interpolation lerp is a no-op (no pop-back).
        assert_eq!(anim.prev, initial);
        assert_eq!(anim.current, initial);
    }

    #[test]
    fn interpolation_tracks_motion_across_one_tick() {
        let mut world = bevy::ecs::world::World::new();
        let (x0, z0) = grid_to_world(1, 2);
        let initial = Vec3::new(x0, 0.45, z0);
        let entity = world
            .spawn((
                GridPos { x: 1, y: 2 },
                MoveAnim {
                    prev: initial,
                    current: initial,
                },
            ))
            .id();

        // First tick: nothing has moved — confirm steady-state.
        run_snapshot_sync_once(&mut world);
        let anim = world.get::<MoveAnim>(entity).unwrap();
        assert_eq!(anim.prev, initial);
        assert_eq!(anim.current, initial);

        // Simulate step_workers moving the worker east between snapshot and sync.
        // The schedule runs both systems in one pass, so we mutate GridPos first
        // and let the next cycle observe it.
        world.get_mut::<GridPos>(entity).unwrap().x = 2;
        run_snapshot_sync_once(&mut world);

        let (x1, z1) = grid_to_world(2, 2);
        let anim = world.get::<MoveAnim>(entity).unwrap();
        assert_eq!(anim.prev, initial, "prev must be where we were last tick");
        assert_eq!(
            anim.current,
            Vec3::new(x1, 0.45, z1),
            "current must reflect the new GridPos"
        );
    }

    #[test]
    fn snapshot_pins_facing_prev_to_current() {
        let mut world = bevy::ecs::world::World::new();
        // Seed with a stale prev_yaw, as would happen one tick after a Move.
        let entity = world
            .spawn(Facing {
                prev_yaw: 0.0,
                current_yaw: 1.5,
            })
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems(snapshot_anim_state);
        schedule.run(&mut world);

        let f = world.get::<Facing>(entity).unwrap();
        // Snapshot must collapse prev onto current so interpolation is a no-op
        // until step_workers writes a new current_yaw.
        assert_eq!(f.prev_yaw, 1.5);
        assert_eq!(f.current_yaw, 1.5);
    }

    // --- world generation ----------------------------------------------------

    #[test]
    fn start_cell_is_passable() {
        let world = World::generate();
        assert!(world.get(START.x, START.y).unwrap().passable());
    }

    #[test]
    fn all_passable_cells_reachable_from_start() {
        let world = World::generate();
        let seen = bfs_reachable(&world, START.x, START.y);
        for y in 0..GRID_H {
            for x in 0..GRID_W {
                let t = world.get(x, y).unwrap();
                if t.passable() {
                    assert!(
                        seen[World::idx(x, y)],
                        "passable cell ({x}, {y}) = {t:?} is unreachable from START"
                    );
                }
            }
        }
    }

    #[test]
    fn ensure_connected_links_isolated_islands() {
        // Wall everywhere except two lone grass cells in opposite corners.
        let mut tiles = vec![Terrain::Wall; (GRID_W * GRID_H) as usize];
        tiles[World::idx(START.x, START.y)] = Terrain::Grass;
        let far = (GRID_W - 1, GRID_H - 1);
        tiles[World::idx(far.0, far.1)] = Terrain::Grass;

        let mut world = World { tiles };
        world.ensure_connected();

        let seen = bfs_reachable(&world, START.x, START.y);
        assert!(
            seen[World::idx(far.0, far.1)],
            "far corner should be reachable after ensure_connected"
        );
    }

    #[test]
    fn value_noise_stays_in_unit_interval() {
        for y in -3..GRID_H + 3 {
            for x in -3..GRID_W + 3 {
                let n = value_noise(x, y);
                assert!((0.0..=1.0).contains(&n), "value_noise({x},{y}) = {n}");
            }
        }
    }
}
