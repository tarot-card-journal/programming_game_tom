use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use std::collections::{HashMap, HashSet, VecDeque};

const GRID_W: i32 = 20;
const GRID_H: i32 = 15;
const START: GridPos = GridPos { x: 5, y: 5 };
const NUM_WORKERS: usize = 10;
// A worker whose movement intent is blocked this many ticks in a row gives
// up — its carried energy spills onto its last tile and the entity
// despawns. Only blocked Move / NavigateTo steps increment the counter;
// Wait, Pickup, Drop, and ticks where NavigateTo finds no plan leave it
// alone. The counter resets on any successful step, so productive workers
// never die.
const MAX_BUMPS: u32 = 10;
const CARDINAL_DIRS: [Direction; 4] = [
    Direction::North,
    Direction::South,
    Direction::East,
    Direction::West,
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    fn delta(self) -> (i32, i32) {
        // Camera looks toward -Z from a +Z vantage point, so -Z appears at
        // the top of the screen. North is "up on screen / away from player",
        // which is -Z in world == -Y on the grid.
        match self {
            Direction::North => (0, -1),
            Direction::South => (0, 1),
            Direction::East => (1, 0),
            Direction::West => (-1, 0),
        }
    }

    // Yaw that makes Bevy's default forward (-Z) align with this world direction.
    fn yaw(self) -> f32 {
        use std::f32::consts::{FRAC_PI_2, PI};
        match self {
            Direction::North => 0.0,
            Direction::South => PI,
            Direction::East => -FRAC_PI_2,
            Direction::West => FRAC_PI_2,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NavQualifier {
    Closest,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Target {
    Resource(ResourceKind),
    Base,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Action {
    Move(Direction),
    Wait,
    // None = pick any resource on the tile; Some(kind) = filter by kind.
    Pickup(Option<ResourceKind>),
    Drop,
    NavigateTo(NavQualifier, Target),
    // Branching IR — emitted by the parser when compiling if/else and
    // while/until blocks. Users don't write these directly; they're flat-IR
    // equivalents of the block syntax. The usize is an instruction index
    // into the same Program.
    Jump(usize),
    JumpIf(Condition, usize),
    JumpUnless(Condition, usize),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Condition {
    // True when the worker's inventory queue contains at least one of this kind.
    Carrying(ResourceKind),
}

// Per-tick tile lock key. Each (GridPos, TileAction) pair can be acquired at
// most once per tick — the first worker to try wins. Generalises "two
// workers can't both pickup from the same tile this tick" to anything else
// that needs the same uniqueness later (combat, building, etc.).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum TileAction {
    Pickup,
}

// Collectible resource kinds. #[repr(usize)] so `kind as usize` indexes the
// per-kind arrays on Inventory and Base — keeps pickup/drop allocation-free
// and avoids parallel `if let` branches per resource type.
#[repr(usize)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ResourceKind {
    Energy = 0,
    Grass = 1,
    Wood = 2,
}

impl ResourceKind {
    // ALL is the single source of truth for the kind list; COUNT derives
    // from it so the two can't drift if a new variant is added.
    const ALL: [Self; 3] = [Self::Energy, Self::Grass, Self::Wood];
    const COUNT: usize = Self::ALL.len();

    fn label(self) -> &'static str {
        match self {
            Self::Energy => "energy",
            Self::Grass => "grass",
            Self::Wood => "wood",
        }
    }

    fn from_token(token: &str) -> Option<ResourceKind> {
        match token {
            "energy" => Some(Self::Energy),
            "grass" => Some(Self::Grass),
            "wood" => Some(Self::Wood),
            _ => None,
        }
    }

    // Per-kind material parameters. Kept here (rather than positionally in
    // ResourceAssets::build) so the kind → material mapping is checked by
    // match exhaustiveness — adding a new variant forces a deliberate choice
    // instead of silently shifting which color belongs to which kind.
    fn material_spec(self) -> StandardMaterial {
        match self {
            Self::Energy => StandardMaterial {
                base_color: Color::srgb(1.0, 0.85, 0.15),
                emissive: LinearRgba::new(0.9, 0.65, 0.1, 1.0),
                perceptual_roughness: 0.3,
                metallic: 0.2,
                ..default()
            },
            Self::Grass => StandardMaterial {
                base_color: Color::srgb(0.35, 0.78, 0.30),
                emissive: LinearRgba::new(0.04, 0.10, 0.03, 1.0),
                perceptual_roughness: 0.85,
                ..default()
            },
            Self::Wood => StandardMaterial {
                base_color: Color::srgb(0.42, 0.27, 0.13),
                perceptual_roughness: 0.9,
                ..default()
            },
        }
    }

    // Per-kind mesh shape, ground-level Y, and base Z-rotation. Same rationale
    // as material_spec — match arms keep kind → look correspondence explicit.
    fn visual_spec(self) -> ResourceVisualSpec {
        use std::f32::consts::FRAC_PI_4;
        match self {
            Self::Energy => ResourceVisualSpec {
                mesh_size: Vec3::splat(0.32),
                y: 0.35,
                rotation_y: FRAC_PI_4,
            },
            Self::Grass => ResourceVisualSpec {
                mesh_size: Vec3::new(0.5, 0.18, 0.5),
                y: 0.14,
                rotation_y: 0.0,
            },
            Self::Wood => ResourceVisualSpec {
                mesh_size: Vec3::new(0.6, 0.22, 0.22),
                y: 0.16,
                rotation_y: 0.0,
            },
        }
    }
}

struct ResourceVisualSpec {
    mesh_size: Vec3,
    y: f32,
    rotation_y: f32,
}

#[derive(Component, Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

// Named ResourceNode (not Resource) to avoid shadowing bevy::prelude::Resource,
// and to mirror the original EnergyNode component this generalizes.
#[derive(Component, Copy, Clone)]
struct ResourceNode {
    kind: ResourceKind,
}

// One of the worker's carry-indicator child entities. The usize is the queue
// index this slot mirrors (0 = oldest, 1 = newest).
#[derive(Component, Copy, Clone)]
struct CarrySlot(usize);

// Local-Y position of each carry slot above the worker. Array length is tied
// to Inventory::CAPACITY so a capacity change is a compile error here, not a
// silently-missing slot at runtime.
const CARRY_SLOT_Y: [f32; Inventory::CAPACITY] = [0.55, 0.85];

// FIFO queue of carried resources, bounded to CAPACITY. On overflow, `push`
// returns the evicted oldest entry to the caller — step_workers' Pickup arm
// respawns it onto the worker's tile as a fresh ResourceNode so it remains
// collectible.
#[derive(Component, Default)]
struct Inventory {
    queue: VecDeque<ResourceKind>,
}

impl Inventory {
    const CAPACITY: usize = 2;

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    // Push `kind` onto the back of the queue. Returns the kind that was
    // evicted from the front, if the queue was already at capacity, so callers
    // can decide what to do with it (e.g. respawn it onto the worker's tile).
    fn push(&mut self, kind: ResourceKind) -> Option<ResourceKind> {
        self.queue.push_back(kind);
        if self.queue.len() > Self::CAPACITY {
            self.queue.pop_front()
        } else {
            None
        }
    }

    fn drain_into(&mut self, counts: &mut [u32; ResourceKind::COUNT]) {
        while let Some(k) = self.queue.pop_front() {
            counts[k as usize] += 1;
        }
    }
}

// Tally of consecutive ticks where a movement intent (Move or NavigateTo
// step) was blocked. Non-moving instructions (Wait/Pickup/Drop) and ticks
// where NavigateTo has no plan don't change the counter. Reset to 0 on any
// successful step. When this reaches `MAX_BUMPS` the worker despawns and
// spills its inventory at its current tile.
#[derive(Component, Default, Debug, Clone, Copy)]
struct BumpCount(u32);

#[derive(Component, Default)]
struct Base {
    stored: [u32; ResourceKind::COUNT],
}

// Cached navigation plan. While `plan` is non-empty, NavigateTo holds the pc
// and consumes one step per tick. Plan is recomputed lazily whenever it's
// empty at the start of a NavigateTo execution.
//
// `reserved_tile` is the tile this worker has reserved while pathing toward
// a resource. step_workers consults every worker's claim before picking a
// fresh target so two workers don't converge on the same node. A claim is
// set only when navigate_to(resource) computes a non-empty path — if the
// worker is already standing on the target tile no claim is needed (the
// next Pickup fires immediately). The claim is released when the worker
// picks up at the reserved tile, re-plans navigate_to, or the program is
// recompiled.
#[derive(Component, Default)]
struct NavState {
    plan: VecDeque<Direction>,
    reserved_tile: Option<GridPos>,
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

// Shared per-kind rendering bundle (mesh + material + ground placement). Used
// by the world-scatter loop, the eviction-respawn path in step_workers, and
// the carry indicator on each worker — one source of truth so a respawned
// resource matches its original visual exactly.
struct ResourceVisual {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    y: f32,
    rotation: Quat,
}

#[derive(Resource)]
struct ResourceAssets {
    visuals: [ResourceVisual; ResourceKind::COUNT],
}

impl ResourceAssets {
    fn build(meshes: &mut Assets<Mesh>, materials: &mut Assets<StandardMaterial>) -> Self {
        Self {
            visuals: ResourceKind::ALL.map(|kind| {
                let spec = kind.visual_spec();
                ResourceVisual {
                    mesh: meshes.add(Cuboid::from_size(spec.mesh_size)),
                    material: materials.add(kind.material_spec()),
                    y: spec.y,
                    rotation: Quat::from_rotation_y(spec.rotation_y),
                }
            }),
        }
    }

    fn visual(&self, kind: ResourceKind) -> &ResourceVisual {
        &self.visuals[kind as usize]
    }

    fn material_for(&self, kind: ResourceKind) -> Handle<StandardMaterial> {
        self.visual(kind).material.clone()
    }
}

// Spawn a resource node at the given grid position using the shared visual.
// Used both during initial world scatter and when an inventory overflow
// evicts a resource back to the worker's tile.
fn spawn_resource_node(
    commands: &mut Commands,
    assets: &ResourceAssets,
    kind: ResourceKind,
    pos: GridPos,
) {
    let v = assets.visual(kind);
    let (x, z) = grid_to_world(pos.x, pos.y);
    commands.spawn((
        ResourceNode { kind },
        pos,
        Mesh3d(v.mesh.clone()),
        MeshMaterial3d(v.material.clone()),
        Transform::from_xyz(x, v.y, z).with_rotation(v.rotation),
    ));
}

#[derive(Resource, Default)]
struct Tick(u64);

#[derive(Resource)]
struct Editor {
    source: String,
    status: String,
}

const DEFAULT_PROGRAM: &str = "\
# Gather one energy, then deliver. The outer pc still wraps at end-of-program,
# so this restarts forever. Comments start with '#'.
while not carrying(energy) {
  navigate_to(closest, energy)
  pickup(energy)
}
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
            (
                orbit_camera_input,
                interpolate_transforms,
                spin_energy,
                update_carry_display,
            ),
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

    // Per-kind world-scatter rates. Visuals come from ResourceAssets; only
    // the placement tuning (salt + chance) lives here, since respawned
    // resources don't go through a placement roll.
    //
    // Salts are arbitrary 32-bit constants; the originals are the two halves
    // of 0x9E3779B9 (golden-ratio mix used by xorshift) for energy. Grass and
    // wood use unrelated values to keep the three rolls decorrelated.
    let resource_assets = ResourceAssets::build(&mut meshes, &mut materials);
    let spawn_table: [(ResourceKind, i32, i32, u32); ResourceKind::COUNT] = [
        (ResourceKind::Energy, 0x9e37, 0x79b9, 8),
        (ResourceKind::Grass, 0x51a5, 0xb47c, 12),
        (ResourceKind::Wood, 0x2c8d, 0xe6f1, 8),
    ];

    // Track tiles that received any resource so worker spawn positions can
    // avoid them (otherwise a worker would trigger an instant pickup on tick
    // 1). A tile may appear here multiple times when kinds overlap — fine,
    // the HashSet de-dupes.
    let mut resource_positions: HashSet<GridPos> = HashSet::new();
    for gy in 0..GRID_H {
        for gx in 0..GRID_W {
            if (gx, gy) == (START.x, START.y) {
                continue;
            }
            if !matches!(world.get(gx, gy), Some(Terrain::Grass)) {
                continue;
            }
            for &(kind, salt_x, salt_y, chance) in &spawn_table {
                if tile_hash(gx ^ salt_x, gy ^ salt_y) % 100 < chance {
                    let pos = GridPos { x: gx, y: gy };
                    resource_positions.insert(pos);
                    spawn_resource_node(&mut commands, &resource_assets, kind, pos);
                }
            }
        }
    }

    // Stress-test cluster: pile energy onto every passable cell in the
    // lower-right of the map. ensure_connected guarantees the region is
    // reachable from START via a (typically narrow) carved corridor, so
    // workers all converge there and trigger the bump/re-plan/despawn
    // paths in normal play.
    for gy in 10..GRID_H {
        for gx in (GRID_W - 6)..GRID_W {
            let pos = GridPos { x: gx, y: gy };
            if pos == START {
                continue;
            }
            if !matches!(world.get(gx, gy), Some(t) if t.passable()) {
                continue;
            }
            if resource_positions.insert(pos) {
                spawn_resource_node(&mut commands, &resource_assets, ResourceKind::Energy, pos);
            }
        }
    }

    // Compute spawn cells before moving `world` into the ECS resource. Skip
    // tiles that already hold a resource so workers don't materialise on top
    // of one (which would trigger an instant pickup on tick 1).
    let worker_starts = worker_start_positions(&world, START, NUM_WORKERS, &resource_positions);
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

    // Workers — share one mesh + material so adding more is cheap. The base
    // sits on START, so worker_start_positions hands back START itself first
    // and the remaining workers cluster outward over passable terrain.
    let initial = parse_program(DEFAULT_PROGRAM).unwrap_or_default();
    let initial_yaw = initial
        .iter()
        .find_map(|a| match a {
            Action::Move(d) => Some(d.yaw()),
            _ => None,
        })
        .unwrap_or(0.0);
    let assets = WorkerAssets {
        body_mesh: meshes.add(Cuboid::new(0.7, 0.7, 0.7)),
        body_mat: materials.add(StandardMaterial {
            base_color: Color::srgb(0.95, 0.75, 0.25),
            perceptual_roughness: 0.4,
            metallic: 0.1,
            ..default()
        }),
        nose_mesh: meshes.add(Cuboid::new(0.22, 0.22, 0.22)),
        nose_mat: materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.10, 0.05),
            perceptual_roughness: 0.6,
            ..default()
        }),
        // Carry-slot mesh is uniform across kinds; update_carry_display swaps
        // the material per slot from ResourceAssets each frame. Energy's
        // material is just a placeholder so the slot has a valid handle while
        // it's still hidden.
        slot_mesh: meshes.add(Cuboid::new(0.22, 0.22, 0.22)),
        slot_placeholder_mat: resource_assets.material_for(ResourceKind::Energy),
    };
    for grid_pos in worker_starts {
        spawn_worker(&mut commands, grid_pos, initial.clone(), initial_yaw, &assets);
    }
    commands.insert_resource(resource_assets);
}

// Mesh/material handles shared across every worker spawn. Bundling them keeps
// spawn_worker's signature short and makes "one extra clone per worker" the
// obvious cost model.
struct WorkerAssets {
    body_mesh: Handle<Mesh>,
    body_mat: Handle<StandardMaterial>,
    nose_mesh: Handle<Mesh>,
    nose_mat: Handle<StandardMaterial>,
    slot_mesh: Handle<Mesh>,
    slot_placeholder_mat: Handle<StandardMaterial>,
}

fn spawn_worker(
    commands: &mut Commands,
    grid_pos: GridPos,
    program: Vec<Action>,
    initial_yaw: f32,
    assets: &WorkerAssets,
) {
    let (x, z) = grid_to_world(grid_pos.x, grid_pos.y);
    let world_pos = Vec3::new(x, 0.45, z);
    commands
        .spawn((
            Worker,
            grid_pos,
            Inventory::default(),
            NavState::default(),
            BumpCount::default(),
            Program {
                instructions: program,
                pc: 0,
            },
            MoveAnim {
                prev: world_pos,
                current: world_pos,
            },
            Facing {
                prev_yaw: initial_yaw,
                current_yaw: initial_yaw,
            },
            Mesh3d(assets.body_mesh.clone()),
            MeshMaterial3d(assets.body_mat.clone()),
            Transform {
                translation: world_pos,
                rotation: Quat::from_rotation_y(initial_yaw),
                ..default()
            },
        ))
        .with_children(|p| {
            // Nose marker — sits just in front of the worker (local -Z face).
            p.spawn((
                Mesh3d(assets.nose_mesh.clone()),
                MeshMaterial3d(assets.nose_mat.clone()),
                Transform::from_xyz(0.0, 0.05, -0.45),
            ));
            // Carry-slot indicators above the worker. Hidden until the worker
            // picks something up; update_carry_display swaps the material and
            // visibility from the Inventory queue each frame.
            for (slot, &slot_y) in CARRY_SLOT_Y.iter().enumerate().take(Inventory::CAPACITY) {
                p.spawn((
                    CarrySlot(slot),
                    Mesh3d(assets.slot_mesh.clone()),
                    MeshMaterial3d(assets.slot_placeholder_mat.clone()),
                    Transform::from_xyz(0.0, slot_y, 0.0),
                    Visibility::Hidden,
                ));
            }
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
// Action::NavigateTo, so future motion verbs can't drift apart. Returns
// `true` if the worker moved, `false` if blocked (by terrain or by another
// worker in `occupied`). Callers driving a cached path use the return value
// to decide whether to consume the planned step.
//
// `occupied` is the set of tiles currently held by *other* workers (plus the
// caller's own tile). If the destination is in the set the worker stays put
// — facing still updates so they visibly turn toward the blocker.
fn step_in_direction(
    world: &World,
    pos: &mut GridPos,
    facing: &mut Facing,
    dir: Direction,
    occupied: &mut HashSet<GridPos>,
) -> bool {
    facing.current_yaw = dir.yaw();
    let Some(new_pos) = try_walk(world, *pos, dir) else {
        return false;
    };
    if occupied.contains(&new_pos) {
        return false;
    }
    occupied.remove(pos);
    occupied.insert(new_pos);
    *pos = new_pos;
    true
}

// BFS from `start` over passable terrain, treating cells in `blocked` as
// impassable. Returns the path to the first cell satisfying `is_target` —
// also the closest such cell since BFS on unit-cost grids expands in order
// of distance. Returns None if no reachable target exists; returns
// Some(empty) if the start cell itself is a target. Pass `&HashSet::new()`
// when there are no dynamic obstacles to consider.
fn find_path(
    world: &World,
    start: GridPos,
    is_target: impl Fn(GridPos) -> bool,
    blocked: &HashSet<GridPos>,
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

    while let Some((cx, cy)) = queue.pop_front() {
        for dir in CARDINAL_DIRS {
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
            let cell = GridPos { x: nx, y: ny };
            if blocked.contains(&cell) {
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

// BFS outward from `start` over passable terrain, collecting up to `n` cells
// that aren't in `excluded`. Used at spawn time to lay out N workers in a
// connected cluster near the base without stacking them on the same tile or
// landing them on top of existing entities (e.g. energy nodes). BFS still
// expands *through* excluded cells, so the cluster can extend past them. If
// `start` is in `excluded` or not passable, or fewer than `n` eligible
// cells are reachable, the returned vector is correspondingly shorter.
fn worker_start_positions(
    world: &World,
    start: GridPos,
    n: usize,
    excluded: &HashSet<GridPos>,
) -> Vec<GridPos> {
    let mut out = Vec::with_capacity(n);
    if n == 0 || !matches!(world.get(start.x, start.y), Some(t) if t.passable()) {
        return out;
    }
    let mut visited: HashSet<(i32, i32)> = HashSet::new();
    let mut queue: VecDeque<(i32, i32)> = VecDeque::new();
    queue.push_back((start.x, start.y));
    visited.insert((start.x, start.y));

    while let Some((cx, cy)) = queue.pop_front() {
        let cell = GridPos { x: cx, y: cy };
        if !excluded.contains(&cell) {
            out.push(cell);
            if out.len() == n {
                return out;
            }
        }
        for dir in CARDINAL_DIRS {
            let (dx, dy) = dir.delta();
            let nx = cx + dx;
            let ny = cy + dy;
            if visited.contains(&(nx, ny)) {
                continue;
            }
            let Some(t) = world.get(nx, ny) else { continue };
            if !t.passable() {
                continue;
            }
            visited.insert((nx, ny));
            queue.push_back((nx, ny));
        }
    }
    out
}

// Grid cell (gx, gy) maps to world (x, z). Y is up in 3D and reserved for height.
fn grid_to_world(gx: i32, gy: i32) -> (f32, f32) {
    let ox = -(GRID_W as f32) / 2.0 + 0.5;
    let oz = -(GRID_H as f32) / 2.0 + 0.5;
    (ox + gx as f32, oz + gy as f32)
}

// Open block tracked during parsing. The held usize is the instruction index
// of a pending Jump/JumpUnless whose target needs to be patched once we see
// the closing `}` or `} else {`.
enum BlockFrame {
    // Open `if`. The held index is the JumpUnless that should land at the
    // start of an `else` (if one appears) or at the first instruction past
    // the body (otherwise). `line` is the line number of the opening `if`
    // so unclosed-block errors can point at the right place.
    If { skip_body_jump: usize, line: usize },
    // Open `else`. The held index is the Jump that should skip past the
    // else body once we see `}`. `line` is the `} else {` line.
    Else { skip_else_jump: usize, line: usize },
    // Open `while`. `exit_jump` is the JumpUnless/JumpIf at the top of the
    // loop (which depends on whether the user wrote `while cond` or
    // `while not cond`); `loop_start` is the index to jump back to from
    // the bottom. `pending_breaks` collects indices of break-jumps that
    // need their target patched to the loop's end when `}` closes it.
    Loop {
        exit_jump: usize,
        loop_start: usize,
        line: usize,
        pending_breaks: Vec<usize>,
    },
}

impl BlockFrame {
    fn opened_at(&self) -> usize {
        match self {
            BlockFrame::If { line, .. }
            | BlockFrame::Else { line, .. }
            | BlockFrame::Loop { line, .. } => *line,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            BlockFrame::If { .. } => "if",
            BlockFrame::Else { .. } => "else",
            BlockFrame::Loop { .. } => "while",
        }
    }
}

fn parse_program(src: &str) -> Result<Vec<Action>, String> {
    let mut out: Vec<Action> = Vec::new();
    let mut frames: Vec<BlockFrame> = Vec::new();

    for (i, line) in src.lines().enumerate() {
        let code = line.split('#').next().unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        // Normalize paren/comma syntax to whitespace so navigate_to(closest,
        // energy) and navigate_to closest energy both parse the same way.
        // Braces are kept as standalone tokens — pad with spaces so they
        // tokenize cleanly even when adjacent to other words (e.g. `cond){`).
        let normalized = code
            .replace(['(', ')', ','], " ")
            .replace('{', " { ")
            .replace('}', " } ")
            .to_ascii_lowercase();
        let words: Vec<&str> = normalized.split_whitespace().collect();

        // Block control: `if cond {`, `} else {`, `while cond {`, `until cond {`,
        // `}`. These don't push an action of their own; they push/pop frames
        // and emit synthetic jumps.
        match words.as_slice() {
            ["if", cond_tokens @ .., "{"] => {
                let expr = parse_cond_expr(cond_tokens).map_err(|e| {
                    format!("line {}: {}", i + 1, e)
                })?;
                let idx = out.len();
                out.push(cond_jump(expr));
                frames.push(BlockFrame::If {
                    skip_body_jump: idx,
                    line: i + 1,
                });
                continue;
            }
            ["while", cond_tokens @ .., "{"] => {
                let expr = parse_cond_expr(cond_tokens).map_err(|e| {
                    format!("line {}: {}", i + 1, e)
                })?;
                // Positive cond → JumpUnless (exit when false, i.e. loop
                // while true). Negated `not` cond → JumpIf (exit when true,
                // i.e. loop until cond becomes true).
                let loop_start = out.len();
                let exit_jump = out.len();
                out.push(cond_jump(expr));
                frames.push(BlockFrame::Loop {
                    exit_jump,
                    loop_start,
                    line: i + 1,
                    pending_breaks: Vec::new(),
                });
                continue;
            }
            ["break"] => {
                // Find the innermost enclosing loop and queue a placeholder
                // jump to be patched when the loop closes. If/Else frames
                // are skipped so `break` inside nested `if`s still targets
                // the surrounding loop.
                let pending_breaks = frames.iter_mut().rev().find_map(|f| match f {
                    BlockFrame::Loop { pending_breaks, .. } => Some(pending_breaks),
                    _ => None,
                });
                let pending_breaks = pending_breaks.ok_or_else(|| {
                    format!("line {}: `break` outside a loop", i + 1)
                })?;
                let idx = out.len();
                out.push(Action::Jump(usize::MAX));
                pending_breaks.push(idx);
                continue;
            }
            ["continue"] => {
                // Continue jumps to the loop's top (where the condition is
                // re-evaluated). Target is known immediately, no patching.
                let loop_start = frames.iter().rev().find_map(|f| match f {
                    BlockFrame::Loop { loop_start, .. } => Some(*loop_start),
                    _ => None,
                });
                let loop_start = loop_start.ok_or_else(|| {
                    format!("line {}: `continue` outside a loop", i + 1)
                })?;
                out.push(Action::Jump(loop_start));
                continue;
            }
            ["}", "else", "{"] => {
                let Some(BlockFrame::If { skip_body_jump, .. }) = frames.pop() else {
                    return Err(format!(
                        "line {}: `else` without a matching `if`",
                        i + 1
                    ));
                };
                // Emit Jump-past-else; patch the if's JumpUnless to land just
                // after it (= start of else body).
                let end_jump = out.len();
                out.push(Action::Jump(usize::MAX));
                let else_start = out.len();
                patch_jump_target(&mut out[skip_body_jump], else_start);
                frames.push(BlockFrame::Else {
                    skip_else_jump: end_jump,
                    line: i + 1,
                });
                continue;
            }
            ["}"] => {
                let frame = frames.pop().ok_or_else(|| {
                    format!("line {}: unmatched `}}`", i + 1)
                })?;
                // For all three block kinds, closing `}` patches one or more
                // pending jumps to land just past the block. Loops also emit
                // a back-jump first so the body iterates, then patch any
                // collected break-jumps to the same end address.
                match frame {
                    BlockFrame::If { skip_body_jump, .. } => {
                        let block_end = out.len();
                        patch_jump_target(&mut out[skip_body_jump], block_end);
                    }
                    BlockFrame::Else { skip_else_jump, .. } => {
                        let block_end = out.len();
                        patch_jump_target(&mut out[skip_else_jump], block_end);
                    }
                    BlockFrame::Loop {
                        exit_jump,
                        loop_start,
                        pending_breaks,
                        ..
                    } => {
                        out.push(Action::Jump(loop_start));
                        let block_end = out.len();
                        patch_jump_target(&mut out[exit_jump], block_end);
                        for brk in pending_breaks {
                            patch_jump_target(&mut out[brk], block_end);
                        }
                    }
                }
                continue;
            }
            _ => {}
        }

        let action = match words.as_slice() {
            ["n"] | ["north"] | ["up"] => Action::Move(Direction::North),
            ["s"] | ["south"] | ["down"] => Action::Move(Direction::South),
            ["e"] | ["east"] | ["right"] => Action::Move(Direction::East),
            ["w"] | ["west"] | ["left"] => Action::Move(Direction::West),
            ["wait"] | ["noop"] => Action::Wait,
            ["pickup"] | ["grab"] | ["take"] => Action::Pickup(None),
            ["pickup", k] | ["grab", k] | ["take", k] => {
                let kind = ResourceKind::from_token(k).ok_or_else(|| {
                    format!("line {}: unknown resource kind '{}'", i + 1, k)
                })?;
                Action::Pickup(Some(kind))
            }
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
                let target = if *t == "base" {
                    Target::Base
                } else if let Some(kind) = ResourceKind::from_token(t) {
                    Target::Resource(kind)
                } else {
                    return Err(format!("line {}: unknown nav target '{}'", i + 1, t));
                };
                Action::NavigateTo(qualifier, target)
            }
            _ => return Err(format!("line {}: unknown instruction '{}'", i + 1, code)),
        };
        out.push(action);
    }

    if let Some(frame) = frames.first() {
        return Err(format!(
            "unclosed `{}` block opened on line {}",
            frame.kind(),
            frame.opened_at()
        ));
    }
    if out.is_empty() {
        return Err("program is empty".into());
    }
    Ok(out)
}

// Set the target of a Jump or JumpUnless that was pushed with a placeholder.
// Callers guarantee `action` is one of those two variants — anything else is
// a parser bug, not a user error.
fn patch_jump_target(action: &mut Action, target: usize) {
    match action {
        Action::Jump(t) | Action::JumpIf(_, t) | Action::JumpUnless(_, t) => *t = target,
        _ => unreachable!("patch_jump_target called on non-jump action"),
    }
}

// A parsed condition expression: a positive `Condition` plus a `negated`
// flag for leading `not`. Kept as a flag rather than `Condition::Not(...)`
// so `Condition` stays Copy and the IR jump variants (JumpIf/JumpUnless)
// directly express both polarities without extra runtime negation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct CondExpr {
    condition: Condition,
    negated: bool,
}

fn parse_cond_expr(words: &[&str]) -> Result<CondExpr, String> {
    match words {
        ["not", rest @ ..] => Ok(CondExpr {
            condition: parse_condition(rest)?,
            negated: true,
        }),
        _ => Ok(CondExpr {
            condition: parse_condition(words)?,
            negated: false,
        }),
    }
}

fn parse_condition(words: &[&str]) -> Result<Condition, String> {
    match words {
        ["carrying", kind] => {
            let kind = ResourceKind::from_token(kind)
                .ok_or_else(|| format!("unknown resource kind '{}' in condition", kind))?;
            Ok(Condition::Carrying(kind))
        }
        _ => Err(format!("unknown condition: `{}`", words.join(" "))),
    }
}

// Build the loop-exit / if-skip jump that should be emitted at the top of
// a conditional block. Positive cond → JumpUnless (exit/skip when false);
// negated cond → JumpIf (exit/skip when true).
fn cond_jump(expr: CondExpr) -> Action {
    if expr.negated {
        Action::JumpIf(expr.condition, usize::MAX)
    } else {
        Action::JumpUnless(expr.condition, usize::MAX)
    }
}

fn advance_tick(mut tick: ResMut<Tick>) {
    tick.0 = tick.0.wrapping_add(1);
}

// Evaluate a branching condition against a worker's inventory. Pulled out
// as a free function so it stays pure and new condition variants don't
// muddy step_workers — match exhaustiveness will flag a missing arm here.
fn evaluate_condition(cond: Condition, inv: &Inventory) -> bool {
    match cond {
        Condition::Carrying(kind) => inv.queue.contains(&kind),
    }
}

// Bevy's Query<Data, Filter> signatures are inherently nested; a type alias
// per system body would be more noise than the inline form.
#[allow(clippy::type_complexity)]
fn step_workers(
    world: Res<World>,
    assets: Res<ResourceAssets>,
    mut commands: Commands,
    resource_q: Query<
        (Entity, &GridPos, &ResourceNode),
        (With<ResourceNode>, Without<Worker>, Without<Base>),
    >,
    mut base_q: Query<(&GridPos, &mut Base), (Without<Worker>, Without<ResourceNode>)>,
    mut workers: Query<
        (
            Entity,
            &mut GridPos,
            &mut Program,
            &mut Facing,
            &mut Inventory,
            &mut NavState,
            Option<&mut BumpCount>,
        ),
        (With<Worker>, Without<ResourceNode>, Without<Base>),
    >,
) {
    // Tile actions taken so far this tick. The first worker to attempt a
    // (tile, action) wins and inserts the key; later workers see the lock
    // and skip. Today this stops two workers from both despawning the same
    // energy entity (and double-incrementing inventory) when they share a
    // tile — Bevy's commands are queued, so energy_q is otherwise stale.
    let mut tile_locks: HashSet<(GridPos, TileAction)> = HashSet::new();

    // Snapshot of every worker's outstanding resource-tile reservation. Each
    // worker planning a fresh navigate_to(resource) excludes tiles already
    // in this set, then inserts its own choice — so later workers in the
    // same tick see earlier workers' claims. Tile-only (not per-kind):
    // workers fan out even across kinds at the cost of slight over-restriction
    // on the rare grass+wood overlap.
    let mut resource_claims: HashSet<GridPos> = workers
        .iter()
        .filter_map(|(_, _, _, _, _, nav, _)| nav.reserved_tile)
        .collect();

    // Snapshot of currently occupied worker tiles. Each successful move
    // updates the set so later workers in this tick see fresh occupancy.
    // Stepping into an occupied tile is a no-op ("bump"); the worker still
    // turns to face the obstacle but doesn't move.
    let mut occupied: HashSet<GridPos> = workers
        .iter()
        .map(|(_, pos, _, _, _, _, _)| *pos)
        .collect();

    for (entity, mut pos, mut prog, mut facing, mut inv, mut nav, mut bumps) in &mut workers {
        if prog.instructions.is_empty() {
            continue;
        }
        let action = prog.instructions[prog.pc];
        let mut advance_pc = true;
        // Per-tick movement bookkeeping. `moved` resets the bump counter
        // (the worker isn't stuck); `blocked` increments it (intent
        // foiled). Wait/Pickup/Drop set neither and leave the counter alone.
        let mut moved = false;
        let mut blocked = false;
        match action {
            Action::Move(dir) => {
                if step_in_direction(&world, &mut pos, &mut facing, dir, &mut occupied) {
                    moved = true;
                } else {
                    blocked = true;
                }
            }
            Action::Pickup(filter) => {
                // insert() returns true on first acquire, false if another
                // worker already claimed this tile-pickup this tick. Without
                // this, two workers on the same tile would both despawn the
                // same node (commands are queued, so resource_q is stale
                // until next tick).
                if tile_locks.insert((*pos, TileAction::Pickup)) {
                    // Grab one resource of the requested kind (or any kind
                    // if filter is None). If the queue is full, the evicted
                    // oldest item is dropped back onto the worker's tile as
                    // a fresh ResourceNode so it remains collectible.
                    let hit = resource_q.iter().find(|(_, rpos, res)| {
                        **rpos == *pos && filter.is_none_or(|want| res.kind == want)
                    });
                    if let Some((ent, _, res)) = hit {
                        commands.entity(ent).despawn();
                        if let Some(evicted) = inv.push(res.kind) {
                            spawn_resource_node(&mut commands, &assets, evicted, *pos);
                        }
                    }
                }
                // Clear the reservation only if it was for this tile —
                // we're done with it. We deliberately do NOT remove it
                // from `resource_claims` this tick: despawns are queued,
                // so resource_q still lists the just-consumed node, and
                // dropping the claim now would let other workers later
                // in this same tick re-target the stale tile. The claim
                // falls out naturally next tick when `resource_claims` is
                // rebuilt from (now-cleared) NavStates.
                if nav.reserved_tile == Some(*pos) {
                    nav.reserved_tile = None;
                }
            }
            Action::Drop => {
                if !inv.is_empty() {
                    for (bpos, mut base) in &mut base_q {
                        if *bpos == *pos {
                            inv.drain_into(&mut base.stored);
                            break;
                        }
                    }
                }
            }
            Action::Wait => {}
            Action::NavigateTo(_qualifier, target) => {
                // Closure shared by initial plan and bump re-plan. `claims`
                // and `blocked` are passed by ref each call so the closure
                // doesn't capture `resource_claims` (which we mutate around
                // these calls).
                let make_plan = |start: GridPos,
                                 claims: &HashSet<GridPos>,
                                 blocked: &HashSet<GridPos>| {
                    // HashSet (not Vec) so `is_target` in BFS is O(1); the
                    // dev-mode dense energy cluster + frequent re-plans
                    // make this contention-sensitive.
                    let targets: HashSet<GridPos> = match target {
                        Target::Resource(kind) => resource_q
                            .iter()
                            .filter(|(_, _, r)| r.kind == kind)
                            .map(|(_, p, _)| *p)
                            .filter(|p| !claims.contains(p))
                            .collect(),
                        Target::Base => base_q.iter().map(|(p, _)| *p).collect(),
                    };
                    find_path(&world, start, |p| targets.contains(&p), blocked)
                        .unwrap_or_default()
                };

                // Lazy plan: only recompute when we have no cached steps.
                // The initial plan ignores other workers — they're transient
                // obstacles, no point routing around them eagerly.
                if nav.plan.is_empty() {
                    if let Some(prev) = nav.reserved_tile.take() {
                        resource_claims.remove(&prev);
                    }
                    let plan = make_plan(*pos, &resource_claims, &HashSet::new());
                    if let (Target::Resource(_), false) = (target, plan.is_empty()) {
                        let dest = path_destination(*pos, &plan);
                        nav.reserved_tile = Some(dest);
                        resource_claims.insert(dest);
                    }
                    nav.plan = plan;
                }
                // Attempt one step. On success, drain the planned direction.
                // On a bump, throw the cached plan out and re-plan against
                // the actual occupancy — and if even that fails, sidestep.
                if let Some(&dir) = nav.plan.front() {
                    if step_in_direction(
                        &world,
                        &mut pos,
                        &mut facing,
                        dir,
                        &mut occupied,
                    ) {
                        nav.plan.pop_front();
                        moved = true;
                    } else {
                        blocked = true;
                        nav.plan.clear();
                        if let Some(prev) = nav.reserved_tile.take() {
                            resource_claims.remove(&prev);
                        }
                        // Re-plan with every other worker treated as a wall.
                        // We exclude ourselves so we can leave our own tile.
                        let mut obstacles = occupied.clone();
                        obstacles.remove(&*pos);
                        let alt = make_plan(*pos, &resource_claims, &obstacles);
                        if !alt.is_empty() {
                            if matches!(target, Target::Resource(_)) {
                                let dest = path_destination(*pos, &alt);
                                nav.reserved_tile = Some(dest);
                                resource_claims.insert(dest);
                            }
                            nav.plan = alt;
                            // We don't take the first step inline — wait a
                            // tick. Next tick the plan executes from the
                            // current position.
                        } else {
                            // No detour exists (corridor, dead end, etc.).
                            // Sidestep into any open neighbor other than the
                            // bumped direction so the standoff breaks. We
                            // iterate CARDINAL_DIRS in its fixed [N, S, E, W]
                            // order, skipping the bumped direction — the
                            // first cardinal that's free wins.
                            for sidestep in CARDINAL_DIRS {
                                if sidestep == dir {
                                    continue;
                                }
                                if step_in_direction(
                                    &world,
                                    &mut pos,
                                    &mut facing,
                                    sidestep,
                                    &mut occupied,
                                ) {
                                    moved = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                // Advance pc when there's genuinely nothing left to do — the
                // plan drained via successful steps, or no path could ever be
                // found. A bump leaves us still trying to navigate.
                advance_pc = !blocked && nav.plan.is_empty();
            }
            // Branches are zero-tick: they move pc and the next tick executes
            // the new instruction. Setting advance_pc = false suppresses the
            // automatic +1 since we've already chosen the next pc directly.
            Action::Jump(target) => {
                prog.pc = target % prog.instructions.len();
                advance_pc = false;
            }
            Action::JumpIf(cond, target) => {
                if evaluate_condition(cond, &inv) {
                    prog.pc = target % prog.instructions.len();
                    advance_pc = false;
                }
            }
            Action::JumpUnless(cond, target) => {
                if !evaluate_condition(cond, &inv) {
                    prog.pc = target % prog.instructions.len();
                    advance_pc = false;
                }
            }
        }
        if advance_pc {
            prog.pc = (prog.pc + 1) % prog.instructions.len();
        }

        // Update the bump counter. Any movement (planned or sidestep) resets
        // it — productive workers never die. A blocked intent without
        // movement increments it; at MAX_BUMPS the worker spills its
        // inventory and despawns.
        if let Some(bumps) = bumps.as_mut() {
            if moved {
                bumps.0 = 0;
            } else if blocked {
                bumps.0 += 1;
                if bumps.0 >= MAX_BUMPS {
                    // Release any outstanding resource reservation so later
                    // workers in this same tick can target the tile.
                    if let Some(prev) = nav.reserved_tile.take() {
                        resource_claims.remove(&prev);
                    }
                    // Spill every carried resource back onto the worker's
                    // tile as a fresh ResourceNode so others can collect.
                    while let Some(kind) = inv.queue.pop_front() {
                        spawn_resource_node(&mut commands, &assets, kind, *pos);
                    }
                    commands.entity(entity).despawn();
                }
            }
        }
    }
}

// Walks a plan from `start` and returns the cell it terminates at. Used to
// derive the destination of a BFS path so it can be reserved without changing
// find_path's signature.
fn path_destination(start: GridPos, plan: &VecDeque<Direction>) -> GridPos {
    let mut p = start;
    for d in plan {
        let (dx, dy) = d.delta();
        p.x += dx;
        p.y += dy;
    }
    p
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

// Mirror each worker's Inventory.queue onto its CarrySlot children: visible
// + colored when occupied, hidden when empty. Runs every frame so a drop or
// pickup is reflected on the next render with no extra wiring.
fn update_carry_display(
    workers: Query<(&Inventory, &Children), With<Worker>>,
    mut slots: Query<(
        &CarrySlot,
        &mut Visibility,
        &mut MeshMaterial3d<StandardMaterial>,
    )>,
    assets: Res<ResourceAssets>,
) {
    for (inv, children) in &workers {
        for &child in children {
            let Ok((slot, mut vis, mut mat)) = slots.get_mut(child) else {
                continue;
            };
            match inv.queue.get(slot.0) {
                Some(&kind) => {
                    *vis = Visibility::Visible;
                    mat.0 = assets.material_for(kind);
                }
                None => *vis = Visibility::Hidden,
            }
        }
    }
}

// Visual flourish — slow Y rotation so energy gems catch the eye. Grass and
// wood don't spin; they're meant to read as static groundcover.
fn spin_energy(time: Res<Time>, mut q: Query<(&ResourceNode, &mut Transform)>) {
    let dt = time.delta_secs();
    for (res, mut tf) in &mut q {
        if matches!(res.kind, ResourceKind::Energy) {
            tf.rotate_y(dt * 1.5);
        }
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

fn show_resource_counts(ui: &mut egui::Ui, header: &str, counts: &[u32; ResourceKind::COUNT]) {
    ui.label(header);
    for kind in ResourceKind::ALL {
        ui.monospace(format!("  {}: {}", kind.label(), counts[kind as usize]));
    }
}

// --- DSL syntax highlighter ----------------------------------------------

// Token categories the editor colorizes. Punctuation/whitespace/unknown all
// render in the default color; the three "named" categories — keyword
// (control flow), verb (action), identifier (built-in argument) — each get
// a distinct color so a quick visual scan tells you what each line does.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TokenKind {
    Keyword,
    Verb,
    Identifier,
    Comment,
    Punctuation,
    Whitespace,
    Unknown,
}

// Highlighter palette tuned for the dark side-panel background. RGB values
// borrowed from Atom's One Dark theme — they read well against #0d0e17 and
// keep the keyword/verb/identifier groups visually distinct at small sizes.
const KEYWORD_COLOR: egui::Color32 = egui::Color32::from_rgb(198, 120, 221); // purple
const VERB_COLOR: egui::Color32 = egui::Color32::from_rgb(97, 175, 239); // blue
const IDENTIFIER_COLOR: egui::Color32 = egui::Color32::from_rgb(229, 192, 123); // gold
const COMMENT_COLOR: egui::Color32 = egui::Color32::from_rgb(120, 120, 120); // gray

// Hardcoded vocabulary lists. These mirror the parser's match arms — when a
// new keyword or verb is added there, it must also be added here. A shared
// source of truth would be cleaner but the two consumers want different
// shapes (parser does slice-pattern matching, highlighter classifies one
// identifier at a time), so the duplication is accepted for now.
const KEYWORDS: &[&str] = &["if", "else", "while", "break", "continue", "not"];
const VERBS: &[&str] = &[
    "pickup",
    "grab",
    "take",
    "drop",
    "deposit",
    "deliver",
    "wait",
    "noop",
    "navigate_to",
    "goto",
    "nav",
];
const ARG_IDENTIFIERS: &[&str] = &[
    "energy", "grass", "wood", "base", "carrying", "closest", "n", "s", "e", "w", "north", "south",
    "east", "west", "up", "down", "left", "right",
];

fn classify_identifier(ident: &str) -> TokenKind {
    let in_list = |list: &[&str]| list.iter().any(|w| ident.eq_ignore_ascii_case(w));
    if in_list(KEYWORDS) {
        TokenKind::Keyword
    } else if in_list(VERBS) {
        TokenKind::Verb
    } else if in_list(ARG_IDENTIFIERS) {
        TokenKind::Identifier
    } else {
        TokenKind::Unknown
    }
}

// Lossless tokenization: every byte of `src` ends up in exactly one returned
// slice, in order. The editor's layouter relies on this so the rendered
// galley has the same character offsets as the source.
fn tokenize(src: &str) -> Vec<(&str, TokenKind)> {
    let mut out: Vec<(&str, TokenKind)> = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        // Comment: `#` to end of line (exclusive of the trailing newline).
        if b == b'#' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'\n' {
                j += 1;
            }
            out.push((&src[i..j], TokenKind::Comment));
            i = j;
            continue;
        }
        // Whitespace run (including newlines).
        if b.is_ascii_whitespace() {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            out.push((&src[i..j], TokenKind::Whitespace));
            i = j;
            continue;
        }
        // Identifier: ASCII alnum + underscore. Covers all DSL names.
        if b.is_ascii_alphanumeric() || b == b'_' {
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            let text = &src[i..j];
            out.push((text, classify_identifier(text)));
            i = j;
            continue;
        }
        // Single-char punctuation (braces, parens, commas).
        if matches!(b, b'{' | b'}' | b'(' | b')' | b',') {
            out.push((&src[i..i + 1], TokenKind::Punctuation));
            i += 1;
            continue;
        }
        // Anything else (a stray symbol like ':', or non-ASCII pasted into a
        // comment) — emit as Unknown using a char-aware length so we never
        // slice across a UTF-8 codepoint boundary.
        let ch_len = src[i..].chars().next().map(char::len_utf8).unwrap_or(1);
        out.push((&src[i..i + ch_len], TokenKind::Unknown));
        i += ch_len;
    }
    out
}

fn color_for(kind: TokenKind, default: egui::Color32) -> egui::Color32 {
    match kind {
        TokenKind::Keyword => KEYWORD_COLOR,
        TokenKind::Verb => VERB_COLOR,
        TokenKind::Identifier => IDENTIFIER_COLOR,
        TokenKind::Comment => COMMENT_COLOR,
        TokenKind::Punctuation | TokenKind::Whitespace | TokenKind::Unknown => default,
    }
}

fn highlight_layout(src: &str, style: &egui::Style) -> egui::text::LayoutJob {
    let font_id = egui::TextStyle::Monospace.resolve(style);
    let default_color = style.visuals.text_color();
    let mut job = egui::text::LayoutJob::default();
    for (text, kind) in tokenize(src) {
        job.append(
            text,
            0.0,
            egui::TextFormat {
                font_id: font_id.clone(),
                color: color_for(kind, default_color),
                italics: matches!(kind, TokenKind::Comment),
                ..Default::default()
            },
        );
    }
    job
}

fn editor_ui(
    mut contexts: EguiContexts,
    mut editor: ResMut<Editor>,
    mut q: Query<(&mut Program, &Inventory, &mut NavState), With<Worker>>,
    bases: Query<&Base>,
    tick: Res<Tick>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::SidePanel::left("editor")
        .default_width(260.0)
        .show(ctx, |ui| {
            ui.heading("Worker Program");
            ui.label("One instruction per line. Tokens:");
            ui.monospace("N S E W Wait Drop");
            ui.monospace("pickup [energy|grass|wood]");
            ui.monospace("navigate_to(closest, energy|grass|wood|base)");
            ui.monospace("if [not] carrying(kind) { ... } else { ... }");
            ui.monospace("while [not] carrying(kind) { ... }");
            ui.monospace("break | continue (inside a loop)");
            ui.label("'#' starts a comment.");
            ui.separator();

            let mut layouter = |ui: &egui::Ui, text: &dyn egui::TextBuffer, wrap_width: f32| {
                let mut job = highlight_layout(text.as_str(), ui.style());
                job.wrap.max_width = wrap_width;
                ui.fonts_mut(|f| f.layout_job(job))
            };
            ui.add(
                egui::TextEdit::multiline(&mut editor.source)
                    .desired_rows(16)
                    .desired_width(f32::INFINITY)
                    .font(egui::TextStyle::Monospace)
                    .layouter(&mut layouter),
            );

            if ui.button("Compile & Load").clicked() {
                match parse_program(&editor.source) {
                    Ok(instrs) => {
                        let n = instrs.len();
                        for (mut prog, _, mut nav) in &mut q {
                            prog.instructions = instrs.clone();
                            prog.pc = 0;
                            // Drop any cached plan + reservation so the new
                            // program doesn't continue walking toward a tile
                            // the old script was targeting.
                            *nav = NavState::default();
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
            // All workers run the same script, so pc/program length are the
            // same across them; sample the first. Energy is per-worker so we
            // sum it.
            ui.label(format!("workers: {}", q.iter().count()));
            if let Some((prog, _, _)) = q.iter().next() {
                ui.label(format!(
                    "pc: {} / {}",
                    prog.pc,
                    prog.instructions.len()
                ));
            }
            // Sum each kind across all worker inventories. Per-worker queue
            // contents aren't shown — that'd need N panels for N workers.
            let mut carried = [0u32; ResourceKind::COUNT];
            for (_, inv, _) in &q {
                for &k in &inv.queue {
                    carried[k as usize] += 1;
                }
            }
            show_resource_counts(ui, "carried (all workers):", &carried);
            if let Ok(base) = bases.single() {
                show_resource_counts(ui, "delivered:", &base.stored);
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
        let p = find_path(
            &w,
            GridPos { x: 5, y: 5 },
            |p| p == GridPos { x: 5, y: 5 },
            &HashSet::new(),
        );
        assert_eq!(p, Some(VecDeque::new()));
    }

    #[test]
    fn find_path_walks_a_straight_line_on_open_grass() {
        let w = flat_grass_world();
        let p = find_path(
            &w,
            GridPos { x: 5, y: 5 },
            |p| p == GridPos { x: 8, y: 5 },
            &HashSet::new(),
        )
        .unwrap();
        assert_eq!(p, VecDeque::from(vec![Direction::East; 3]));
    }

    #[test]
    fn find_path_detours_around_a_wall() {
        let mut w = flat_grass_world();
        // Single-cell wall directly east of (5,5); detour must go north or south.
        place(&mut w, 6, 5, Terrain::Wall);
        let p = find_path(
            &w,
            GridPos { x: 5, y: 5 },
            |p| p == GridPos { x: 7, y: 5 },
            &HashSet::new(),
        )
        .unwrap();
        // Manhattan distance is 2; the detour adds 2 extra steps.
        assert_eq!(p.len(), 4);
    }

    #[test]
    fn find_path_returns_none_when_target_is_walled_in() {
        let mut w = flat_grass_world();
        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            place(&mut w, 7 + dx, 5 + dy, Terrain::Wall);
        }
        let p = find_path(
            &w,
            GridPos { x: 5, y: 5 },
            |p| p == GridPos { x: 7, y: 5 },
            &HashSet::new(),
        );
        assert_eq!(p, None);
    }

    #[test]
    fn find_path_returns_none_when_no_cell_matches() {
        let w = flat_grass_world();
        let p = find_path(&w, GridPos { x: 5, y: 5 }, |_| false, &HashSet::new());
        assert_eq!(p, None);
    }

    #[test]
    fn find_path_picks_the_closest_of_multiple_targets() {
        let w = flat_grass_world();
        let near = GridPos { x: 5, y: 6 };
        let far = GridPos { x: 5, y: 12 };
        let p = find_path(
            &w,
            GridPos { x: 5, y: 5 },
            |p| p == near || p == far,
            &HashSet::new(),
        )
        .unwrap();
        // BFS expands by distance, so the 1-step target wins over the 7-step one.
        // (+1 in grid Y is South after the N/S swap.)
        assert_eq!(p, VecDeque::from(vec![Direction::South]));
    }

    // --- worker_start_positions ---------------------------------------------

    #[test]
    fn worker_start_positions_returns_start_first() {
        let w = flat_grass_world();
        let starts =
            worker_start_positions(&w, GridPos { x: 5, y: 5 }, 1, &HashSet::new());
        assert_eq!(starts, vec![GridPos { x: 5, y: 5 }]);
    }

    #[test]
    fn worker_start_positions_returns_n_distinct_cells() {
        let w = flat_grass_world();
        let starts =
            worker_start_positions(&w, GridPos { x: 5, y: 5 }, 4, &HashSet::new());
        assert_eq!(starts.len(), 4);
        // All distinct.
        for i in 0..starts.len() {
            for j in (i + 1)..starts.len() {
                assert_ne!(starts[i], starts[j], "duplicate start at {i} and {j}");
            }
        }
        // First slot is always the requested start cell.
        assert_eq!(starts[0], GridPos { x: 5, y: 5 });
    }

    #[test]
    fn worker_start_positions_skips_impassable_neighbors() {
        let mut w = flat_grass_world();
        // East neighbor wall, west neighbor water. North/south stay passable,
        // so 3 cells (start + N + S) must come back — and neither blocked
        // neighbor cell should appear.
        place(&mut w, 6, 5, Terrain::Wall);
        place(&mut w, 4, 5, Terrain::Water);
        let starts =
            worker_start_positions(&w, GridPos { x: 5, y: 5 }, 3, &HashSet::new());
        assert_eq!(starts.len(), 3);
        for p in &starts {
            let t = w.get(p.x, p.y).unwrap();
            assert!(t.passable(), "start {p:?} landed on impassable {t:?}");
            assert_ne!(*p, GridPos { x: 6, y: 5 });
            assert_ne!(*p, GridPos { x: 4, y: 5 });
        }
    }

    #[test]
    fn worker_start_positions_caps_when_not_enough_passable_cells() {
        let mut w = flat_grass_world();
        // Box the start cell in — only START itself is passable.
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            place(&mut w, 5 + dx, 5 + dy, Terrain::Wall);
        }
        let starts =
            worker_start_positions(&w, GridPos { x: 5, y: 5 }, 8, &HashSet::new());
        assert_eq!(starts, vec![GridPos { x: 5, y: 5 }]);
    }

    #[test]
    fn worker_start_positions_skips_excluded_tiles_but_expands_through_them() {
        let mut w = flat_grass_world();
        // Wall off S/E/W of start so the only escape is north. The north
        // neighbor (5,4) is grass but excluded (simulating an energy node
        // sitting on it). Without expanding through excluded cells we'd
        // get only (5,5); with it, (5,3) and beyond stay reachable.
        place(&mut w, 5, 6, Terrain::Wall);
        place(&mut w, 6, 5, Terrain::Wall);
        place(&mut w, 4, 5, Terrain::Wall);
        let mut excluded = HashSet::new();
        excluded.insert(GridPos { x: 5, y: 4 });
        let starts = worker_start_positions(&w, GridPos { x: 5, y: 5 }, 3, &excluded);
        assert_eq!(starts.len(), 3, "starts={starts:?}");
        assert_eq!(starts[0], GridPos { x: 5, y: 5 });
        assert!(
            !starts.contains(&GridPos { x: 5, y: 4 }),
            "excluded tile should not appear in starts"
        );
        // The cell two north of start is only reachable by expanding through
        // the excluded tile.
        assert!(
            starts.contains(&GridPos { x: 5, y: 3 }),
            "expected (5,3) reachable via excluded (5,4); starts={starts:?}"
        );
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

    // --- step_workers reservation + pickup ----------------------------------

    // step_workers reads Res<ResourceAssets> for the eviction respawn path.
    // Tests don't exercise that path (queue stays under capacity) but Bevy
    // still requires the resource to exist, so insert dummy handles.
    fn dummy_resource_assets() -> ResourceAssets {
        ResourceAssets {
            visuals: ResourceKind::ALL.map(|_| ResourceVisual {
                mesh: Handle::default(),
                material: Handle::default(),
                y: 0.0,
                rotation: Quat::IDENTITY,
            }),
        }
    }

    fn count_kind(inv: &Inventory, kind: ResourceKind) -> u32 {
        inv.queue.iter().filter(|&&k| k == kind).count() as u32
    }

    // Spawn a stationary worker that runs Wait forever — useful as an
    // obstacle in occupancy tests.
    fn spawn_blocker(world: &mut bevy::ecs::world::World, pos: GridPos) {
        world.spawn((
            Worker,
            pos,
            Inventory::default(),
            NavState::default(),
            Facing {
                prev_yaw: 0.0,
                current_yaw: 0.0,
            },
            Program {
                instructions: vec![Action::Wait],
                pc: 0,
            },
        ));
    }

    fn spawn_navigator(world: &mut bevy::ecs::world::World, pos: GridPos) -> Entity {
        world
            .spawn((
                Worker,
                pos,
                Inventory::default(),
                NavState::default(),
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                Program {
                    instructions: vec![Action::NavigateTo(
                        NavQualifier::Closest,
                        Target::Resource(ResourceKind::Energy),
                    )],
                    pc: 0,
                },
            ))
            .id()
    }

    #[test]
    fn navigate_to_energy_reserves_target_so_workers_pick_different_tiles() {
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(World {
            tiles: vec![Terrain::Grass; (GRID_W * GRID_H) as usize],
        });
        world.insert_resource(dummy_resource_assets());

        // Two energy tiles five and six steps north of the workers — far
        // enough that one-tick paths don't drain to empty, so reserved_tile
        // is still set when we inspect it.
        world.spawn((ResourceNode { kind: ResourceKind::Energy }, GridPos { x: 5, y: 10 }));
        world.spawn((ResourceNode { kind: ResourceKind::Energy }, GridPos { x: 5, y: 11 }));

        let w1 = spawn_navigator(&mut world, GridPos { x: 5, y: 5 });
        let w2 = spawn_navigator(&mut world, GridPos { x: 5, y: 5 });

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        let t1 = world.get::<NavState>(w1).unwrap().reserved_tile;
        let t2 = world.get::<NavState>(w2).unwrap().reserved_tile;
        assert!(t1.is_some() && t2.is_some(), "both workers should hold a claim");
        assert_ne!(t1, t2, "workers must reserve different energy tiles");
        let mut targets = [t1.unwrap(), t2.unwrap()];
        targets.sort_by_key(|p| (p.y, p.x));
        assert_eq!(
            targets,
            [GridPos { x: 5, y: 10 }, GridPos { x: 5, y: 11 }]
        );
    }

    #[test]
    fn pickup_does_not_unclaim_tile_within_same_tick() {
        // Two workers, two energy tiles. Worker A is already on its
        // reserved tile and Pickups this tick. Worker B is far away and
        // plans navigate_to(energy) this tick. Because A's despawn is
        // deferred, resource_q still lists A's tile during B's planning —
        // so the reservation set must keep A's tile claimed until next
        // tick, otherwise B would re-target the stale tile.
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(World {
            tiles: vec![Terrain::Grass; (GRID_W * GRID_H) as usize],
        });
        world.insert_resource(dummy_resource_assets());
        let a_tile = GridPos { x: 5, y: 5 };
        let other = GridPos { x: 10, y: 10 };
        world.spawn((ResourceNode { kind: ResourceKind::Energy }, a_tile));
        world.spawn((ResourceNode { kind: ResourceKind::Energy }, other));

        // Worker A — about to pick up at its reserved tile.
        world.spawn((
            Worker,
            a_tile,
            Inventory::default(),
            NavState {
                plan: VecDeque::new(),
                reserved_tile: Some(a_tile),
            },
            Facing {
                prev_yaw: 0.0,
                current_yaw: 0.0,
            },
            Program {
                instructions: vec![Action::Pickup(None)],
                pc: 0,
            },
        ));
        // Worker B — about to plan a fresh navigate_to(energy).
        let b = world
            .spawn((
                Worker,
                GridPos { x: 0, y: 0 },
                Inventory::default(),
                NavState::default(),
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                Program {
                    instructions: vec![Action::NavigateTo(
                        NavQualifier::Closest,
                        Target::Resource(ResourceKind::Energy),
                    )],
                    pc: 0,
                },
            ))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        let target_b = world.get::<NavState>(b).unwrap().reserved_tile;
        assert_ne!(
            target_b,
            Some(a_tile),
            "B targeted the tile A just picked up — claim was released too eagerly"
        );
        assert_eq!(
            target_b,
            Some(other),
            "B should have targeted the only other available tile"
        );
    }

    #[test]
    fn pickup_releases_reservation() {
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(World {
            tiles: vec![Terrain::Grass; (GRID_W * GRID_H) as usize],
        });
        world.insert_resource(dummy_resource_assets());
        world.spawn((ResourceNode { kind: ResourceKind::Energy }, GridPos { x: 5, y: 5 }));

        // Worker is standing on an energy tile with a stale reservation,
        // running Pickup. After the tick, the reservation must be cleared
        // so other workers' next plan sees the tile as free.
        let w = world
            .spawn((
                Worker,
                GridPos { x: 5, y: 5 },
                Inventory::default(),
                NavState {
                    plan: VecDeque::new(),
                    reserved_tile: Some(GridPos { x: 5, y: 5 }),
                },
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                Program {
                    instructions: vec![Action::Pickup(None)],
                    pc: 0,
                },
            ))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        assert_eq!(
            world.get::<NavState>(w).unwrap().reserved_tile,
            None,
            "Pickup must release the reservation"
        );
        assert_eq!(
            count_kind(world.get::<Inventory>(w).unwrap(), ResourceKind::Energy),
            1,
        );
    }

    // --- step_in_direction occupancy ----------------------------------------

    #[test]
    fn step_in_direction_blocked_by_occupied_destination() {
        let w = flat_grass_world();
        let mut pos = GridPos { x: 5, y: 5 };
        let mut facing = Facing {
            prev_yaw: 0.0,
            current_yaw: 0.0,
        };
        let mut occupied: HashSet<GridPos> =
            [pos, GridPos { x: 5, y: 4 }].into_iter().collect();

        let moved =
            step_in_direction(&w, &mut pos, &mut facing, Direction::North, &mut occupied);

        // Position unchanged, but facing still updates so the worker visibly
        // turns toward the blocker (mirrors the wall-block behavior).
        assert!(!moved, "blocked step should report no movement");
        assert_eq!(pos, GridPos { x: 5, y: 5 });
        assert_eq!(facing.current_yaw, Direction::North.yaw());
        assert!(occupied.contains(&GridPos { x: 5, y: 5 }));
        assert!(occupied.contains(&GridPos { x: 5, y: 4 }));
    }

    #[test]
    fn step_in_direction_moves_into_free_tile_and_updates_set() {
        let w = flat_grass_world();
        let mut pos = GridPos { x: 5, y: 5 };
        let mut facing = Facing {
            prev_yaw: 0.0,
            current_yaw: 0.0,
        };
        let mut occupied: HashSet<GridPos> = [pos].into_iter().collect();

        let moved =
            step_in_direction(&w, &mut pos, &mut facing, Direction::North, &mut occupied);

        assert!(moved, "successful step should report movement");
        assert_eq!(pos, GridPos { x: 5, y: 4 });
        assert!(!occupied.contains(&GridPos { x: 5, y: 5 }), "old tile freed");
        assert!(occupied.contains(&GridPos { x: 5, y: 4 }), "new tile claimed");
    }

    #[test]
    fn bump_during_navigate_to_replans_around_blocker() {
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(World {
            tiles: vec![Terrain::Grass; (GRID_W * GRID_H) as usize],
        });
        world.insert_resource(dummy_resource_assets());
        // Energy at (5,3): direct path from A is straight north, but B
        // parks at (5,4) blocking that path. After one tick A should still
        // be at (5,5) (we don't take an inline step), still on navigate_to,
        // and the cached plan should now detour around B (no longer start
        // with North).
        world.spawn((
            ResourceNode {
                kind: ResourceKind::Energy,
            },
            GridPos { x: 5, y: 3 },
        ));
        let a = world
            .spawn((
                Worker,
                GridPos { x: 5, y: 5 },
                Inventory::default(),
                NavState {
                    plan: VecDeque::from(vec![Direction::North, Direction::North]),
                    reserved_tile: None,
                },
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                Program {
                    instructions: vec![Action::NavigateTo(
                        NavQualifier::Closest,
                        Target::Resource(ResourceKind::Energy),
                    )],
                    pc: 0,
                },
            ))
            .id();
        spawn_blocker(&mut world, GridPos { x: 5, y: 4 });

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        assert_eq!(*world.get::<GridPos>(a).unwrap(), GridPos { x: 5, y: 5 });
        let plan = &world.get::<NavState>(a).unwrap().plan;
        assert!(
            !plan.is_empty(),
            "expected a fresh detour plan, got empty; should re-plan around B"
        );
        assert_ne!(
            plan[0],
            Direction::North,
            "re-plan must route around the blocker, not back through them; plan={plan:?}"
        );
        assert_eq!(
            world.get::<Program>(a).unwrap().pc,
            0,
            "pc should not advance off navigate_to while we haven't arrived"
        );
    }

    #[test]
    fn bump_with_no_alt_path_sidesteps_into_open_neighbor() {
        // Tight corridor: only path from (5,5) to energy at (5,3) is
        // through (5,4) — boxed in by walls on the other sides of both
        // start and target. (4,5) is left as grass to provide a single
        // valid sidestep.
        let mut tiles = vec![Terrain::Grass; (GRID_W * GRID_H) as usize];
        for (wx, wy) in [
            (4, 4),
            (6, 4),
            (4, 3),
            (6, 3),
            (5, 2),
            (6, 5),
            (5, 6),
        ] {
            tiles[World::idx(wx, wy)] = Terrain::Wall;
        }
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(World { tiles });
        world.insert_resource(dummy_resource_assets());
        world.spawn((
            ResourceNode {
                kind: ResourceKind::Energy,
            },
            GridPos { x: 5, y: 3 },
        ));

        let a = world
            .spawn((
                Worker,
                GridPos { x: 5, y: 5 },
                Inventory::default(),
                NavState::default(),
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                Program {
                    instructions: vec![Action::NavigateTo(
                        NavQualifier::Closest,
                        Target::Resource(ResourceKind::Energy),
                    )],
                    pc: 0,
                },
            ))
            .id();
        spawn_blocker(&mut world, GridPos { x: 5, y: 4 });

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        // Initial plan goes N, A bumps B, re-plan with B as obstacle finds
        // no alternative (all other neighbors of (5,5) are walls). Sidestep
        // iterates CARDINAL_DIRS skipping N; S/E are walls, W=(4,5) is
        // grass and free.
        assert_eq!(
            *world.get::<GridPos>(a).unwrap(),
            GridPos { x: 4, y: 5 },
            "A should sidestep west — the only open non-bumped neighbor"
        );
        assert_eq!(
            world.get::<Program>(a).unwrap().pc,
            0,
            "still on navigate_to; re-plan happens next tick from new pos"
        );
    }

    #[test]
    fn workers_cannot_step_onto_another_worker() {
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(World {
            tiles: vec![Terrain::Grass; (GRID_W * GRID_H) as usize],
        });
        world.insert_resource(dummy_resource_assets());
        // A is at (5,5) and wants to walk North. B is parked at (5,4) running
        // Wait — so A's destination is permanently occupied this tick.
        let a = world
            .spawn((
                Worker,
                GridPos { x: 5, y: 5 },
                Inventory::default(),
                NavState::default(),
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                Program {
                    instructions: vec![Action::Move(Direction::North)],
                    pc: 0,
                },
            ))
            .id();
        spawn_blocker(&mut world, GridPos { x: 5, y: 4 });

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        assert_eq!(
            *world.get::<GridPos>(a).unwrap(),
            GridPos { x: 5, y: 5 },
            "A should bump off B and stay put"
        );
    }

    // --- bump count + despawn ----------------------------------------------

    #[test]
    fn bump_count_increments_when_move_is_blocked() {
        let mut w = flat_grass_world();
        place(&mut w, 5, 4, Terrain::Wall);
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(w);
        world.insert_resource(dummy_resource_assets());

        let a = world
            .spawn((
                Worker,
                GridPos { x: 5, y: 5 },
                Inventory::default(),
                NavState::default(),
                BumpCount::default(),
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                Program {
                    instructions: vec![Action::Move(Direction::North)],
                    pc: 0,
                },
            ))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        assert_eq!(world.get::<BumpCount>(a).unwrap().0, 1);
        assert_eq!(*world.get::<GridPos>(a).unwrap(), GridPos { x: 5, y: 5 });
    }

    #[test]
    fn bump_count_resets_on_successful_move() {
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(World {
            tiles: vec![Terrain::Grass; (GRID_W * GRID_H) as usize],
        });
        world.insert_resource(dummy_resource_assets());
        let a = world
            .spawn((
                Worker,
                GridPos { x: 5, y: 5 },
                Inventory::default(),
                NavState::default(),
                BumpCount(7),
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                Program {
                    instructions: vec![Action::Move(Direction::East)],
                    pc: 0,
                },
            ))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        assert_eq!(
            world.get::<BumpCount>(a).unwrap().0,
            0,
            "successful step must reset the bump counter"
        );
        assert_eq!(*world.get::<GridPos>(a).unwrap(), GridPos { x: 6, y: 5 });
    }

    // Build a world where the worker at (5,5) bumps a wall every tick, seed
    // it one tick from MAX_BUMPS so the next tick triggers the despawn-spill,
    // run step_workers once, and report what's left at (5,5).
    fn run_despawn_with_inventory(carried: usize) -> (bool, usize) {
        let mut w = flat_grass_world();
        place(&mut w, 5, 4, Terrain::Wall);
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(w);
        world.insert_resource(dummy_resource_assets());

        let queue: VecDeque<ResourceKind> =
            std::iter::repeat_n(ResourceKind::Energy, carried).collect();
        let a = world
            .spawn((
                Worker,
                GridPos { x: 5, y: 5 },
                Inventory { queue },
                NavState::default(),
                BumpCount(MAX_BUMPS - 1),
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                Program {
                    instructions: vec![Action::Move(Direction::North)],
                    pc: 0,
                },
            ))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        let despawned = world.get_entity(a).is_err();
        let spilled = world
            .query::<(&ResourceNode, &GridPos)>()
            .iter(&world)
            .filter(|(_, p)| **p == GridPos { x: 5, y: 5 })
            .count();
        (despawned, spilled)
    }

    #[test]
    fn worker_despawns_and_spills_inventory_at_max_bumps() {
        // Inventory::CAPACITY = 2, so the worker can carry at most 2 items.
        let (despawned, spilled) = run_despawn_with_inventory(Inventory::CAPACITY);
        assert!(despawned, "worker should despawn at MAX_BUMPS");
        assert_eq!(
            spilled,
            Inventory::CAPACITY,
            "expected one ResourceNode per carried item"
        );
    }

    #[test]
    fn empty_inventory_despawn_spills_nothing() {
        let (despawned, spilled) = run_despawn_with_inventory(0);
        assert!(despawned);
        assert_eq!(spilled, 0);
    }

    // --- step_workers pickup contention -------------------------------------

    #[test]
    fn pickup_only_one_worker_consumes_same_tile_energy() {
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(World {
            tiles: vec![Terrain::Grass; (GRID_W * GRID_H) as usize],
        });
        world.insert_resource(dummy_resource_assets());

        let energy_ent = world
            .spawn((ResourceNode { kind: ResourceKind::Energy }, GridPos { x: 5, y: 5 }))
            .id();

        let pickup_prog = || Program {
            instructions: vec![Action::Pickup(None)],
            pc: 0,
        };
        let make_worker = |w: &mut bevy::ecs::world::World| {
            w.spawn((
                Worker,
                GridPos { x: 5, y: 5 },
                Inventory::default(),
                NavState::default(),
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                pickup_prog(),
            ))
            .id()
        };
        let w1 = make_worker(&mut world);
        let w2 = make_worker(&mut world);

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        // Energy entity must be despawned exactly once.
        assert!(
            world.get_entity(energy_ent).is_err(),
            "energy entity should be despawned"
        );
        // Exactly one worker should have picked it up — phantom energy is a bug.
        let e1 = count_kind(world.get::<Inventory>(w1).unwrap(), ResourceKind::Energy);
        let e2 = count_kind(world.get::<Inventory>(w2).unwrap(), ResourceKind::Energy);
        assert_eq!(
            e1 + e2,
            1,
            "expected exactly one worker to grab the energy, got e1={e1} e2={e2}"
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

    // --- parser --------------------------------------------------------------

    #[test]
    fn parse_program_recognizes_navigate_to_each_resource_kind() {
        let src = "\
navigate_to(closest, energy)
navigate_to(closest, grass)
navigate_to(closest, wood)
navigate_to(closest, base)
";
        let actions = parse_program(src).unwrap();
        assert_eq!(
            actions,
            vec![
                Action::NavigateTo(NavQualifier::Closest, Target::Resource(ResourceKind::Energy)),
                Action::NavigateTo(NavQualifier::Closest, Target::Resource(ResourceKind::Grass)),
                Action::NavigateTo(NavQualifier::Closest, Target::Resource(ResourceKind::Wood)),
                Action::NavigateTo(NavQualifier::Closest, Target::Base),
            ]
        );
    }

    #[test]
    fn parse_program_pickup_without_kind_is_any() {
        let actions = parse_program("pickup\n").unwrap();
        assert_eq!(actions, vec![Action::Pickup(None)]);
    }

    #[test]
    fn parse_program_pickup_with_kind_filters() {
        let actions = parse_program(
            "pickup(energy)\npickup(grass)\npickup(wood)\n",
        )
        .unwrap();
        assert_eq!(
            actions,
            vec![
                Action::Pickup(Some(ResourceKind::Energy)),
                Action::Pickup(Some(ResourceKind::Grass)),
                Action::Pickup(Some(ResourceKind::Wood)),
            ]
        );
    }

    #[test]
    fn parse_program_rejects_unknown_pickup_kind() {
        let err = parse_program("pickup(stone)").unwrap_err();
        assert!(err.contains("stone"), "error should mention the bad token: {err}");
    }

    #[test]
    fn parse_program_rejects_unknown_resource_target() {
        let err = parse_program("navigate_to(closest, stone)").unwrap_err();
        assert!(
            err.contains("stone"),
            "error should mention the unknown token: {err}"
        );
    }

    // --- if/else branching --------------------------------------------------

    #[test]
    fn parse_program_compiles_bare_if_to_jump_ir() {
        // `if cond { body }` becomes [JumpUnless(cond, past_body), <body>].
        let actions = parse_program("if carrying(energy) {\ndrop\n}\n").unwrap();
        assert_eq!(
            actions,
            vec![
                Action::JumpUnless(Condition::Carrying(ResourceKind::Energy), 2),
                Action::Drop,
            ]
        );
    }

    #[test]
    fn parse_program_compiles_if_else_to_jump_ir() {
        // `if cond { a } else { b }` becomes:
        //   [JumpUnless(cond, else_start), <a>, Jump(end), <b>]
        let actions = parse_program(
            "if carrying(energy) {\ndrop\n} else {\npickup(energy)\n}\n",
        )
        .unwrap();
        assert_eq!(
            actions,
            vec![
                Action::JumpUnless(Condition::Carrying(ResourceKind::Energy), 3),
                Action::Drop,
                Action::Jump(4),
                Action::Pickup(Some(ResourceKind::Energy)),
            ]
        );
    }

    #[test]
    fn parse_program_compiles_nested_if() {
        // Two nested ifs both targeting the same end address.
        let actions =
            parse_program("if carrying(energy) {\nif carrying(grass) {\ndrop\n}\n}\n").unwrap();
        assert_eq!(
            actions,
            vec![
                Action::JumpUnless(Condition::Carrying(ResourceKind::Energy), 3),
                Action::JumpUnless(Condition::Carrying(ResourceKind::Grass), 3),
                Action::Drop,
            ]
        );
    }

    #[test]
    fn parse_program_rejects_unclosed_if_block() {
        let err = parse_program("if carrying(energy) {\ndrop\n").unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("unclosed")
                || err.to_ascii_lowercase().contains("unmatched"),
            "error should mention the unclosed block: {err}"
        );
    }

    #[test]
    fn parse_program_rejects_dangling_close_brace() {
        let err = parse_program("drop\n}\n").unwrap_err();
        assert!(
            err.contains('}') || err.to_ascii_lowercase().contains("unmatched"),
            "error should mention the stray brace: {err}"
        );
    }

    #[test]
    fn parse_program_rejects_else_without_if() {
        let err = parse_program("drop\n} else {\ndrop\n}\n").unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("else") || err.contains('}'),
            "error should mention the unexpected else: {err}"
        );
    }

    #[test]
    fn parse_program_rejects_unknown_condition() {
        let err = parse_program("if at_base {\ndrop\n}\n").unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("condition") || err.contains("at_base"),
            "error should mention the unknown condition: {err}"
        );
    }

    #[test]
    fn parse_program_compiles_while_to_jump_ir() {
        // `while cond { body }` becomes:
        //   [JumpUnless(cond, end), <body>, Jump(loop_start)]
        // where `end` lands past the back-jump.
        let actions = parse_program("while carrying(energy) {\ndrop\n}\n").unwrap();
        assert_eq!(
            actions,
            vec![
                Action::JumpUnless(Condition::Carrying(ResourceKind::Energy), 3),
                Action::Drop,
                Action::Jump(0),
            ]
        );
    }

    #[test]
    fn parse_program_compiles_while_not_to_jump_ir() {
        // `while not cond { body }` exits when cond becomes TRUE — the same
        // shape `until cond` used to compile to, but expressed via Python's
        // `not` keyword instead of inventing our own loop verb.
        let actions = parse_program("while not carrying(energy) {\npickup(energy)\n}\n").unwrap();
        assert_eq!(
            actions,
            vec![
                Action::JumpIf(Condition::Carrying(ResourceKind::Energy), 3),
                Action::Pickup(Some(ResourceKind::Energy)),
                Action::Jump(0),
            ]
        );
    }

    #[test]
    fn parse_program_compiles_if_not_to_jump_ir() {
        // `if not cond { body }` skips body when cond is TRUE — same logic
        // as `while not`'s exit, but for one-shot branching.
        let actions = parse_program("if not carrying(energy) {\ndrop\n}\n").unwrap();
        assert_eq!(
            actions,
            vec![
                Action::JumpIf(Condition::Carrying(ResourceKind::Energy), 2),
                Action::Drop,
            ]
        );
    }

    #[test]
    fn parse_program_rejects_until_keyword() {
        let err = parse_program("until carrying(energy) {\ndrop\n}\n").unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("until")
                || err.to_ascii_lowercase().contains("unknown"),
            "until should no longer parse: {err}"
        );
    }

    #[test]
    fn parse_program_compiles_break_to_jump_past_loop() {
        // while cond { break }
        // becomes:
        //   [JumpUnless(cond, 3), Jump(3) <- break, Jump(0) <- back-edge]
        // The break jumps past the back-edge (= loop exit).
        let actions = parse_program("while carrying(energy) {\nbreak\n}\n").unwrap();
        assert_eq!(
            actions,
            vec![
                Action::JumpUnless(Condition::Carrying(ResourceKind::Energy), 3),
                Action::Jump(3),
                Action::Jump(0),
            ]
        );
    }

    #[test]
    fn parse_program_compiles_continue_to_jump_to_loop_start() {
        // while cond { continue }
        // becomes:
        //   [JumpUnless(cond, 3), Jump(0) <- continue, Jump(0) <- back-edge]
        // continue jumps to the loop's top (where cond is re-evaluated).
        let actions = parse_program("while carrying(energy) {\ncontinue\n}\n").unwrap();
        assert_eq!(
            actions,
            vec![
                Action::JumpUnless(Condition::Carrying(ResourceKind::Energy), 3),
                Action::Jump(0),
                Action::Jump(0),
            ]
        );
    }

    #[test]
    fn parse_program_compiles_break_inside_nested_if_to_outer_loop_end() {
        // while cond1 { if cond2 { break } }
        // The break inside the `if` should still target the while's end,
        // not just the if's end. Layout:
        //   0: JumpUnless(energy, 4)   <- while top; exit at end of loop
        //   1: JumpUnless(grass, 3)    <- if top; skip body lands at back-edge
        //   2: Jump(4)                 <- break (targets while end)
        //   3: Jump(0)                 <- back-edge to while top
        //   4: <while end>
        let actions = parse_program(
            "while carrying(energy) {\nif carrying(grass) {\nbreak\n}\n}\n",
        )
        .unwrap();
        assert_eq!(
            actions,
            vec![
                Action::JumpUnless(Condition::Carrying(ResourceKind::Energy), 4),
                Action::JumpUnless(Condition::Carrying(ResourceKind::Grass), 3),
                Action::Jump(4),
                Action::Jump(0),
            ]
        );
    }

    #[test]
    fn parse_program_rejects_break_outside_loop() {
        let err = parse_program("break\n").unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("break"),
            "error should mention break: {err}"
        );
    }

    #[test]
    fn parse_program_rejects_continue_outside_loop() {
        let err = parse_program("continue\n").unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("continue"),
            "error should mention continue: {err}"
        );
    }

    #[test]
    fn parse_program_rejects_break_inside_if_without_enclosing_loop() {
        // if alone (no enclosing while) — break has nowhere to target.
        let err = parse_program("if carrying(energy) {\nbreak\n}\n").unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("break"),
            "error should mention break: {err}"
        );
    }

    #[test]
    fn parse_program_rejects_unclosed_while() {
        let err = parse_program("while carrying(energy) {\ndrop\n").unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("unclosed")
                || err.to_ascii_lowercase().contains("while"),
            "error should mention the unclosed loop: {err}"
        );
    }

    #[test]
    fn parse_program_unclosed_if_reports_opening_line() {
        let err = parse_program("drop\nif carrying(energy) {\ndrop\n").unwrap_err();
        // The `if` is on line 2; the error should point at it, not the EOF.
        assert!(
            err.contains("line 2") || err.contains("2"),
            "error should mention line 2: {err}"
        );
    }

    // --- syntax highlight tokenizer (pure) ----------------------------------

    #[test]
    fn classify_identifier_recognizes_keywords() {
        for kw in ["if", "else", "while", "break", "continue", "not"] {
            assert_eq!(classify_identifier(kw), TokenKind::Keyword, "{kw}");
        }
    }

    #[test]
    fn classify_identifier_recognizes_verbs() {
        for v in [
            "pickup",
            "grab",
            "take",
            "drop",
            "deposit",
            "deliver",
            "wait",
            "noop",
            "navigate_to",
            "goto",
            "nav",
        ] {
            assert_eq!(classify_identifier(v), TokenKind::Verb, "{v}");
        }
    }

    #[test]
    fn classify_identifier_recognizes_arg_identifiers() {
        for id in [
            "energy", "grass", "wood", "base", "carrying", "closest", "n", "s", "e", "w", "north",
            "south", "east", "west", "up", "down", "left", "right",
        ] {
            assert_eq!(classify_identifier(id), TokenKind::Identifier, "{id}");
        }
    }

    #[test]
    fn classify_identifier_is_case_insensitive() {
        assert_eq!(classify_identifier("IF"), TokenKind::Keyword);
        assert_eq!(classify_identifier("PickUp"), TokenKind::Verb);
        assert_eq!(classify_identifier("Energy"), TokenKind::Identifier);
    }

    #[test]
    fn classify_identifier_unknown_for_random_word() {
        assert_eq!(classify_identifier("foo_bar"), TokenKind::Unknown);
    }

    #[test]
    fn tokenize_classifies_a_full_if_line() {
        let toks = tokenize("if carrying(energy) {");
        let kinds: Vec<TokenKind> = toks.iter().map(|(_, k)| *k).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::Keyword,    // if
                TokenKind::Whitespace, // ' '
                TokenKind::Identifier, // carrying
                TokenKind::Punctuation,// (
                TokenKind::Identifier, // energy
                TokenKind::Punctuation,// )
                TokenKind::Whitespace, // ' '
                TokenKind::Punctuation,// {
            ]
        );
    }

    #[test]
    fn tokenize_treats_hash_to_eol_as_comment() {
        let toks = tokenize("pickup(energy) # grab one\nnext");
        // Find the comment token and verify its text.
        let comment = toks.iter().find(|(_, k)| *k == TokenKind::Comment).unwrap();
        assert_eq!(comment.0, "# grab one");
        // The newline after the comment should be Whitespace, and `next`
        // should be an identifier (unknown name in our DSL).
        let post: Vec<_> = toks
            .iter()
            .skip_while(|(_, k)| *k != TokenKind::Comment)
            .skip(1)
            .collect();
        assert!(post.first().is_some_and(|(_, k)| *k == TokenKind::Whitespace));
    }

    #[test]
    fn tokenize_handles_navigate_to_underscore_as_one_token() {
        let toks = tokenize("navigate_to(closest, base)");
        assert_eq!(toks[0].0, "navigate_to");
        assert_eq!(toks[0].1, TokenKind::Verb);
    }

    #[test]
    fn tokenize_does_not_panic_on_non_ascii_input() {
        // Multi-byte UTF-8 outside identifiers used to slice into the middle
        // of a codepoint; this test pins the fix so a pasted emoji or accent
        // in a comment doesn't crash the editor.
        let toks = tokenize("# café 🎮\npickup");
        // Just asserting we got a usable token list back is enough — the
        // pre-fix panic happened during slicing.
        assert!(toks.iter().any(|(_, k)| *k == TokenKind::Comment));
        assert!(toks.iter().any(|(t, k)| *t == "pickup" && *k == TokenKind::Verb));
    }

    // --- evaluate_condition (pure) ------------------------------------------

    #[test]
    fn evaluate_carrying_true_when_kind_in_queue() {
        let mut inv = Inventory::default();
        inv.push(ResourceKind::Energy);
        assert!(evaluate_condition(
            Condition::Carrying(ResourceKind::Energy),
            &inv,
        ));
    }

    #[test]
    fn evaluate_carrying_false_when_kind_absent() {
        let mut inv = Inventory::default();
        inv.push(ResourceKind::Grass);
        assert!(!evaluate_condition(
            Condition::Carrying(ResourceKind::Energy),
            &inv,
        ));
    }

    // --- step_workers jump execution ----------------------------------------

    fn spawn_jump_test_worker(
        world: &mut bevy::ecs::world::World,
        inv: Inventory,
        instructions: Vec<Action>,
    ) -> Entity {
        world
            .spawn((
                Worker,
                GridPos { x: 5, y: 5 },
                inv,
                NavState::default(),
                Facing {
                    prev_yaw: 0.0,
                    current_yaw: 0.0,
                },
                Program {
                    instructions,
                    pc: 0,
                },
            ))
            .id()
    }

    fn jump_test_world() -> bevy::ecs::world::World {
        let mut world = bevy::ecs::world::World::new();
        world.insert_resource(World {
            tiles: vec![Terrain::Grass; (GRID_W * GRID_H) as usize],
        });
        world.insert_resource(dummy_resource_assets());
        world
    }

    #[test]
    fn jump_unless_redirects_pc_when_condition_false() {
        // Empty inventory → carrying(energy) is false → JumpUnless takes the
        // jump to pc=2, skipping the Wait at pc=1.
        let mut world = jump_test_world();
        let w = spawn_jump_test_worker(
            &mut world,
            Inventory::default(),
            vec![
                Action::JumpUnless(Condition::Carrying(ResourceKind::Energy), 2),
                Action::Wait,
                Action::Drop,
            ],
        );

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        assert_eq!(world.get::<Program>(w).unwrap().pc, 2);
    }

    #[test]
    fn jump_unless_falls_through_when_condition_true() {
        // Inventory carries energy → JumpUnless does NOT jump → pc advances
        // by 1 to the Wait.
        let mut world = jump_test_world();
        let mut inv = Inventory::default();
        inv.push(ResourceKind::Energy);
        let w = spawn_jump_test_worker(
            &mut world,
            inv,
            vec![
                Action::JumpUnless(Condition::Carrying(ResourceKind::Energy), 2),
                Action::Wait,
                Action::Drop,
            ],
        );

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        assert_eq!(world.get::<Program>(w).unwrap().pc, 1);
    }

    #[test]
    fn jump_sets_pc_unconditionally() {
        let mut world = jump_test_world();
        let w = spawn_jump_test_worker(
            &mut world,
            Inventory::default(),
            vec![Action::Jump(2), Action::Wait, Action::Drop],
        );

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        assert_eq!(world.get::<Program>(w).unwrap().pc, 2);
    }

    #[test]
    fn jump_if_redirects_pc_when_condition_true() {
        // Inverse of jump_unless: JumpIf takes the jump when cond is TRUE.
        let mut world = jump_test_world();
        let mut inv = Inventory::default();
        inv.push(ResourceKind::Energy);
        let w = spawn_jump_test_worker(
            &mut world,
            inv,
            vec![
                Action::JumpIf(Condition::Carrying(ResourceKind::Energy), 2),
                Action::Wait,
                Action::Drop,
            ],
        );

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        assert_eq!(world.get::<Program>(w).unwrap().pc, 2);
    }

    #[test]
    fn jump_if_falls_through_when_condition_false() {
        let mut world = jump_test_world();
        let w = spawn_jump_test_worker(
            &mut world,
            Inventory::default(),
            vec![
                Action::JumpIf(Condition::Carrying(ResourceKind::Energy), 2),
                Action::Wait,
                Action::Drop,
            ],
        );

        let mut schedule = Schedule::default();
        schedule.add_systems(step_workers);
        schedule.run(&mut world);

        assert_eq!(world.get::<Program>(w).unwrap().pc, 1);
    }

    // --- inventory queue ----------------------------------------------------

    #[test]
    fn inventory_starts_empty() {
        let inv = Inventory::default();
        assert!(inv.is_empty());
    }

    #[test]
    fn inventory_push_holds_up_to_capacity() {
        let mut inv = Inventory::default();
        inv.push(ResourceKind::Energy);
        inv.push(ResourceKind::Grass);
        assert_eq!(
            inv.queue.iter().copied().collect::<Vec<_>>(),
            vec![ResourceKind::Energy, ResourceKind::Grass]
        );
    }

    #[test]
    fn inventory_push_evicts_oldest_when_full() {
        let mut inv = Inventory::default();
        inv.push(ResourceKind::Energy);
        inv.push(ResourceKind::Grass);
        inv.push(ResourceKind::Wood);
        // Energy (oldest) was evicted; queue now [Grass, Wood].
        assert_eq!(
            inv.queue.iter().copied().collect::<Vec<_>>(),
            vec![ResourceKind::Grass, ResourceKind::Wood]
        );
    }

    #[test]
    fn inventory_push_returns_none_within_capacity() {
        let mut inv = Inventory::default();
        assert_eq!(inv.push(ResourceKind::Energy), None);
        assert_eq!(inv.push(ResourceKind::Grass), None);
    }

    #[test]
    fn inventory_push_returns_evicted_kind_when_full() {
        let mut inv = Inventory::default();
        inv.push(ResourceKind::Energy);
        inv.push(ResourceKind::Grass);
        assert_eq!(inv.push(ResourceKind::Wood), Some(ResourceKind::Energy));
    }

    #[test]
    fn inventory_drain_into_clears_queue_and_increments_counts() {
        let mut inv = Inventory::default();
        inv.push(ResourceKind::Energy);
        inv.push(ResourceKind::Grass);
        let mut counts = [0u32; ResourceKind::COUNT];
        inv.drain_into(&mut counts);
        assert!(inv.is_empty());
        assert_eq!(counts[ResourceKind::Energy as usize], 1);
        assert_eq!(counts[ResourceKind::Grass as usize], 1);
        assert_eq!(counts[ResourceKind::Wood as usize], 0);
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
