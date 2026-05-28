use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};

const GRID_W: i32 = 20;
const GRID_H: i32 = 15;
const START: GridPos = GridPos { x: 5, y: 5 };

#[derive(Copy, Clone, Debug)]
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
enum Action {
    Move(Direction),
    Wait,
    Pickup,
    Drop,
}

#[derive(Component, Copy, Clone, Debug)]
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
        if x < 0 || x >= GRID_W || y < 0 || y >= GRID_H {
            return None;
        }
        Some(self.tiles[Self::idx(x, y)])
    }

    fn generate() -> Self {
        let mut tiles = vec![Terrain::Grass; (GRID_W * GRID_H) as usize];
        for y in 0..GRID_H {
            for x in 0..GRID_W {
                let h = tile_hash(x, y);
                let t = match h % 100 {
                    0..=4 => Terrain::Stone,
                    5..=9 => Terrain::Dirt,
                    10..=12 => Terrain::Water,
                    13..=15 => Terrain::Wall,
                    _ => Terrain::Grass,
                };
                tiles[Self::idx(x, y)] = t;
            }
        }
        // Guarantee worker's starting cell is walkable.
        tiles[Self::idx(START.x, START.y)] = Terrain::Grass;
        Self { tiles }
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
# one instruction per line
# N S E W Wait  (case-insensitive)
E
E
N
N
W
W
S
S
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
            (advance_tick, step_workers, update_move_anim).chain(),
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

// Grid cell (gx, gy) maps to world (x, z). Y is up in 3D and reserved for height.
fn grid_to_world(gx: i32, gy: i32) -> (f32, f32) {
    let ox = -(GRID_W as f32) / 2.0 + 0.5;
    let oz = -(GRID_H as f32) / 2.0 + 0.5;
    (ox + gx as f32, oz + gy as f32)
}

fn parse_program(src: &str) -> Result<Vec<Action>, String> {
    let mut out = Vec::new();
    for (i, line) in src.lines().enumerate() {
        let trimmed = line.split('#').next().unwrap_or("").trim();
        if trimmed.is_empty() {
            continue;
        }
        let action = match trimmed.to_ascii_lowercase().as_str() {
            "n" | "north" | "up" => Action::Move(Direction::North),
            "s" | "south" | "down" => Action::Move(Direction::South),
            "e" | "east" | "right" => Action::Move(Direction::East),
            "w" | "west" | "left" => Action::Move(Direction::West),
            "wait" | "noop" => Action::Wait,
            "pickup" | "grab" | "take" => Action::Pickup,
            "drop" | "deposit" | "deliver" => Action::Drop,
            other => return Err(format!("line {}: unknown instruction '{}'", i + 1, other)),
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

fn step_workers(
    world: Res<World>,
    mut commands: Commands,
    energy_q: Query<(Entity, &GridPos), (With<EnergyNode>, Without<Worker>, Without<Base>)>,
    mut base_q: Query<(&GridPos, &mut Base), (Without<Worker>, Without<EnergyNode>)>,
    mut workers: Query<
        (&mut GridPos, &mut Program, &mut Facing, &mut Inventory),
        (With<Worker>, Without<EnergyNode>, Without<Base>),
    >,
) {
    for (mut pos, mut prog, mut facing, mut inv) in &mut workers {
        if prog.instructions.is_empty() {
            continue;
        }
        let action = prog.instructions[prog.pc];
        match action {
            Action::Move(dir) => {
                // Always turn to face the direction — even if the move is
                // blocked, the worker turns toward the obstacle.
                facing.prev_yaw = facing.current_yaw;
                facing.current_yaw = dir.yaw();
                let (dx, dy) = dir.delta();
                let nx = pos.x + dx;
                let ny = pos.y + dy;
                if matches!(world.get(nx, ny), Some(t) if t.passable()) {
                    pos.x = nx;
                    pos.y = ny;
                }
            }
            Action::Pickup => {
                for (ent, epos) in &energy_q {
                    if epos.x == pos.x && epos.y == pos.y {
                        commands.entity(ent).despawn();
                        inv.energy += 1;
                        break;
                    }
                }
            }
            Action::Drop => {
                if inv.energy > 0 {
                    for (bpos, mut base) in &mut base_q {
                        if bpos.x == pos.x && bpos.y == pos.y {
                            base.stored += inv.energy;
                            inv.energy = 0;
                            break;
                        }
                    }
                }
            }
            Action::Wait => {}
        }
        prog.pc = (prog.pc + 1) % prog.instructions.len();
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

// Runs in FixedUpdate after step_workers. Rolls prev=current and writes the
// new current whenever GridPos changed this tick.
fn update_move_anim(mut q: Query<(&GridPos, &mut MoveAnim), Changed<GridPos>>) {
    for (pos, mut anim) in &mut q {
        let (x, z) = grid_to_world(pos.x, pos.y);
        anim.prev = anim.current;
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
            ui.monospace("N S E W Wait Pickup Drop  (# = comment)");
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
