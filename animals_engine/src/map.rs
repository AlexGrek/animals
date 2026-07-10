use rand::Rng;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Terrain {
    Grass,
    Road,
    Water,
    Rock,
}



#[derive(Clone, Debug)]
pub struct Map {
    pub width: i32,
    pub height: i32,
    pub tiles: Vec<Terrain>,
}

impl Map {
    pub fn new(width: i32, height: i32) -> Self {
        let mut map = Self {
            width,
            height,
            tiles: vec![Terrain::Grass; (width * height) as usize],
        };
        map.generate();
        map
    }

    pub fn get_terrain(&self, x: i32, y: i32) -> Terrain {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            Terrain::Rock // Treat out of bounds as rock
        } else {
            self.tiles[(y * self.width + x) as usize]
        }
    }

    pub fn set_terrain(&mut self, x: i32, y: i32, terrain: Terrain) {
        if x >= 0 && x < self.width && y >= 0 && y < self.height {
            self.tiles[(y * self.width + x) as usize] = terrain;
        }
    }

    fn generate(&mut self) {
        let mut rng = rand::thread_rng();

        // Scale feature COUNTS (not radii) by field area relative to the
        // 100x100 training baseline, so a larger field isn't mostly empty
        // grass. At 100x100 this is exactly 1.0, so counts are unchanged.
        let area_scale = ((self.width * self.height) as f32 / (100.0 * 100.0)).max(1.0);

        // 1. Generate Roads (Large random-shaped objects)
        let num_roads = (rng.gen_range(2..=5) as f32 * area_scale).round() as i32;
        for _ in 0..num_roads {
            let mut px = rng.gen_range(0..self.width) as f32;
            let mut py = rng.gen_range(0..self.height) as f32;
            let mut angle = rng.gen_range(0.0..std::f32::consts::TAU);
            let length = rng.gen_range(20..80);
            let radius = rng.gen_range(3.0..6.0);

            for _ in 0..length {
                self.draw_circle(px as i32, py as i32, radius as i32, Terrain::Road);
                angle += rng.gen_range(-0.5..0.5);
                px += angle.cos() * 2.0;
                py += angle.sin() * 2.0;
            }
        }

        // 2. Generate Water ponds (small ponds, diameter 6-20 -> radius 3-10)
        let num_ponds = (rng.gen_range(4..=10) as f32 * area_scale).round() as i32;
        for _ in 0..num_ponds {
            let cx = rng.gen_range(0..self.width);
            let cy = rng.gen_range(0..self.height);
            let radius = rng.gen_range(3..=10);
            self.draw_circle(cx, cy, radius, Terrain::Water);
        }

        // 3. Generate Rocks (circular-like, radius 3-16)
        let num_rocks = (rng.gen_range(5..=15) as f32 * area_scale).round() as i32;
        for _ in 0..num_rocks {
            let cx = rng.gen_range(0..self.width);
            let cy = rng.gen_range(0..self.height);
            let radius = rng.gen_range(3..=16);
            self.draw_circle(cx, cy, radius, Terrain::Rock);
        }
    }

    fn draw_circle(&mut self, cx: i32, cy: i32, radius: i32, terrain: Terrain) {
        let r_sq = radius * radius;
        for y in (cy - radius)..=(cy + radius) {
            for x in (cx - radius)..=(cx + radius) {
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy <= r_sq {
                    self.set_terrain(x, y, terrain);
                }
            }
        }
    }
}
