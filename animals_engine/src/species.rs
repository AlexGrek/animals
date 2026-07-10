use crate::map::Terrain;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Snake,
    Prey,
    Amphibia,
}

impl Species {
    pub fn speed_on(&self, terrain: Terrain) -> f32 {
        match self {
            Species::Snake | Species::Prey => match terrain {
                Terrain::Grass => 0.8,
                Terrain::Road => 1.0,
                Terrain::Water => 0.2,
                Terrain::Rock => 0.0,
            },
            Species::Amphibia => match terrain {
                Terrain::Grass => 0.6,
                Terrain::Road => 0.8,
                Terrain::Water => 1.0,
                Terrain::Rock => 0.0,
            },
        }
    }
}
