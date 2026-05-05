use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateAtom {
    pub name: String,
    pub element: String,
    pub coords: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueTemplate {
    pub name: String,
    pub atoms: Vec<TemplateAtom>,
    pub bonds: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResidueLibrary {
    pub templates: std::collections::HashMap<String, ResidueTemplate>,
}

impl ResidueLibrary {
    pub fn new() -> Self {
        Self {
            templates: std::collections::HashMap::new(),
        }
    }

    pub fn insert(&mut self, template: ResidueTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    pub fn get(&self, name: &str) -> Option<&ResidueTemplate> {
        self.templates.get(name)
    }

    pub fn new_standard() -> Self {
        let mut lib = Self::new();
        
        // Define ALA template
        let ala = ResidueTemplate {
            name: "ALA".to_string(),
            atoms: vec![
                TemplateAtom { name: "N".to_string(), element: "N".to_string(), coords: [-1.444, -0.596, 0.968] },
                TemplateAtom { name: "CA".to_string(), element: "C".to_string(), coords: [-0.194, -0.546, 0.198] },
                TemplateAtom { name: "CB".to_string(), element: "C".to_string(), coords: [-0.584, -0.626, -1.282] },
                TemplateAtom { name: "C".to_string(), element: "C".to_string(), coords: [0.716, 0.684, 0.478] },
                TemplateAtom { name: "O".to_string(), element: "O".to_string(), coords: [1.506, 1.084, -0.362] },
            ],
            bonds: vec![
                ("N".to_string(), "CA".to_string()),
                ("CA".to_string(), "CB".to_string()),
                ("CA".to_string(), "C".to_string()),
                ("C".to_string(), "O".to_string()),
            ],
        };
        
        lib.insert(ala);
        lib
    }
}
